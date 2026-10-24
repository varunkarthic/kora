//! Raw Vulkan reference paths on the same physical device as the KORA Runtime: the matched compute
//! workloads, buffer copies, plus KORA transfer-engine, staging-ring, and command-pool costs.

use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use kora::DeviceId;
use kora::device::{DeviceInfo, MemoryTopology};
use kora::memory::{
    MappedPtr, MemoryManager, MemoryPolicy, MemoryPool, MemoryPoolConfig, MemoryTypeRequirements, SharedPool,
    StagingOptions, StagingRing, TransferBuffer, TransferEngine, TransferOptions,
};
use kora::vulkan::ash::vk;
use kora::vulkan::{
    CommandPoolSet, Device, DeviceOptions, InstanceOptions, Loader, PhysicalDeviceInfo, QueueContext,
    QueueContextOptions, device_info,
};

use crate::gpu::{
    COMPUTE_SIZES, MATRIX_DIMENSIONS, REDUCTION, SAXPY, SCAN, STENCIL, TILED_MATMUL, VECTOR_ADD, VECTOR_SIZES,
};
use crate::report::{
    Failure, GpuDevice, Measurement, Options, Outcome, run, run_checked, run_timestamps, run_transfer,
};

fn vk_failed(operation: &str) -> impl FnOnce(vk::Result) -> Failure + '_ {
    move |result| Failure::Failed(format!("raw Vulkan {operation} failed: {result:?}"))
}

/// A host-visible (device-local when available) buffer mapped for its whole lifetime.
struct HostBuffer {
    device: Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
    allocation_size: u64,
    mapped: NonNull<u8>,
    coherent: bool,
}

// SAFETY: the mapping is owned by this buffer and only accessed through `&self`/`&mut self`
// methods; Vulkan handles are plain identifiers.
unsafe impl Send for HostBuffer {}

impl HostBuffer {
    fn new(device: &Device, info: &DeviceInfo, size: u64) -> Outcome<HostBuffer> {
        let raw = device.raw();
        let create = vk::BufferCreateInfo::default()
            .size(size)
            .usage(
                vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::STORAGE_BUFFER,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: valid device and create info.
        let buffer = unsafe { raw.create_buffer(&create, None) }.map_err(vk_failed("buffer creation"))?;
        // SAFETY: `buffer` was created on this device.
        let requirements = unsafe { raw.get_buffer_memory_requirements(buffer) };
        let destroy_buffer = || {
            // SAFETY: the buffer is unused and owned here.
            unsafe { raw.destroy_buffer(buffer, None) };
        };
        let Ok(index) = MemoryManager::select_memory_type_from(
            &info.memory_types,
            &info.memory_type_preferences,
            MemoryPolicy::DevicePreferred,
            requirements.memory_type_bits,
            &MemoryTypeRequirements {
                device_local: true,
                host_visible: true,
                ..MemoryTypeRequirements::default()
            },
        ) else {
            destroy_buffer();
            return Err(Failure::Unavailable(
                "GPU benchmarks unavailable: no device-local host-visible memory for the raw Vulkan reference"
                    .into(),
            ));
        };
        let memory = match device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(index),
        ) {
            Ok(memory) => memory,
            Err(error) => {
                destroy_buffer();
                return Err(Failure::Failed(format!("raw Vulkan allocation failed: {error}")));
            }
        };
        let free = || {
            destroy_buffer();
            // SAFETY: the memory is unused and owned here.
            unsafe { raw.free_memory(memory, None) };
        };
        // SAFETY: memory satisfies the buffer's requirements and is unbound.
        if let Err(result) = unsafe { raw.bind_buffer_memory(buffer, memory, 0) } {
            free();
            return Err(vk_failed("buffer bind")(result));
        }
        // SAFETY: host-visible memory, mapped once for its whole size.
        let mapped =
            match unsafe { raw.map_memory(memory, 0, requirements.size, vk::MemoryMapFlags::empty()) } {
                Ok(pointer) => pointer,
                Err(result) => {
                    free();
                    return Err(vk_failed("memory map")(result));
                }
            };
        Ok(HostBuffer {
            device: device.clone(),
            buffer,
            memory,
            size,
            allocation_size: requirements.size,
            mapped: NonNull::new(mapped.cast()).expect("vkMapMemory returned a non-null pointer"),
            coherent: info.memory_types[index as usize].host_coherent,
        })
    }

    fn bytes(&self) -> &[u8] {
        // SAFETY: the whole allocation is mapped while `self` lives; callers synchronize with the
        // device (fence waits and `sync`) before reading.
        unsafe { std::slice::from_raw_parts(self.mapped.as_ptr(), self.size as usize) }
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: as in `bytes`, and `&mut self` excludes other host access.
        unsafe { std::slice::from_raw_parts_mut(self.mapped.as_ptr(), self.size as usize) }
    }

    fn fill<T: kora::bytemuck::Pod>(&mut self, value: T) {
        for chunk in kora::bytemuck::cast_slice_mut::<u8, T>(self.bytes_mut()) {
            *chunk = value;
        }
    }

    fn values<T: kora::bytemuck::Pod>(&self) -> &[T] {
        kora::bytemuck::cast_slice(self.bytes())
    }

    /// Flushes host writes (or invalidates for host reads) on non-coherent memory.
    fn sync(&self, flush: bool) -> Outcome {
        if self.coherent {
            return Ok(());
        }
        let range = vk::MappedMemoryRange::default()
            .memory(self.memory)
            .offset(0)
            .size(vk::WHOLE_SIZE);
        let raw = self.device.raw();
        // SAFETY: the memory is mapped and the range covers it.
        let result = unsafe {
            if flush {
                raw.flush_mapped_memory_ranges(&[range])
            } else {
                raw.invalidate_mapped_memory_ranges(&[range])
            }
        };
        result.map_err(vk_failed(if flush { "flush" } else { "invalidate" }))
    }

    fn view(&self) -> TransferBuffer {
        TransferBuffer {
            buffer: self.buffer,
            size: self.size,
            memory: self.memory,
            memory_allocation_size: self.allocation_size,
            // SAFETY: the mapping lives as long as `self`, which outlives every use of the view.
            mapped: Some(unsafe { MappedPtr::from_raw(self.mapped) }),
            host_visible: true,
            host_coherent: self.coherent,
            ..TransferBuffer::default()
        }
    }
}

impl Drop for HostBuffer {
    fn drop(&mut self) {
        let raw = self.device.raw();
        // SAFETY: no submission uses the buffer any more (every benchmark waits for its fence).
        unsafe {
            raw.unmap_memory(self.memory);
            raw.destroy_buffer(self.buffer, None);
            raw.free_memory(self.memory, None);
        }
    }
}

/// One compute pipeline with its own descriptor set, command buffer, fence, and timestamp queries.
struct RawCompute {
    device: Device,
    queue: vk::Queue,
    timestamp_period: f64,
    timestamp_bits: u32,
    shader: vk::ShaderModule,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    command_pool: vk::CommandPool,
    command: vk::CommandBuffer,
    fence: vk::Fence,
    query_pool: vk::QueryPool,
}

impl RawCompute {
    fn new(device: &Device, spirv: &[u8], buffers: &[&HostBuffer], push_size: u32) -> Outcome<RawCompute> {
        let raw = device.raw();
        let physical = device.physical();
        let family = device.compute_queue_family();
        let mut runner = RawCompute {
            device: device.clone(),
            queue: device.compute_queue(),
            timestamp_period: f64::from(physical.properties.limits.timestamp_period),
            timestamp_bits: physical.queue_families[family as usize].timestamp_valid_bits,
            shader: vk::ShaderModule::null(),
            set_layout: vk::DescriptorSetLayout::null(),
            layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_set: vk::DescriptorSet::null(),
            command_pool: vk::CommandPool::null(),
            command: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            query_pool: vk::QueryPool::null(),
        };
        let words =
            kora::kernel::spirv::words_from_bytes(spirv).map_err(crate::report::failed("raw shader"))?;
        // SAFETY: (all calls below) valid device, create infos that reference live locals, and
        // handles stored in `runner`, whose Drop destroys whatever was created on early return.
        unsafe {
            runner.shader = raw
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
                .map_err(vk_failed("shader-module creation"))?;
            let bindings: Vec<_> = (0..buffers.len() as u32)
                .map(|binding| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(binding)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            runner.set_layout = raw
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .map_err(vk_failed("descriptor-layout creation"))?;
            let push_ranges = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .size(push_size)];
            let set_layouts = [runner.set_layout];
            runner.layout = raw
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&set_layouts)
                        .push_constant_ranges(&push_ranges),
                    None,
                )
                .map_err(vk_failed("pipeline-layout creation"))?;
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(runner.shader)
                .name(c"main");
            runner.pipeline = raw
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(stage)
                        .layout(runner.layout)],
                    None,
                )
                .map_err(|(_, result)| vk_failed("compute-pipeline creation")(result))?[0];
            let pool_sizes = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: buffers.len() as u32,
            }];
            runner.descriptor_pool = raw
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(1)
                        .pool_sizes(&pool_sizes),
                    None,
                )
                .map_err(vk_failed("descriptor-pool creation"))?;
            runner.descriptor_set = raw
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(runner.descriptor_pool)
                        .set_layouts(&set_layouts),
                )
                .map_err(vk_failed("descriptor-set allocation"))?[0];
            let infos: Vec<[vk::DescriptorBufferInfo; 1]> = buffers
                .iter()
                .map(|b| [vk::DescriptorBufferInfo::default().buffer(b.buffer).range(b.size)])
                .collect();
            let writes: Vec<_> = infos
                .iter()
                .enumerate()
                .map(|(binding, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(runner.descriptor_set)
                        .dst_binding(binding as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(info)
                })
                .collect();
            raw.update_descriptor_sets(&writes, &[]);
            runner.command_pool = raw
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                        .queue_family_index(family),
                    None,
                )
                .map_err(vk_failed("command-pool creation"))?;
            runner.command = raw
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(runner.command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .map_err(vk_failed("command-buffer allocation"))?[0];
            runner.fence = raw
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(vk_failed("fence creation"))?;
            if runner.timestamp_bits != 0 {
                runner.query_pool = raw
                    .create_query_pool(
                        &vk::QueryPoolCreateInfo::default()
                            .query_type(vk::QueryType::TIMESTAMP)
                            .query_count(2),
                        None,
                    )
                    .unwrap_or(vk::QueryPool::null());
            }
        }
        Ok(runner)
    }

    /// Records, submits, and waits for one dispatch; returns its device time when `timestamp`.
    fn dispatch(&self, groups: [u32; 2], push: &[u8], timestamp: bool) -> Outcome<Option<u64>> {
        let raw = self.device.raw();
        let command = self.command;
        let timed = timestamp && self.query_pool != vk::QueryPool::null();
        // SAFETY: the command buffer, pipeline, set, and queries belong to this runner and the
        // previous submission finished (fence wait) before it is reset.
        unsafe {
            raw.reset_command_buffer(command, vk::CommandBufferResetFlags::empty())
                .map_err(vk_failed("command reset"))?;
            raw.begin_command_buffer(
                command,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(vk_failed("command begin"))?;
            if timed {
                raw.cmd_reset_query_pool(command, self.query_pool, 0, 2);
                raw.cmd_write_timestamp(command, vk::PipelineStageFlags::TOP_OF_PIPE, self.query_pool, 0);
            }
            raw.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::HOST_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)],
                &[],
                &[],
            );
            raw.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            raw.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[self.descriptor_set],
                &[],
            );
            raw.cmd_push_constants(command, self.layout, vk::ShaderStageFlags::COMPUTE, 0, push);
            raw.cmd_dispatch(command, groups[0], groups[1], 1);
            if timed {
                raw.cmd_write_timestamp(
                    command,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    self.query_pool,
                    1,
                );
            }
            raw.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ)],
                &[],
                &[],
            );
            raw.end_command_buffer(command)
                .map_err(vk_failed("command end"))?;
            let commands = [command];
            raw.queue_submit(
                self.queue,
                &[vk::SubmitInfo::default().command_buffers(&commands)],
                self.fence,
            )
            .map_err(vk_failed("dispatch submit"))?;
            raw.wait_for_fences(&[self.fence], true, u64::MAX)
                .map_err(vk_failed("dispatch wait"))?;
            raw.reset_fences(&[self.fence])
                .map_err(vk_failed("fence reset"))?;
        }
        if !timed {
            return Ok(None);
        }
        let mut stamps = [0_u64; 2];
        // SAFETY: both queries were written by the completed submission.
        let read = unsafe {
            raw.get_query_pool_results(self.query_pool, 0, &mut stamps, vk::QueryResultFlags::TYPE_64)
        };
        if read.is_err() {
            return Ok(None);
        }
        let mut ticks = stamps[1].wrapping_sub(stamps[0]);
        if self.timestamp_bits < 64 {
            ticks &= (1_u64 << self.timestamp_bits) - 1;
        }
        Ok(Some((ticks as f64 * self.timestamp_period).round() as u64))
    }
}

impl Drop for RawCompute {
    fn drop(&mut self) {
        let raw = self.device.raw();
        // SAFETY: every submission waited for its fence; null handles are ignored by Vulkan.
        unsafe {
            raw.destroy_query_pool(self.query_pool, None);
            raw.destroy_fence(self.fence, None);
            raw.destroy_command_pool(self.command_pool, None);
            raw.destroy_descriptor_pool(self.descriptor_pool, None);
            raw.destroy_pipeline(self.pipeline, None);
            raw.destroy_pipeline_layout(self.layout, None);
            raw.destroy_descriptor_set_layout(self.set_layout, None);
            raw.destroy_shader_module(self.shader, None);
        }
    }
}

/// Collects the first error of an infallible timing closure.
#[derive(Default)]
struct Errors(Option<Failure>);

impl Errors {
    fn note<T>(&mut self, outcome: Outcome<T>) -> Option<T> {
        match outcome {
            Ok(value) => Some(value),
            Err(failure) => {
                self.0.get_or_insert(failure);
                None
            }
        }
    }
    fn check(&mut self) -> Outcome {
        self.0.take().map_or(Ok(()), Err)
    }
}

fn add_raw_vector_add(
    results: &mut Vec<Measurement>,
    options: &Options,
    device: &Device,
    info: &DeviceInfo,
    report: &mut GpuDevice,
) -> Outcome {
    for count in VECTOR_SIZES {
        let bytes = u64::from(count) * 4;
        let mut a = HostBuffer::new(device, info, bytes)?;
        let mut b = HostBuffer::new(device, info, bytes)?;
        let output = HostBuffer::new(device, info, bytes)?;
        a.fill(3_u32);
        b.fill(7_u32);
        a.sync(true)?;
        b.sync(true)?;
        let runner = RawCompute::new(device, VECTOR_ADD, &[&a, &b, &output], 4)?;
        let push = count.to_le_bytes();
        let groups = [count.div_ceil(64), 1];
        let mut errors = Errors::default();
        // The C++ reference records timestamps on every vector-add submission when it can.
        let mut profiled = || errors.note(runner.dispatch(groups, &push, true)).flatten();
        profiled();
        output.sync(false)?;
        if !output.values::<u32>().iter().all(|v| *v == 10) {
            return Err(Failure::Failed(format!(
                "raw Vulkan vector add {count} produced wrong sums"
            )));
        }
        results.push(run(
            format!("raw_vulkan_vector_add_{count}_dispatch_wait"),
            options,
            || {
                profiled();
            },
        ));
        if let Some(samples) = run_timestamps(
            format!("raw_vulkan_vector_add_{count}_gpu_execution"),
            options,
            &mut profiled,
        ) {
            report.raw_timestamp_sizes.push(count);
            results.push(samples);
        }
        errors.check()?;
    }
    Ok(())
}

/// Times one raw workload with validation and, when available, device timestamps.
#[allow(clippy::too_many_arguments)]
fn raw_workload(
    results: &mut Vec<Measurement>,
    options: &Options,
    name: &str,
    runner: &RawCompute,
    groups: [u32; 2],
    push: &[u8],
    mut prepare: impl FnMut() -> Outcome,
    mut validate: impl FnMut() -> bool,
) -> Outcome {
    let errors = std::cell::RefCell::new(Errors::default());
    let prepare = std::cell::RefCell::new(&mut prepare);
    let dispatch = |timestamp: bool| {
        let mut errors = errors.borrow_mut();
        errors.note((prepare.borrow_mut())())?;
        errors.note(runner.dispatch(groups, push, timestamp)).flatten()
    };
    results.push(run_checked(
        format!("raw_vulkan_{name}_dispatch_wait"),
        options,
        || {
            dispatch(false);
        },
        &mut validate,
    )?);
    if let Some(samples) = run_timestamps(format!("raw_vulkan_{name}_gpu_execution"), options, || {
        dispatch(true)
    }) {
        results.push(samples);
    }
    dispatch(false);
    errors.borrow_mut().check()?;
    if !validate() {
        return Err(Failure::Failed(format!(
            "raw Vulkan output validation failed: {name}"
        )));
    }
    Ok(())
}

fn add_raw_compute(
    results: &mut Vec<Measurement>,
    options: &Options,
    device: &Device,
    info: &DeviceInfo,
) -> Outcome {
    let make = |bytes: u64| HostBuffer::new(device, info, bytes);
    for count in COMPUTE_SIZES {
        let n = count as usize;
        let float_bytes = u64::from(count) * 4;

        let mut x = make(float_bytes)?;
        x.fill(2.0_f32);
        x.sync(true)?;
        let y = std::cell::RefCell::new(make(float_bytes)?);
        let saxpy = RawCompute::new(device, SAXPY, &[&x, &y.borrow()], 8)?;
        let mut push = 2.0_f32.to_le_bytes().to_vec();
        push.extend_from_slice(&count.to_le_bytes());
        raw_workload(
            results,
            options,
            &format!("saxpy_{count}_upload"),
            &saxpy,
            [count / 64, 1],
            &push,
            || {
                let mut y = y.borrow_mut();
                y.fill(1.0_f32);
                y.sync(true)
            },
            || {
                let y = y.borrow();
                y.sync(false).is_ok() && y.values::<f32>().iter().all(|v| *v == 5.0)
            },
        )?;

        let count_push = count.to_le_bytes();
        let mut ones = make(float_bytes)?;
        ones.fill(1.0_f32);
        ones.sync(true)?;
        let reduced = make(u64::from(count / 64) * 4)?;
        let reduction = RawCompute::new(device, REDUCTION, &[&ones, &reduced], 4)?;
        raw_workload(
            results,
            options,
            &format!("reduction_{count}"),
            &reduction,
            [count / 64, 1],
            &count_push,
            || Ok(()),
            || reduced.sync(false).is_ok() && reduced.values::<f32>().iter().all(|v| *v == 64.0),
        )?;

        let mut scan_input = make(float_bytes)?;
        scan_input.fill(1_u32);
        scan_input.sync(true)?;
        let scanned = make(float_bytes)?;
        let scan = RawCompute::new(device, SCAN, &[&scan_input, &scanned], 4)?;
        raw_workload(
            results,
            options,
            &format!("tiled_scan_{count}"),
            &scan,
            [count / 64, 1],
            &count_push,
            || Ok(()),
            || {
                scanned.sync(false).is_ok()
                    && scanned
                        .values::<u32>()
                        .iter()
                        .enumerate()
                        .all(|(i, v)| *v == (i as u32 % 64) + 1)
            },
        )?;

        let stencil_output = make(float_bytes)?;
        let stencil = RawCompute::new(device, STENCIL, &[&ones, &stencil_output], 4)?;
        raw_workload(
            results,
            options,
            &format!("stencil_{count}"),
            &stencil,
            [count / 64, 1],
            &count_push,
            || Ok(()),
            || {
                let v = stencil_output.values::<f32>();
                stencil_output.sync(false).is_ok()
                    && v[0] == 2.0
                    && v[n - 1] == 2.0
                    && v[1..n - 1].iter().all(|x| *x == 3.0)
            },
        )?;
    }
    for dimension in MATRIX_DIMENSIONS {
        let bytes = u64::from(dimension * dimension) * 4;
        let mut a = make(bytes)?;
        let mut b = make(bytes)?;
        let c = make(bytes)?;
        a.fill(1.0_f32);
        b.fill(1.0_f32);
        a.sync(true)?;
        b.sync(true)?;
        let matmul = RawCompute::new(device, TILED_MATMUL, &[&a, &b, &c], 4)?;
        raw_workload(
            results,
            options,
            &format!("tiled_matmul_{dimension}"),
            &matmul,
            [dimension / 8, dimension / 8],
            &dimension.to_le_bytes(),
            || Ok(()),
            || c.sync(false).is_ok() && c.values::<f32>().iter().all(|v| *v == dimension as f32),
        )?;
    }
    Ok(())
}

fn host_pool(device: &Device, info: &DeviceInfo) -> Outcome<SharedPool> {
    let index = MemoryManager::select_memory_type_from(
        &info.memory_types,
        &info.memory_type_preferences,
        MemoryPolicy::HostVisible,
        u32::MAX,
        &MemoryTypeRequirements {
            host_visible: true,
            ..MemoryTypeRequirements::default()
        },
    )
    .map_err(crate::report::unavailable("GPU benchmarks unavailable"))?;
    let memory_type = info.memory_types[index as usize];
    let heap = info.memory_heaps[memory_type.heap_index as usize].size_bytes;
    let pool = MemoryPool::new(
        device.clone(),
        MemoryPoolConfig {
            memory_type_index: index,
            heap_index: memory_type.heap_index,
            host_visible: true,
            host_coherent: memory_type.host_coherent,
            heap_size: heap,
            budget: heap,
            block_size_hint: 4 << 20,
            ..MemoryPoolConfig::default()
        },
    )
    .map_err(crate::report::failed("upload pool"))?;
    Ok(Arc::new(Mutex::new(pool)))
}

/// `StagingRing::reserve` alone, with a controlled number of live ranges (setup and cancel are
/// outside every timed interval; no transfer or queue work is included).
fn add_staging_ring(
    results: &mut Vec<Measurement>,
    options: &Options,
    device: &Device,
    info: &DeviceInfo,
    pool: &SharedPool,
) -> Outcome {
    let atom = device.physical().properties.limits.non_coherent_atom_size;
    let unit = atom.max(256);
    fn setup(what: &'static str) -> impl FnOnce(kora::Error) -> Failure {
        move |error| Failure::Failed(format!("staging-ring {what} failed: {error}"))
    }
    for live_count in [0_u64, 1, 8, 32, 128, 512] {
        for fragmented in [false, true] {
            let capacity = if fragmented {
                (3 * live_count + 2) * unit
            } else {
                (live_count + 2) * unit
            };
            let ring = StagingRing::new(
                device.clone(),
                Arc::clone(pool),
                StagingOptions {
                    topology: info.memory_topology,
                    measured_staging_benefit: true,
                    non_coherent_atom_size: atom,
                    initial_capacity: capacity,
                    maximum_capacity: capacity,
                    ..StagingOptions::default()
                },
            )
            .map_err(setup("setup"))?;
            let mut live = Vec::new();
            for _ in 0..live_count {
                live.push(
                    ring.reserve(if fragmented { 2 * unit } else { unit }, 1, 0, None)
                        .map_err(setup("live range"))?,
                );
                if fragmented {
                    let hole = ring.reserve(unit, 1, 0, None).map_err(setup("fragmentation"))?;
                    ring.cancel(hole).map_err(setup("fragmentation"))?;
                }
            }
            let tail = ring.reserve(2 * unit, 1, 0, None).map_err(setup("tail"))?;
            ring.cancel(tail).map_err(setup("tail"))?;
            let sample = || -> Outcome<u64> {
                let begin = Instant::now();
                let slice = ring.reserve(2 * unit, 1, 0, None);
                let elapsed = begin.elapsed().as_nanos() as u64;
                ring.cancel(slice.map_err(setup("timed reservation"))?)
                    .map_err(setup("timed cancel"))?;
                Ok(elapsed)
            };
            for _ in 0..options.warmup {
                sample()?;
            }
            let samples = (0..options.iterations)
                .map(|_| sample())
                .collect::<Outcome<Vec<_>>>()?;
            let layout = if fragmented { "fragmented" } else { "dense" };
            results.push(Measurement::from_samples(
                format!("staging_ring_reserve_{layout}_{live_count}"),
                samples,
            ));
            for slice in live {
                ring.cancel(slice).map_err(setup("cleanup"))?;
            }
        }
    }
    Ok(())
}

fn add_transfers(
    results: &mut Vec<Measurement>,
    options: &Options,
    device: &Device,
    info: &DeviceInfo,
) -> Outcome {
    let upload_pool = host_pool(device, info)?;
    let readback_pool = host_pool(device, info)?;
    add_staging_ring(results, options, device, info, &upload_pool)?;

    let queue = Arc::new(
        QueueContext::new(
            device.clone(),
            device.transfer_queue(),
            QueueContextOptions::default(),
        )
        .map_err(crate::report::failed("transfer queue"))?,
    );
    let direct_options = TransferOptions {
        queue_family: device.transfer_queue_family(),
        command_buffer_count: 8,
        topology: MemoryTopology::Uma,
        non_coherent_atom_size: device.physical().properties.limits.non_coherent_atom_size,
        ..TransferOptions::default()
    };
    let staged_options = TransferOptions {
        measured_staging_benefit: true,
        staging_initial_capacity: 4 << 20,
        staging_maximum_capacity: 8 << 20,
        ..direct_options
    };
    let engine = |options: TransferOptions| {
        TransferEngine::new(
            device.clone(),
            Arc::clone(&queue),
            Arc::clone(&upload_pool),
            Arc::clone(&readback_pool),
            options,
        )
        .map_err(crate::report::failed("transfer engine"))
    };
    let direct = engine(direct_options)?;
    let staged = engine(staged_options)?;

    const MAXIMUM: u64 = 1 << 20;
    let source = HostBuffer::new(device, info, MAXIMUM)?;
    let destination = HostBuffer::new(device, info, MAXIMUM)?;
    let (source_view, destination_view) = (source.view(), destination.view());
    let mut errors = Errors::default();
    for size in [4096_u64, 65536, MAXIMUM] {
        let input = vec![0x5a_u8; size as usize];
        let output = Arc::new(Mutex::new(Vec::new()));
        let upload = |engine: &TransferEngine| engine.upload(&source_view, 0, &input).and_then(|e| e.wait());
        let download = |engine: &TransferEngine| {
            let sink = Arc::clone(&output);
            engine
                .download(&source_view, 0, size, None, move |bytes| {
                    *sink.lock().unwrap() = bytes.to_vec();
                    Ok(())
                })
                .and_then(|e| e.wait())
        };
        upload(&direct).map_err(crate::report::failed("Vulkan upload"))?;
        if source.bytes()[..size as usize] != input[..] {
            return Err(Failure::Failed("direct upload produced wrong bytes".into()));
        }
        results.push(run_transfer(
            format!("transfer_direct_upload_{size}"),
            size,
            options,
            || {
                errors.note(upload(&direct).map_err(crate::report::failed("Vulkan upload")));
            },
        ));
        results.push(run_transfer(
            format!("transfer_staged_upload_{size}"),
            size,
            options,
            || {
                errors.note(upload(&staged).map_err(crate::report::failed("Vulkan upload")));
            },
        ));
        for engine in [&direct, &staged] {
            download(engine).map_err(crate::report::failed("Vulkan download"))?;
            if *output.lock().unwrap() != input {
                return Err(Failure::Failed("download produced wrong bytes".into()));
            }
        }
        results.push(run_transfer(
            format!("transfer_direct_download_{size}"),
            size,
            options,
            || {
                errors.note(download(&direct).map_err(crate::report::failed("Vulkan download")));
            },
        ));
        results.push(run_transfer(
            format!("transfer_staged_download_{size}"),
            size,
            options,
            || {
                errors.note(download(&staged).map_err(crate::report::failed("Vulkan download")));
            },
        ));
        errors.check()?;
        staged
            .copy(&source_view, 0, &destination_view, 0, size)
            .and_then(|e| e.wait())
            .map_err(crate::report::failed("KORA copy"))?;
        if destination.bytes()[..size as usize] != input[..] {
            return Err(Failure::Failed("KORA copy produced wrong bytes".into()));
        }
    }

    // Raw single-command copies on the transfer queue against the KORA transfer engine's copy.
    let raw = device.raw();
    // SAFETY: (this block and the guard) valid device; the pool, command buffer, and fence are
    // destroyed only after the last submission's fence wait.
    let (pool, command, fence) = unsafe {
        let pool = raw
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                    .queue_family_index(device.transfer_queue_family()),
                None,
            )
            .map_err(vk_failed("command-pool creation"))?;
        let command = raw
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
            .map_err(vk_failed("command-buffer allocation"))?[0];
        let fence = raw
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .map_err(vk_failed("fence creation"))?;
        (pool, command, fence)
    };
    struct Guard<'a>(&'a kora::vulkan::ash::Device, vk::CommandPool, vk::Fence);
    impl Drop for Guard<'_> {
        fn drop(&mut self) {
            // SAFETY: see above.
            unsafe {
                self.0.destroy_fence(self.2, None);
                self.0.destroy_command_pool(self.1, None);
            }
        }
    }
    let _guard = Guard(raw, pool, fence);
    for size in [4096_u64, 65536, MAXIMUM] {
        let raw_copy = || -> Outcome {
            source.sync(true)?;
            // SAFETY: as above; the copy region lies inside both buffers.
            unsafe {
                raw.reset_command_buffer(command, vk::CommandBufferResetFlags::empty())
                    .map_err(vk_failed("command reset"))?;
                raw.begin_command_buffer(
                    command,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(vk_failed("command begin"))?;
                raw.cmd_copy_buffer(
                    command,
                    source.buffer,
                    destination.buffer,
                    &[vk::BufferCopy::default().size(size)],
                );
                raw.end_command_buffer(command)
                    .map_err(vk_failed("command end"))?;
                let commands = [command];
                raw.queue_submit(
                    device.transfer_queue(),
                    &[vk::SubmitInfo::default().command_buffers(&commands)],
                    fence,
                )
                .map_err(vk_failed("transfer submit"))?;
                raw.wait_for_fences(&[fence], true, u64::MAX)
                    .map_err(vk_failed("transfer wait"))?;
                raw.reset_fences(&[fence]).map_err(vk_failed("fence reset"))?;
            }
            destination.sync(false)
        };
        raw_copy()?;
        let same = || source.bytes()[..size as usize] == destination.bytes()[..size as usize];
        if !same() {
            return Err(Failure::Failed("raw Vulkan copy produced wrong bytes".into()));
        }
        results.push(run_transfer(
            format!("raw_vulkan_copy_{size}"),
            size,
            options,
            || {
                errors.note(raw_copy());
            },
        ));
        results.push(run_transfer(
            format!("kora_transfer_copy_{size}"),
            size,
            options,
            || {
                errors.note(
                    staged
                        .copy(&source_view, 0, &destination_view, 0, size)
                        .and_then(|e| e.wait())
                        .map_err(crate::report::failed("KORA transfer copy")),
                );
            },
        ));
        errors.check()?;
        destination.sync(false)?;
        if !same() {
            return Err(Failure::Failed("KORA transfer copy produced wrong bytes".into()));
        }
    }
    drop((direct, staged));
    Ok(())
}

fn add_command_pools(results: &mut Vec<Measurement>, options: &Options, device: &Device) -> Outcome {
    let family = device.compute_queue_family();
    let record_and_discard = |pool: &mut CommandPoolSet<Device>| -> kora::Status {
        let command = pool.acquire(0)?;
        pool.end_recording(command)?;
        pool.discard(command)
    };
    let mut reused =
        CommandPoolSet::new(device.clone(), family, 16).map_err(crate::report::failed("command pool"))?;
    if reused.capacity() != 16 {
        return Err(Failure::Failed("command pool has the wrong capacity".into()));
    }
    let mut errors = Errors::default();
    results.push(run("command_pool_reuse_steady_state", options, || {
        errors.note(record_and_discard(&mut reused).map_err(crate::report::failed("command-buffer reuse")));
    }));
    results.push(run("command_pool_cold_create_record_destroy", options, || {
        let pool = CommandPoolSet::new(device.clone(), family, 1);
        errors.note(
            pool.and_then(|mut pool| record_and_discard(&mut pool))
                .map_err(crate::report::failed("command pool create")),
        );
    }));
    errors.check()
}

fn kind(device_type: vk::PhysicalDeviceType) -> &'static str {
    match device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => "discrete_gpu",
        vk::PhysicalDeviceType::INTEGRATED_GPU => "integrated_gpu",
        vk::PhysicalDeviceType::VIRTUAL_GPU => "virtual_gpu",
        vk::PhysicalDeviceType::CPU => "cpu_vulkan",
        _ => "other",
    }
}

/// Every raw Vulkan and KORA transfer measurement, on the Runtime's device.
pub fn add_raw_benchmarks(
    results: &mut Vec<Measurement>,
    options: &Options,
    selected: &DeviceId,
    report: &mut GpuDevice,
) -> Outcome {
    let unavailable = crate::report::unavailable;
    let loader = Loader::load().map_err(unavailable("raw Vulkan unavailable"))?;
    let instance = loader
        .create_instance(&InstanceOptions::default())
        .map_err(unavailable("raw Vulkan unavailable"))?;
    let physical_devices = instance
        .physical_devices()
        .map_err(unavailable("raw Vulkan unavailable"))?;
    let physical: &PhysicalDeviceInfo = physical_devices.get(selected.index as usize).ok_or_else(|| {
        Failure::Unavailable(
            "GPU benchmarks unavailable: Runtime-selected Vulkan device was not enumerated".into(),
        )
    })?;
    let info = device_info(physical);
    if info.uuid != *selected.uuid {
        return Err(Failure::Unavailable(
            "GPU benchmarks unavailable: Runtime and raw Vulkan device identities do not match".into(),
        ));
    }
    let device = instance
        .create_device(
            physical,
            &DeviceOptions {
                enable_dedicated_transfer_queue: false,
                enable_timeline_semaphore: false,
                enable_synchronization2: false,
                ..DeviceOptions::default()
            },
        )
        .map_err(unavailable("raw Vulkan device"))?;
    report.index = selected.index;
    report.name = info.name.clone();
    report.device_kind = kind(physical.properties.device_type).into();
    report.api_version = info.api_version.to_string();
    report.driver_name = if physical.driver_name.is_empty() {
        "unreported".into()
    } else {
        physical.driver_name.clone()
    };
    report.compute_queue_family = device.compute_queue_family();
    add_transfers(results, options, &device, &info)?;
    add_raw_vector_add(results, options, &device, &info, report)?;
    add_raw_compute(results, options, &device, &info)?;
    add_command_pools(results, options, &device)
}
