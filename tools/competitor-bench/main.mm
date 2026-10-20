// competitor-bench: runs KORA's matched compute workloads through other macOS compute paths on the
// same machine so KORA can be compared against them. Optional, macOS-only, and never linked into
// the KORA library (AGENTS.md keeps vendor-specific APIs out of KORA itself).
//
//   competitor-bench [--iterations N] [--warmup N] > competitors.json
//
// Workloads, sizes, inputs, and output checks match kora-bench's matched compute suite. Every
// sample is validated after the timer stops. Host timings cover submit through completion; device
// timings use each API's own GPU clock (Metal command-buffer GPU time, OpenCL profiling events,
// OpenGL GL_TIME_ELAPSED queries).
#define CL_SILENCE_DEPRECATION
#define GL_SILENCE_DEPRECATION
#define ACCELERATE_NEW_LAPACK
#import <Accelerate/Accelerate.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <OpenCL/opencl.h>
#include <OpenGL/OpenGL.h>
#include <OpenGL/gl3.h>
#include <sys/sysctl.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

struct Options {
    std::size_t iterations = 100;
    std::size_t warmup = 10;
};

struct Result {
    std::string name;
    std::string api;
    std::string workload;
    std::uint64_t size = 0;
    std::string domain;
    std::string implementation;
    std::vector<std::uint64_t> samples;
};

std::vector<Result> g_results;
Options g_options;

constexpr std::uint32_t kLocal = 64;
const std::vector<std::uint32_t> kComputeSizes{4096, 65536, 1048576};
const std::vector<std::uint32_t> kMatrixDimensions{16, 64, 128};
const std::vector<std::uint32_t> kVectorSizes{4096, 65536, 1048576, 4194240};

std::uint64_t now_ns() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count());
}

std::uint64_t percentile(std::vector<std::uint64_t> values, std::size_t p) {
    std::sort(values.begin(), values.end());
    const auto rank = std::max<std::size_t>((p * values.size() + 99) / 100, 1);
    return values[rank - 1];
}

// One timed operation: returns the device-time sample (ns) if the API reports one.
using Operation = std::function<std::optional<std::uint64_t>()>;

// Runs warm-ups and measured samples, validating every output outside the timed region.
void measure(const std::string& api, const std::string& workload, std::uint64_t size, const std::string& suffix,
             const std::string& implementation, const Operation& operation, const std::function<bool()>& validate,
             const std::string& host_domain = "host_submit_completion") {
    for (std::size_t i = 0; i < g_options.warmup; ++i) {
        operation();
        if (!validate()) throw std::runtime_error(api + " " + workload + " warm-up output is wrong");
    }
    Result host{api + "_" + workload + "_" + std::to_string(size) + suffix, api, workload, size, host_domain,
                implementation, {}};
    Result device{api + "_" + workload + "_" + std::to_string(size) + "_gpu_execution", api, workload, size,
                  "gpu_execution_timestamp", implementation, {}};
    bool device_available = true;
    for (std::size_t i = 0; i < g_options.iterations; ++i) {
        const auto begin = now_ns();
        const auto device_ns = operation();
        host.samples.push_back(now_ns() - begin);
        if (device_ns && *device_ns > 0)
            device.samples.push_back(*device_ns);
        else
            device_available = false;
        if (!validate()) throw std::runtime_error(api + " " + workload + " output is wrong");
    }
    g_results.push_back(std::move(host));
    if (device_available && host_domain != "host_cpu_execution") g_results.push_back(std::move(device));
}

bool all_equal_u32(const std::uint32_t* values, std::size_t count, std::uint32_t expected) {
    return std::all_of(values, values + count, [=](std::uint32_t v) { return v == expected; });
}
bool all_equal_f32(const float* values, std::size_t count, float expected) {
    return std::all_of(values, values + count, [=](float v) { return v == expected; });
}
bool scan_ok(const std::uint32_t* values, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i)
        if (values[i] != (i % kLocal) + 1) return false;
    return true;
}
bool stencil_ok(const float* values, std::size_t count) {
    return values[0] == 2.0F && values[count - 1] == 2.0F && all_equal_f32(values + 1, count - 2, 3.0F);
}

// ---------------------------------------------------------------------------------------------- Metal
const char* kMetalSource = R"(
#include <metal_stdlib>
using namespace metal;
kernel void vector_add(device const uint* a [[buffer(0)]], device const uint* b [[buffer(1)]],
                       device uint* o [[buffer(2)]], constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < n) o[i] = a[i] + b[i];
}
struct SaxpyParams { float scale; uint count; };
kernel void saxpy(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                  constant SaxpyParams& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p.count) y[i] = p.scale * x[i] + y[i];
}
kernel void reduction(device const float* in [[buffer(0)]], device float* out [[buffer(1)]], constant uint& n [[buffer(2)]],
                      uint i [[thread_position_in_grid]], uint lane [[thread_position_in_threadgroup]],
                      uint group [[threadgroup_position_in_grid]]) {
    threadgroup float partial[64];
    partial[lane] = i < n ? in[i] : 0.0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 32; stride > 0; stride >>= 1) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) out[group] = partial[0];
}
kernel void scan(device const uint* in [[buffer(0)]], device uint* out [[buffer(1)]], constant uint& n [[buffer(2)]],
                 uint i [[thread_position_in_grid]], uint lane [[thread_position_in_threadgroup]]) {
    threadgroup uint partial[64];
    partial[lane] = i < n ? in[i] : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 1; offset < 64; offset <<= 1) {
        uint addend = lane >= offset ? partial[lane - offset] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        partial[lane] += addend;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (i < n) out[i] = partial[lane];
}
kernel void stencil(device const float* in [[buffer(0)]], device float* out [[buffer(1)]], constant uint& n [[buffer(2)]],
                    uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    float sum = in[i];
    if (i > 0) sum += in[i - 1];
    if (i + 1 < n) sum += in[i + 1];
    out[i] = sum;
}
kernel void tiled_matmul(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                         device float* c [[buffer(2)]], constant uint& dimension [[buffer(3)]],
                         uint2 local [[thread_position_in_threadgroup]], uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup float tile_a[8][8];
    threadgroup float tile_b[8][8];
    uint row = group.y * 8 + local.y;
    uint column = group.x * 8 + local.x;
    float sum = 0.0;
    for (uint block = 0; block < dimension; block += 8) {
        uint a_column = block + local.x;
        uint b_row = block + local.y;
        tile_a[local.y][local.x] = row < dimension && a_column < dimension ? a[row * dimension + a_column] : 0.0;
        tile_b[local.y][local.x] = b_row < dimension && column < dimension ? b[b_row * dimension + column] : 0.0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 8; ++k) sum += tile_a[local.y][k] * tile_b[k][local.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < dimension && column < dimension) c[row * dimension + column] = sum;
}
)";

void run_metal(std::string& device_name) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) throw std::runtime_error("no Metal device");
        device_name = device.name.UTF8String;
        id<MTLCommandQueue> queue = [device newCommandQueue];
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:@(kMetalSource) options:nil error:&error];
        if (!library) throw std::runtime_error("Metal library compile failed");
        const auto pipeline = [&](NSString* name) {
            NSError* pipeline_error = nil;
            id<MTLComputePipelineState> state =
                [device newComputePipelineStateWithFunction:[library newFunctionWithName:name] error:&pipeline_error];
            if (!state) throw std::runtime_error("Metal pipeline creation failed");
            return state;
        };
        const auto buffer = [&](std::size_t bytes) {
            return [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        };
        // Encodes one dispatch, commits, waits, and reports the command buffer's GPU time.
        const auto dispatch = [&](id<MTLComputePipelineState> state, std::vector<id<MTLBuffer>> buffers,
                                  const void* constants, std::size_t constant_size, MTLSize groups,
                                  MTLSize threads) -> std::optional<std::uint64_t> {
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:state];
            for (NSUInteger i = 0; i < buffers.size(); ++i) [encoder setBuffer:buffers[i] offset:0 atIndex:i];
            [encoder setBytes:constants length:constant_size atIndex:buffers.size()];
            [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted) throw std::runtime_error("Metal dispatch failed");
            return static_cast<std::uint64_t>((command.GPUEndTime - command.GPUStartTime) * 1e9);
        };
        const auto groups1d = [](std::uint32_t count) { return MTLSizeMake((count + kLocal - 1) / kLocal, 1, 1); };

        auto vector_add = pipeline(@"vector_add");
        for (const auto count : kVectorSizes) {
            auto a = buffer(count * 4), b = buffer(count * 4), out = buffer(count * 4);
            std::fill_n(static_cast<std::uint32_t*>(a.contents), count, 3U);
            std::fill_n(static_cast<std::uint32_t*>(b.contents), count, 7U);
            measure("metal", "vector_add", count, "_dispatch_wait", "algorithm_matched",
                    [&] { return dispatch(vector_add, {a, b, out}, &count, 4, groups1d(count), MTLSizeMake(kLocal, 1, 1)); },
                    [&] { return all_equal_u32(static_cast<const std::uint32_t*>(out.contents), count, 10U); });
        }
        auto saxpy = pipeline(@"saxpy");
        auto reduction = pipeline(@"reduction");
        auto scan = pipeline(@"scan");
        auto stencil = pipeline(@"stencil");
        for (const auto count : kComputeSizes) {
            struct {
                float scale;
                std::uint32_t count;
            } params{2.0F, count};
            const std::vector<float> initial(count, 1.0F);
            auto x = buffer(count * 4), y = buffer(count * 4);
            std::fill_n(static_cast<float*>(x.contents), count, 2.0F);
            measure("metal", "saxpy", count, "_upload_dispatch_wait", "algorithm_matched",
                    [&] {
                        std::memcpy(y.contents, initial.data(), count * 4);  // Same input reset as KORA/raw Vulkan.
                        return dispatch(saxpy, {x, y}, &params, sizeof(params), groups1d(count), MTLSizeMake(kLocal, 1, 1));
                    },
                    [&] { return all_equal_f32(static_cast<const float*>(y.contents), count, 5.0F); });

            auto in = buffer(count * 4), partial = buffer(count / kLocal * 4);
            std::fill_n(static_cast<float*>(in.contents), count, 1.0F);
            measure("metal", "reduction", count, "_dispatch_wait", "algorithm_matched",
                    [&] { return dispatch(reduction, {in, partial}, &count, 4, groups1d(count), MTLSizeMake(kLocal, 1, 1)); },
                    [&] { return all_equal_f32(static_cast<const float*>(partial.contents), count / kLocal, 64.0F); });

            auto scan_in = buffer(count * 4), scan_out = buffer(count * 4);
            std::fill_n(static_cast<std::uint32_t*>(scan_in.contents), count, 1U);
            measure("metal", "tiled_scan", count, "_dispatch_wait", "algorithm_matched",
                    [&] { return dispatch(scan, {scan_in, scan_out}, &count, 4, groups1d(count), MTLSizeMake(kLocal, 1, 1)); },
                    [&] { return scan_ok(static_cast<const std::uint32_t*>(scan_out.contents), count); });

            auto stencil_out = buffer(count * 4);
            measure("metal", "stencil", count, "_dispatch_wait", "algorithm_matched",
                    [&] { return dispatch(stencil, {in, stencil_out}, &count, 4, groups1d(count), MTLSizeMake(kLocal, 1, 1)); },
                    [&] { return stencil_ok(static_cast<const float*>(stencil_out.contents), count); });
        }
        auto matmul = pipeline(@"tiled_matmul");
        for (const auto dimension : kMatrixDimensions) {
            const auto elements = dimension * dimension;
            auto a = buffer(elements * 4), b = buffer(elements * 4), c = buffer(elements * 4);
            std::fill_n(static_cast<float*>(a.contents), elements, 1.0F);
            std::fill_n(static_cast<float*>(b.contents), elements, 1.0F);
            measure("metal", "tiled_matmul", dimension, "_dispatch_wait", "algorithm_matched",
                    [&] {
                        return dispatch(matmul, {a, b, c}, &dimension, 4, MTLSizeMake(dimension / 8, dimension / 8, 1),
                                        MTLSizeMake(8, 8, 1));
                    },
                    [&] { return all_equal_f32(static_cast<const float*>(c.contents), elements, static_cast<float>(dimension)); });
        }
    }
}

// --------------------------------------------------------------------------------------------- OpenCL
const char* kOpenClSource = R"(
__kernel void vector_add(__global const uint* a, __global const uint* b, __global uint* o, uint n) {
    uint i = get_global_id(0);
    if (i < n) o[i] = a[i] + b[i];
}
__kernel void saxpy(__global const float* x, __global float* y, float scale, uint n) {
    uint i = get_global_id(0);
    if (i < n) y[i] = scale * x[i] + y[i];
}
__kernel void reduction(__global const float* in, __global float* out, uint n) {
    __local float partial[64];
    uint i = get_global_id(0), lane = get_local_id(0);
    partial[lane] = i < n ? in[i] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = 32; stride > 0; stride >>= 1) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0) out[get_group_id(0)] = partial[0];
}
__kernel void scan(__global const uint* in, __global uint* out, uint n) {
    __local uint partial[64];
    uint i = get_global_id(0), lane = get_local_id(0);
    partial[lane] = i < n ? in[i] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint offset = 1; offset < 64; offset <<= 1) {
        uint addend = lane >= offset ? partial[lane - offset] : 0;
        barrier(CLK_LOCAL_MEM_FENCE);
        partial[lane] += addend;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (i < n) out[i] = partial[lane];
}
__kernel void stencil(__global const float* in, __global float* out, uint n) {
    uint i = get_global_id(0);
    if (i >= n) return;
    float sum = in[i];
    if (i > 0) sum += in[i - 1];
    if (i + 1 < n) sum += in[i + 1];
    out[i] = sum;
}
__kernel void tiled_matmul(__global const float* a, __global const float* b, __global float* c, uint dimension) {
    __local float tile_a[8][8];
    __local float tile_b[8][8];
    uint lx = get_local_id(0), ly = get_local_id(1);
    uint row = get_group_id(1) * 8 + ly, column = get_group_id(0) * 8 + lx;
    float sum = 0.0f;
    for (uint block = 0; block < dimension; block += 8) {
        uint a_column = block + lx, b_row = block + ly;
        tile_a[ly][lx] = row < dimension && a_column < dimension ? a[row * dimension + a_column] : 0.0f;
        tile_b[ly][lx] = b_row < dimension && column < dimension ? b[b_row * dimension + column] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint k = 0; k < 8; ++k) sum += tile_a[ly][k] * tile_b[k][lx];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (row < dimension && column < dimension) c[row * dimension + column] = sum;
}
)";

void check(cl_int status, const char* what) {
    if (status != CL_SUCCESS) throw std::runtime_error(std::string("OpenCL ") + what + " failed: " + std::to_string(status));
}

void run_opencl(std::string& device_name) {
    cl_platform_id platform = nullptr;
    check(clGetPlatformIDs(1, &platform, nullptr), "platform");
    cl_device_id device = nullptr;
    check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr), "device");
    char name[256]{};
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    device_name = name;
    cl_int status = 0;
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status);
    check(status, "context");
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &status);
    check(status, "queue");
    cl_program program = clCreateProgramWithSource(context, 1, &kOpenClSource, nullptr, &status);
    check(status, "program");
    check(clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr), "build");
    const auto kernel = [&](const char* kernel_name) {
        cl_int kernel_status = 0;
        auto created = clCreateKernel(program, kernel_name, &kernel_status);
        check(kernel_status, "kernel");
        return created;
    };
    const auto buffer = [&](std::size_t bytes) {
        cl_int buffer_status = 0;
        auto created = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, nullptr, &buffer_status);
        check(buffer_status, "buffer");
        return created;
    };
    const auto fill = [&](cl_mem target, const void* data, std::size_t bytes) {
        check(clEnqueueWriteBuffer(queue, target, CL_TRUE, 0, bytes, data, 0, nullptr, nullptr), "write");
    };
    const auto read = [&](cl_mem source, void* data, std::size_t bytes) {
        check(clEnqueueReadBuffer(queue, source, CL_TRUE, 0, bytes, data, 0, nullptr, nullptr), "read");
    };
    // Enqueues the kernel and waits for completion. Apple's OpenCL profiling events report
    // physically impossible durations on this hardware (a 1M-element vector add, 12 MiB of traffic,
    // in under 10 µs), so no OpenCL device time is reported; host time is unaffected.
    const auto run = [&](cl_kernel k, cl_uint dimensions, const std::size_t* global,
                         const std::size_t* local) -> std::optional<std::uint64_t> {
        check(clEnqueueNDRangeKernel(queue, k, dimensions, nullptr, global, local, 0, nullptr, nullptr), "enqueue");
        check(clFinish(queue), "finish");
        return std::nullopt;
    };
    const auto round_up = [](std::uint32_t count) { return std::size_t{(count + kLocal - 1) / kLocal * kLocal}; };
    const std::size_t local = kLocal;

    auto vector_add = kernel("vector_add");
    for (const auto count : kVectorSizes) {
        auto a = buffer(count * 4), b = buffer(count * 4), out = buffer(count * 4);
        fill(a, std::vector<std::uint32_t>(count, 3U).data(), count * 4);
        fill(b, std::vector<std::uint32_t>(count, 7U).data(), count * 4);
        clSetKernelArg(vector_add, 0, sizeof(cl_mem), &a);
        clSetKernelArg(vector_add, 1, sizeof(cl_mem), &b);
        clSetKernelArg(vector_add, 2, sizeof(cl_mem), &out);
        clSetKernelArg(vector_add, 3, sizeof(count), &count);
        std::vector<std::uint32_t> host(count);
        const auto global = round_up(count);
        measure("opencl", "vector_add", count, "_dispatch_wait", "algorithm_matched",
                [&] { return run(vector_add, 1, &global, &local); },
                [&] { read(out, host.data(), count * 4); return all_equal_u32(host.data(), count, 10U); });
        clReleaseMemObject(a);
        clReleaseMemObject(b);
        clReleaseMemObject(out);
    }
    auto saxpy = kernel("saxpy");
    auto reduction = kernel("reduction");
    auto scan = kernel("scan");
    auto stencil = kernel("stencil");
    for (const auto count : kComputeSizes) {
        const auto global = round_up(count);
        const std::vector<float> initial(count, 1.0F);
        std::vector<float> host(count);
        auto x = buffer(count * 4), y = buffer(count * 4);
        fill(x, std::vector<float>(count, 2.0F).data(), count * 4);
        const float scale = 2.0F;
        clSetKernelArg(saxpy, 0, sizeof(cl_mem), &x);
        clSetKernelArg(saxpy, 1, sizeof(cl_mem), &y);
        clSetKernelArg(saxpy, 2, sizeof(scale), &scale);
        clSetKernelArg(saxpy, 3, sizeof(count), &count);
        measure("opencl", "saxpy", count, "_upload_dispatch_wait", "algorithm_matched",
                [&] {
                    check(clEnqueueWriteBuffer(queue, y, CL_FALSE, 0, count * 4, initial.data(), 0, nullptr, nullptr),
                          "reset");
                    return run(saxpy, 1, &global, &local);
                },
                [&] { read(y, host.data(), count * 4); return all_equal_f32(host.data(), count, 5.0F); });

        auto in = buffer(count * 4), partial = buffer(count / kLocal * 4);
        fill(in, std::vector<float>(count, 1.0F).data(), count * 4);
        clSetKernelArg(reduction, 0, sizeof(cl_mem), &in);
        clSetKernelArg(reduction, 1, sizeof(cl_mem), &partial);
        clSetKernelArg(reduction, 2, sizeof(count), &count);
        std::vector<float> sums(count / kLocal);
        measure("opencl", "reduction", count, "_dispatch_wait", "algorithm_matched",
                [&] { return run(reduction, 1, &global, &local); },
                [&] { read(partial, sums.data(), sums.size() * 4); return all_equal_f32(sums.data(), sums.size(), 64.0F); });

        auto scan_in = buffer(count * 4), scan_out = buffer(count * 4);
        fill(scan_in, std::vector<std::uint32_t>(count, 1U).data(), count * 4);
        clSetKernelArg(scan, 0, sizeof(cl_mem), &scan_in);
        clSetKernelArg(scan, 1, sizeof(cl_mem), &scan_out);
        clSetKernelArg(scan, 2, sizeof(count), &count);
        std::vector<std::uint32_t> scanned(count);
        measure("opencl", "tiled_scan", count, "_dispatch_wait", "algorithm_matched",
                [&] { return run(scan, 1, &global, &local); },
                [&] { read(scan_out, scanned.data(), count * 4); return scan_ok(scanned.data(), count); });

        auto stencil_out = buffer(count * 4);
        clSetKernelArg(stencil, 0, sizeof(cl_mem), &in);
        clSetKernelArg(stencil, 1, sizeof(cl_mem), &stencil_out);
        clSetKernelArg(stencil, 2, sizeof(count), &count);
        measure("opencl", "stencil", count, "_dispatch_wait", "algorithm_matched",
                [&] { return run(stencil, 1, &global, &local); },
                [&] { read(stencil_out, host.data(), count * 4); return stencil_ok(host.data(), count); });
        for (auto memory : {x, y, in, partial, scan_in, scan_out, stencil_out}) clReleaseMemObject(memory);
    }
    auto matmul = kernel("tiled_matmul");
    for (const auto dimension : kMatrixDimensions) {
        const auto elements = dimension * dimension;
        auto a = buffer(elements * 4), b = buffer(elements * 4), c = buffer(elements * 4);
        fill(a, std::vector<float>(elements, 1.0F).data(), elements * 4);
        fill(b, std::vector<float>(elements, 1.0F).data(), elements * 4);
        clSetKernelArg(matmul, 0, sizeof(cl_mem), &a);
        clSetKernelArg(matmul, 1, sizeof(cl_mem), &b);
        clSetKernelArg(matmul, 2, sizeof(cl_mem), &c);
        clSetKernelArg(matmul, 3, sizeof(dimension), &dimension);
        const std::size_t global[2]{dimension, dimension};
        const std::size_t tile[2]{8, 8};
        std::vector<float> host(elements);
        measure("opencl", "tiled_matmul", dimension, "_dispatch_wait", "algorithm_matched",
                [&] { return run(matmul, 2, global, tile); },
                [&] { read(c, host.data(), elements * 4); return all_equal_f32(host.data(), elements, static_cast<float>(dimension)); });
        for (auto memory : {a, b, c}) clReleaseMemObject(memory);
    }
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

// --------------------------------------------------------------------------------------------- OpenGL
// macOS OpenGL 4.1 has no compute shaders. Every workload runs as a vertex shader over GL_POINTS
// with rasterization disabled, reading inputs through buffer textures and capturing one output per
// vertex with transform feedback. Elementwise kernels map one-to-one; reduction, scan, and matmul
// have no shared memory, so each output loops over its inputs instead ("emulation").
struct GlProgram {
    GLuint program = 0;
    GLuint vao = 0;
};

GLuint compile(GLenum type, const std::string& source) {
    const GLuint shader = glCreateShader(type);
    const char* text = source.c_str();
    glShaderSource(shader, 1, &text, nullptr);
    glCompileShader(shader);
    GLint ok = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048]{};
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        throw std::runtime_error(std::string("GL shader compile failed: ") + log);
    }
    return shader;
}

GlProgram gl_program(const std::string& body, const char* output_type) {
    const std::string vertex = std::string("#version 410 core\n") +
                               "uniform usamplerBuffer u0; uniform usamplerBuffer u1;\n"
                               "uniform samplerBuffer f0; uniform samplerBuffer f1;\n"
                               "uniform uint n; uniform float scale;\n"
                               "flat out " + output_type + " result;\nvoid main() {\n int i = gl_VertexID;\n" + body + "\n}\n";
    const std::string fragment = "#version 410 core\nout vec4 color;\nvoid main() { color = vec4(0); }\n";
    GlProgram program;
    program.program = glCreateProgram();
    glAttachShader(program.program, compile(GL_VERTEX_SHADER, vertex));
    glAttachShader(program.program, compile(GL_FRAGMENT_SHADER, fragment));
    const char* varyings[] = {"result"};
    glTransformFeedbackVaryings(program.program, 1, varyings, GL_SEPARATE_ATTRIBS);
    glLinkProgram(program.program);
    GLint ok = 0;
    glGetProgramiv(program.program, GL_LINK_STATUS, &ok);
    if (!ok) throw std::runtime_error("GL program link failed");
    glGenVertexArrays(1, &program.vao);
    return program;
}

struct GlBuffer {
    GLuint buffer = 0;
    GLuint texture = 0;
};

GlBuffer gl_buffer(std::size_t bytes, const void* data, GLenum texture_format) {
    GlBuffer result;
    glGenBuffers(1, &result.buffer);
    glBindBuffer(GL_TEXTURE_BUFFER, result.buffer);
    glBufferData(GL_TEXTURE_BUFFER, static_cast<GLsizeiptr>(bytes), data, GL_DYNAMIC_DRAW);
    glGenTextures(1, &result.texture);
    glBindTexture(GL_TEXTURE_BUFFER, result.texture);
    glTexBuffer(GL_TEXTURE_BUFFER, texture_format, result.buffer);
    return result;
}

void run_opengl(std::string& renderer) {
    CGLPixelFormatAttribute attributes[] = {kCGLPFAOpenGLProfile, static_cast<CGLPixelFormatAttribute>(kCGLOGLPVersion_GL4_Core),
                                            kCGLPFAAccelerated, static_cast<CGLPixelFormatAttribute>(0)};
    CGLPixelFormatObj format = nullptr;
    GLint formats = 0;
    if (CGLChoosePixelFormat(attributes, &format, &formats) != kCGLNoError || !format)
        throw std::runtime_error("no accelerated OpenGL 4.1 pixel format");
    CGLContextObj context = nullptr;
    if (CGLCreateContext(format, nullptr, &context) != kCGLNoError) throw std::runtime_error("CGLCreateContext failed");
    CGLSetCurrentContext(context);
    renderer = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
    GLint max_texels = 0;
    glGetIntegerv(GL_MAX_TEXTURE_BUFFER_SIZE, &max_texels);
    // A headless context has no default framebuffer, and draws need a complete one even with
    // rasterization discarded.
    GLuint framebuffer = 0, renderbuffer = 0;
    glGenFramebuffers(1, &framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glGenRenderbuffers(1, &renderbuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, 1, 1);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, renderbuffer);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        throw std::runtime_error("OpenGL framebuffer is incomplete");
    glEnable(GL_RASTERIZER_DISCARD);
    GLuint query = 0;
    glGenQueries(1, &query);

    // Draws `count` points, capturing each vertex's result into `output`; returns GPU time.
    const auto draw = [&](const GlProgram& program, std::uint32_t count, GLuint output,
                          const std::vector<std::pair<const char*, GLuint>>& textures, std::uint32_t n,
                          float scale) -> std::optional<std::uint64_t> {
        glUseProgram(program.program);
        glBindVertexArray(program.vao);
        for (std::size_t unit = 0; unit < textures.size(); ++unit) {
            glActiveTexture(GL_TEXTURE0 + static_cast<GLenum>(unit));
            glBindTexture(GL_TEXTURE_BUFFER, textures[unit].second);
            glUniform1i(glGetUniformLocation(program.program, textures[unit].first), static_cast<GLint>(unit));
        }
        glUniform1ui(glGetUniformLocation(program.program, "n"), n);
        glUniform1f(glGetUniformLocation(program.program, "scale"), scale);
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, output);
        glBeginQuery(GL_TIME_ELAPSED, query);
        glBeginTransformFeedback(GL_POINTS);
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(count));
        glEndTransformFeedback();
        glEndQuery(GL_TIME_ELAPSED);
        glFinish();
        GLuint64 elapsed = 0;
        glGetQueryObjectui64v(query, GL_QUERY_RESULT, &elapsed);
        if (glGetError() != GL_NO_ERROR) throw std::runtime_error("OpenGL draw failed");
        return elapsed > 0 ? std::optional<std::uint64_t>(elapsed) : std::nullopt;
    };
    const auto output_buffer = [](std::size_t bytes) {
        GLuint buffer = 0;
        glGenBuffers(1, &buffer);
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, buffer);
        glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, static_cast<GLsizeiptr>(bytes), nullptr, GL_DYNAMIC_COPY);
        return buffer;
    };
    const auto read_back = [](GLuint buffer, void* destination, std::size_t bytes) {
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, buffer);
        glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, static_cast<GLsizeiptr>(bytes), destination);
    };

    auto vector_add = gl_program(" result = texelFetch(u0, i).r + texelFetch(u1, i).r;", "uint");
    for (const auto count : kVectorSizes) {
        if (count > static_cast<std::uint32_t>(max_texels)) continue;
        auto a = gl_buffer(count * 4, std::vector<std::uint32_t>(count, 3U).data(), GL_R32UI);
        auto b = gl_buffer(count * 4, std::vector<std::uint32_t>(count, 7U).data(), GL_R32UI);
        const auto out = output_buffer(count * 4);
        std::vector<std::uint32_t> host(count);
        measure("opengl", "vector_add", count, "_dispatch_wait", "transform_feedback",
                [&] { return draw(vector_add, count, out, {{"u0", a.texture}, {"u1", b.texture}}, count, 0.0F); },
                [&] { read_back(out, host.data(), count * 4); return all_equal_u32(host.data(), count, 10U); });
    }
    auto saxpy = gl_program(" result = scale * texelFetch(f0, i).r + texelFetch(f1, i).r;", "float");
    auto reduction = gl_program(
        " float sum = 0.0; for (int lane = 0; lane < 64; ++lane) { int j = i * 64 + lane;"
        " if (uint(j) < n) sum += texelFetch(f0, j).r; } result = sum;",
        "float");
    auto scan = gl_program(
        " uint sum = 0u; for (int j = i - i % 64; j <= i; ++j) sum += texelFetch(u0, j).r; result = sum;", "uint");
    auto stencil = gl_program(
        " float sum = texelFetch(f0, i).r; if (i > 0) sum += texelFetch(f0, i - 1).r;"
        " if (uint(i + 1) < n) sum += texelFetch(f0, i + 1).r; result = sum;",
        "float");
    for (const auto count : kComputeSizes) {
        if (count > static_cast<std::uint32_t>(max_texels)) continue;
        const std::vector<float> initial(count, 1.0F);
        auto x = gl_buffer(count * 4, std::vector<float>(count, 2.0F).data(), GL_R32F);
        auto y = gl_buffer(count * 4, initial.data(), GL_R32F);
        const auto saxpy_out = output_buffer(count * 4);
        std::vector<float> host(count);
        measure("opengl", "saxpy", count, "_upload_dispatch_wait", "transform_feedback",
                [&] {
                    glBindBuffer(GL_TEXTURE_BUFFER, y.buffer);
                    glBufferSubData(GL_TEXTURE_BUFFER, 0, count * 4, initial.data());
                    return draw(saxpy, count, saxpy_out, {{"f0", x.texture}, {"f1", y.texture}}, count, 2.0F);
                },
                [&] { read_back(saxpy_out, host.data(), count * 4); return all_equal_f32(host.data(), count, 5.0F); });

        auto ones = gl_buffer(count * 4, initial.data(), GL_R32F);
        const auto sums = output_buffer(count / kLocal * 4);
        std::vector<float> partial(count / kLocal);
        measure("opengl", "reduction", count, "_dispatch_wait", "transform_feedback_emulation",
                [&] { return draw(reduction, count / kLocal, sums, {{"f0", ones.texture}}, count, 0.0F); },
                [&] { read_back(sums, partial.data(), partial.size() * 4); return all_equal_f32(partial.data(), partial.size(), 64.0F); });

        auto scan_in = gl_buffer(count * 4, std::vector<std::uint32_t>(count, 1U).data(), GL_R32UI);
        const auto scan_out = output_buffer(count * 4);
        std::vector<std::uint32_t> scanned(count);
        measure("opengl", "tiled_scan", count, "_dispatch_wait", "transform_feedback_emulation",
                [&] { return draw(scan, count, scan_out, {{"u0", scan_in.texture}}, count, 0.0F); },
                [&] { read_back(scan_out, scanned.data(), count * 4); return scan_ok(scanned.data(), count); });

        const auto stencil_out = output_buffer(count * 4);
        measure("opengl", "stencil", count, "_dispatch_wait", "transform_feedback",
                [&] { return draw(stencil, count, stencil_out, {{"f0", ones.texture}}, count, 0.0F); },
                [&] { read_back(stencil_out, host.data(), count * 4); return stencil_ok(host.data(), count); });
    }
    auto matmul = gl_program(
        " int d = int(n); int row = i / d; int column = i % d; float sum = 0.0;"
        " for (int k = 0; k < d; ++k) sum += texelFetch(f0, row * d + k).r * texelFetch(f1, k * d + column).r;"
        " result = sum;",
        "float");
    for (const auto dimension : kMatrixDimensions) {
        const auto elements = dimension * dimension;
        const std::vector<float> ones(elements, 1.0F);
        auto a = gl_buffer(elements * 4, ones.data(), GL_R32F);
        auto b = gl_buffer(elements * 4, ones.data(), GL_R32F);
        const auto c = output_buffer(elements * 4);
        std::vector<float> host(elements);
        measure("opengl", "tiled_matmul", dimension, "_dispatch_wait", "transform_feedback_emulation",
                [&] { return draw(matmul, elements, c, {{"f0", a.texture}, {"f1", b.texture}}, dimension, 0.0F); },
                [&] { read_back(c, host.data(), elements * 4); return all_equal_f32(host.data(), elements, static_cast<float>(dimension)); });
    }
    CGLSetCurrentContext(nullptr);
    CGLDestroyContext(context);
    CGLDestroyPixelFormat(format);
}

// ------------------------------------------------------------------------------------------ Accelerate
// Tuned CPU library paths (may use several cores). Compared with KORA's CPU baselines, not with GPU
// host latency. Accelerate has no 64-wide tiled integer scan, so that workload is not reported.
void run_accelerate() {
    for (const auto count : kVectorSizes) {
        std::vector<int> a(count, 3), b(count, 7), out(count);
        measure("accelerate", "vector_add", count, "", "library",
                [&] { vDSP_vaddi(a.data(), 1, b.data(), 1, out.data(), 1, count); return std::nullopt; },
                [&] { return std::all_of(out.begin(), out.end(), [](int v) { return v == 10; }); }, "host_cpu_execution");
    }
    for (const auto count : kComputeSizes) {
        std::vector<float> x(count, 2.0F), y(count, 1.0F), out(count);
        const float scale = 2.0F;
        measure("accelerate", "saxpy", count, "", "library",
                [&] { vDSP_vsma(x.data(), 1, &scale, y.data(), 1, out.data(), 1, count); return std::nullopt; },
                [&] { return all_equal_f32(out.data(), count, 5.0F); }, "host_cpu_execution");
        std::vector<float> ones(count, 1.0F), sums(count / kLocal);
        measure("accelerate", "reduction", count, "", "library",
                [&] {
                    for (std::size_t group = 0; group < sums.size(); ++group)
                        vDSP_sve(ones.data() + group * kLocal, 1, &sums[group], kLocal);
                    return std::nullopt;
                },
                [&] { return all_equal_f32(sums.data(), sums.size(), 64.0F); }, "host_cpu_execution");
        // Zero-padded input so the three-tap convolution handles both edges like the GPU stencil.
        std::vector<float> padded(count + 2, 1.0F), filter(3, 1.0F), stencil(count);
        padded.front() = padded.back() = 0.0F;
        measure("accelerate", "stencil", count, "", "library",
                [&] { vDSP_conv(padded.data(), 1, filter.data(), 1, stencil.data(), 1, count, 3); return std::nullopt; },
                [&] { return stencil_ok(stencil.data(), count); }, "host_cpu_execution");
    }
    for (const auto dimension : kMatrixDimensions) {
        const auto elements = dimension * dimension;
        std::vector<float> a(elements, 1.0F), b(elements, 1.0F), c(elements);
        const auto d = static_cast<int>(dimension);
        measure("accelerate", "tiled_matmul", dimension, "", "library",
                [&] {
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, d, d, d, 1.0F, a.data(), d, b.data(), d, 0.0F,
                                c.data(), d);
                    return std::nullopt;
                },
                [&] { return all_equal_f32(c.data(), elements, static_cast<float>(dimension)); }, "host_cpu_execution");
    }
}

std::string json_escape(const std::string& value) {
    std::string out;
    for (const char c : value) {
        if (c == '"' || c == '\\') out.push_back('\\');
        if (static_cast<unsigned char>(c) >= 0x20) out.push_back(c);
    }
    return out;
}

std::string sysctl_string(const char* name) {
    char buffer[256]{};
    std::size_t size = sizeof(buffer);
    return sysctlbyname(name, buffer, &size, nullptr, 0) == 0 ? std::string(buffer) : "unknown";
}
}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string flag = argv[i];
        if (flag == "--iterations") g_options.iterations = std::strtoull(argv[i + 1], nullptr, 10);
        else if (flag == "--warmup") g_options.warmup = std::strtoull(argv[i + 1], nullptr, 10);
    }
    if (g_options.iterations == 0) {
        std::cerr << "--iterations must be positive\n";
        return 2;
    }
    std::string metal_device = "unavailable", opencl_device = "unavailable", gl_renderer = "unavailable";
    std::vector<std::string> unavailable;
    const auto attempt = [&](const char* api, const std::function<void()>& body) {
        try {
            body();
        } catch (const std::exception& error) {
            std::cerr << api << " unavailable: " << error.what() << '\n';
            unavailable.push_back(api);
        }
    };
    attempt("metal", [&] { run_metal(metal_device); });
    attempt("opencl", [&] { run_opencl(opencl_device); });
    attempt("opengl", [&] { run_opengl(gl_renderer); });
    attempt("accelerate", [&] { run_accelerate(); });

    std::ostringstream out;
    out << "{\"schema_version\":1,\"tool\":\"competitor-bench\",\"iterations\":" << g_options.iterations
        << ",\"warmup\":" << g_options.warmup << ",\"environment\":{\"os\":\"macos\",\"cpu_model\":\""
        << json_escape(sysctl_string("machdep.cpu.brand_string")) << "\",\"source_revision\":\""
#ifdef COMPETITOR_BENCH_SOURCE_REVISION
        << COMPETITOR_BENCH_SOURCE_REVISION
#else
        << "unknown"
#endif
        << "\"},\"devices\":{\"metal\":\"" << json_escape(metal_device) << "\",\"opencl\":\"" << json_escape(opencl_device)
        << "\",\"opengl\":\"" << json_escape(gl_renderer) << "\"},\"unavailable\":[";
    for (std::size_t i = 0; i < unavailable.size(); ++i) out << (i ? "," : "") << '"' << unavailable[i] << '"';
    out << "],\"results\":[";
    for (std::size_t i = 0; i < g_results.size(); ++i) {
        const auto& r = g_results[i];
        out << (i ? "," : "") << "{\"name\":\"" << r.name << "\",\"api\":\"" << r.api << "\",\"workload\":\"" << r.workload
            << "\",\"size\":" << r.size << ",\"timing_domain\":\"" << r.domain << "\",\"implementation\":\""
            << r.implementation << "\",\"unit\":\"ns\",\"median\":" << percentile(r.samples, 50)
            << ",\"p95\":" << percentile(r.samples, 95) << ",\"samples\":[";
        for (std::size_t s = 0; s < r.samples.size(); ++s) out << (s ? "," : "") << r.samples[s];
        out << "]}";
    }
    out << "]}\n";
    std::cout << out.str();
    return 0;
}
