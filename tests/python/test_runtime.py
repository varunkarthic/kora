import asyncio
import gc
from pathlib import Path
import shutil
import struct
import subprocess

import pytest

import kora


def make_runtime():
    try:
        return kora.Runtime.create()
    except kora.KoraError as error:
        if error.code in (kora.ErrorCode.NO_VULKAN_IMPLEMENTATION, kora.ErrorCode.NO_ELIGIBLE_DEVICE):
            pytest.skip(f"no usable Vulkan device: {error}")
        raise


def test_buffer_protocol_mapping_keeps_mapping_alive():
    runtime = make_runtime()
    buffer = runtime.create_buffer(16)
    mapped = memoryview(buffer.map())
    mapped[0:4] = b"KORA"
    del buffer
    gc.collect()
    assert mapped[0:4].tobytes() == b"KORA"
    mapped.release()
    del runtime
    gc.collect()


def test_runtime_repeated_create_destroy():
    for _ in range(5):
        runtime = make_runtime()
        assert runtime.devices
        assert all(isinstance(device, kora.DeviceId) for device in runtime.devices)
        buffer = runtime.create_buffer(64)
        buffer.write(bytes(range(64)))
        assert buffer.read() == bytes(range(64))
        del buffer
        del runtime
        gc.collect()


def test_python_vector_add_and_awaitable_event(tmp_path):
    compiler = shutil.which("glslangValidator")
    if compiler is None:
        pytest.skip("glslangValidator is needed to compile the vector-add integration shader")
    shader_source = Path(__file__).resolve().parents[1] / "integration" / "vector_add.comp"
    shader_binary = tmp_path / "vector_add.spv"
    subprocess.run(
        [compiler, "-V", "--target-env", "vulkan1.0", "-o", str(shader_binary), shader_source],
        check=True,
        capture_output=True,
        text=True,
    )

    runtime = make_runtime()
    count = 257
    metadata = kora.KernelMetadata()
    metadata.kernel_id = "python.integration.vector_add"
    metadata.local_size = (64, 1, 1)
    metadata.push_constant_size = 4
    metadata.push_constants = [kora.PushConstantFieldMetadata("count", "u32", 0, 4)]
    metadata.bindings = [
        kora.KernelBindingMetadata(0, 0, "input_a", kora.KernelBindingKind.STORAGE_BUFFER,
                                   kora.KernelAccess.READ),
        kora.KernelBindingMetadata(0, 1, "input_b", kora.KernelBindingKind.STORAGE_BUFFER,
                                   kora.KernelAccess.READ),
        kora.KernelBindingMetadata(0, 2, "output_data", kora.KernelBindingKind.STORAGE_BUFFER,
                                   kora.KernelAccess.WRITE),
    ]
    source_device = runtime.devices[0]
    kernel = runtime.load_kernel(shader_binary, "main", metadata, source_device)
    input_a = runtime.create_buffer(count * 4)
    input_b = runtime.create_buffer(count * 4)
    output = runtime.create_buffer(count * 4)
    values_a = [index * 3 for index in range(count)]
    values_b = [index + 11 for index in range(count)]
    input_a.write(struct.pack(f"<{count}I", *values_a))
    input_b.write(struct.pack(f"<{count}I", *values_b))

    event = (
        runtime.dispatch(kernel)
        .global_size(count)
        .read(input_a)
        .read(input_b)
        .write(output)
        .push(struct.pack("<I", count))
        .submit()
    )
    async def wait_for_event():
        await event

    async def wait_for_readback():
        await runtime.readback(output, source_device)

    asyncio.run(wait_for_event())
    asyncio.run(wait_for_readback())
    result = struct.unpack(f"<{count}I", output.read())
    assert list(result) == [left + right for left, right in zip(values_a, values_b)]


def test_python_task_graph_finalize_and_replay():
    graph = kora.TaskGraph()
    first = kora.TaskMetadata()
    first.type = kora.TaskType.HOST_CALLBACK
    first.priority = 2
    first_id = graph.add_task(first)
    second = kora.TaskMetadata()
    second.type = kora.TaskType.CPU_TASK
    second.cpu_function_id = "python.graph.second"
    second_id = graph.add_task(second)
    graph.add_dependency(first_id, second_id)

    finalized = graph.finalize()
    assert finalized.topological_order == [first_id, second_id]
    plan = kora.ExecutionPlan.create(graph)
    replayed = []
    plan.replay(lambda task: replayed.append(task.id))
    assert replayed == [first_id, second_id]


def test_python_task_graph_executor_failure_preserves_task_id():
    graph = kora.TaskGraph()
    graph.add_task(kora.TaskMetadata())
    plan = kora.ExecutionPlan.create(graph)
    with pytest.raises(kora.KoraError) as captured:
        plan.replay(lambda _task: False)
    assert captured.value.code == kora.ErrorCode.PYTHON_BINDING_ERROR

