import asyncio
import gc
from pathlib import Path
import struct

import pytest

import kora

FIXTURES = Path(__file__).resolve().parents[2] / "crates" / "kora" / "tests" / "fixtures"


def make_runtime():
    options = kora.RuntimeOptions()
    options.enable_startup_calibration = False
    try:
        return kora.Runtime.create(options)
    except kora.KoraError as error:
        if error.code in (kora.ErrorCode.NO_VULKAN_IMPLEMENTATION, kora.ErrorCode.NO_ELIGIBLE_DEVICE):
            pytest.skip(f"no usable Vulkan device: {error}")
        raise


def vector_add_metadata():
    metadata = kora.KernelMetadata()
    metadata.kernel_id = "python.integration.vector_add"
    metadata.local_size = (64, 1, 1)
    metadata.push_constant_size = 4
    metadata.push_constants = [kora.PushConstantFieldMetadata("count", "u32", 0, 4)]
    metadata.bindings = [
        kora.KernelBindingMetadata(0, 0, "input_a", kora.KernelBindingKind.STORAGE_BUFFER, kora.KernelAccess.READ),
        kora.KernelBindingMetadata(0, 1, "input_b", kora.KernelBindingKind.STORAGE_BUFFER, kora.KernelAccess.READ),
        kora.KernelBindingMetadata(
            0, 2, "output_data", kora.KernelBindingKind.STORAGE_BUFFER, kora.KernelAccess.WRITE
        ),
    ]
    return metadata


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


def test_mapping_blocks_other_access_from_the_same_thread():
    runtime = make_runtime()
    buffer = runtime.create_buffer(8)
    view = buffer.map(kora.MapAccess.READ)
    with pytest.raises(kora.KoraError) as captured:
        buffer.read()
    assert captured.value.code == kora.ErrorCode.INVALID_ARGUMENT
    with pytest.raises(TypeError):  # A READ mapping exports read-only memory.
        memoryview(view)[0:1] = b"x"
    view.release()
    assert buffer.read() == bytes(8)


def test_runtime_repeated_create_destroy():
    for _ in range(5):
        runtime = make_runtime()
        assert runtime.devices
        assert all(isinstance(device, kora.DeviceId) for device in runtime.devices)
        buffer = runtime.create_buffer(64)
        buffer.write(bytes(range(64)))
        assert buffer.read() == bytes(range(64))
        assert len(buffer) == 64
        del buffer
        del runtime
        gc.collect()


def test_errors_carry_codes_and_names():
    runtime = make_runtime()
    buffer = runtime.create_buffer(16)
    with pytest.raises(kora.KoraError) as captured:
        buffer.write(b"short")
    assert captured.value.code == kora.ErrorCode.INVALID_ARGUMENT
    assert captured.value.code_name == "InvalidArgument"
    assert "byte count" in str(captured.value)


def test_python_vector_add_and_awaitable_event():
    runtime = make_runtime()
    count = 257
    source_device = runtime.devices[0]
    kernel = runtime.load_kernel(FIXTURES / "vector_add.spv", "main", vector_add_metadata(), source_device)
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
        .defer_readback()
        .submit()
    )

    async def wait_for_event():
        await event

    async def wait_for_readback():
        await runtime.readback(output, source_device)

    asyncio.run(wait_for_event())
    assert event.done
    asyncio.run(wait_for_readback())
    result = struct.unpack(f"<{count}I", output.read())
    assert list(result) == [left + right for left, right in zip(values_a, values_b)]


def test_numpy_buffers_views_and_kernel_bytes():
    np = pytest.importorskip("numpy")
    runtime = make_runtime()
    count = 1000
    a = np.arange(count, dtype=np.uint32)
    b = np.full(count, 5, dtype=np.uint32)
    input_a = runtime.create_buffer_from(a)
    input_b = runtime.create_buffer_from(b)
    output = runtime.create_buffer(a.nbytes)
    kernel = runtime.load_kernel((FIXTURES / "vector_add.spv").read_bytes(), "main", vector_add_metadata())
    (
        runtime.dispatch(kernel)
        .global_size(count)
        .read(input_a)
        .read(input_b)
        .write(output)
        .push(np.array([count], dtype=np.uint32))
        .submit()
        .wait()
    )
    np.testing.assert_array_equal(output.to_numpy(np.uint32), a + 5)

    # A mapping is a zero-copy NumPy view: writes through it change the buffer.
    mapping = output.map()
    view = np.frombuffer(memoryview(mapping), dtype=np.uint32)
    view[:] = 7
    del view
    mapping.release()
    assert set(output.to_numpy(np.uint32).tolist()) == {7}

    floats = np.linspace(0.0, 1.0, 16, dtype=np.float32)
    staged = runtime.create_buffer(floats.nbytes)
    staged.write(floats)
    np.testing.assert_array_equal(staged.to_numpy(np.float32), floats)
    with pytest.raises(kora.KoraError):
        staged.write(np.zeros((4, 4), dtype=np.float32)[:, ::2])  # Not contiguous.


def test_many_events_awaited_concurrently():
    runtime = make_runtime()
    count = 64
    kernel = runtime.load_kernel(FIXTURES / "vector_add.spv", "main", vector_add_metadata())
    left = runtime.create_buffer(count * 4)
    right = runtime.create_buffer(count * 4)
    left.write(struct.pack(f"<{count}I", *range(count)))
    right.write(struct.pack(f"<{count}I", *([1] * count)))
    outputs = [runtime.create_buffer(count * 4) for _ in range(8)]

    async def run_all():
        events = [
            runtime.dispatch(kernel).global_size(count).read(left).read(right).write(out).push(
                struct.pack("<I", count)
            ).submit()
            for out in outputs
        ]
        await asyncio.gather(*events)

    asyncio.run(run_all())
    expected = [index + 1 for index in range(count)]
    for out in outputs:
        assert list(struct.unpack(f"<{count}I", out.read())) == expected


def test_dispatch_rejects_resubmission_and_foreign_buffers():
    runtime = make_runtime()
    other = make_runtime()
    kernel = runtime.load_kernel(FIXTURES / "vector_add.spv", "main", vector_add_metadata())
    foreign = other.create_buffer(16)
    with pytest.raises(kora.KoraError) as captured:
        runtime.dispatch(kernel).read(foreign)
    assert captured.value.code == kora.ErrorCode.INVALID_ARGUMENT
    builder = runtime.dispatch(kernel)
    with pytest.raises(kora.KoraError):
        builder.submit()  # Missing bindings and push constants.
    with pytest.raises(kora.KoraError):
        builder.submit()  # Already spent.


def test_kernel_metadata_json_round_trips():
    metadata = vector_add_metadata()
    constant = kora.SpecializationConstantMetadata(3, "flag", "bool", True)
    metadata.specialization_constants = [constant]
    parsed = kora.KernelMetadata.from_json(metadata.to_json())
    assert parsed.kernel_id == metadata.kernel_id
    assert parsed.local_size == (64, 1, 1)
    assert [b.name for b in parsed.bindings] == ["input_a", "input_b", "output_data"]
    assert parsed.specialization_constants[0].default_value is True


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
    assert finalized.tasks[1].metadata.cpu_function_id == "python.graph.second"
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
    assert captured.value.task_id == 1
    with pytest.raises(kora.KoraError) as raised:
        plan.replay(lambda _task: 1 / 0)
    assert "ZeroDivisionError" in str(raised.value)


def test_task_graph_detects_cycles():
    graph = kora.TaskGraph()
    a = graph.add_task(kora.TaskMetadata())
    b = graph.add_task(kora.TaskMetadata())
    graph.add_dependency(a, b)
    graph.add_dependency(b, a)
    with pytest.raises(kora.KoraError) as captured:
        graph.finalize()
    assert captured.value.code == kora.ErrorCode.GRAPH_CYCLE
