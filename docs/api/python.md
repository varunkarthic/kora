# Python API

```sh
python3 -m pip install ".[numpy]"    # from a clone; builds the abi3 wheel with maturin
```

One wheel supports CPython 3.11 and newer. Blocking calls (runtime creation, buffer I/O, submit,
wait, readback) release the GIL.

## Example

```python title="examples/vector_add.py"
--8<-- "examples/vector_add.py"
```

## Classes

| Class | Highlights |
| --- | --- |
| `Runtime` | `Runtime.create(options=None)`, `devices`, `create_buffer(size_bytes, usage=STORAGE)`, `create_buffer_from(array)`, `load_kernel(path_or_bytes, entry, metadata, device=None)`, `dispatch(kernel)`, `readback(buffer, device=None)` |
| `Buffer` | `len(buffer)`, `id`, `read() -> bytes`, `write(data)`, `to_numpy(dtype)`, `map(access=READ_WRITE) -> MappedView` |
| `MappedView` | Buffer-protocol view of the host copy; `release()` |
| `DispatchBuilder` | `global_size(x, y=1, z=1)` (alias `global`), `read`, `write`, `read_write`, `push(data)`, `defer_readback()`, `submit() -> Event` |
| `Event` | `wait()`, `wait_for(seconds)`, `done`, `await event` |
| `KernelMetadata` | Manifest fields, `from_json(text)`, `to_json()`; with `KernelBindingMetadata`, `PushConstantFieldMetadata`, `SpecializationConstantMetadata` |
| `RuntimeOptions` | `device_policy`, `device_index`, `device_uuid`, `cache_dir`, `enable_validation`, `enable_startup_calibration`, `disable_autotune`, `force_legacy_sync` (each `None` by default) |
| `TaskGraph`, `ExecutionPlan`, `TaskMetadata`, `TaskNode`, `TaskGraphPlan` | [Task graphs](../guides/task-graphs.md) |
| `KoraError` | `code` (`ErrorCode`), `code_name`, `message`, `device`, `kernel_id`, `task_id`, `vk_result`, `recovery_action` |
| Enums | `ErrorCode`, `DevicePolicy`, `BufferUsage`, `MapAccess`, `KernelBindingKind`, `KernelAccess`, `TaskType`, `AccessMode` |

`write`, `push`, and `create_buffer_from` accept any C-contiguous buffer-protocol object: `bytes`,
`bytearray`, `memoryview`, `array.array`, or a NumPy array of any dtype.

## NumPy without copies

```python
mapping = buffer.map()
view = np.frombuffer(memoryview(mapping), dtype=np.float32)   # shares memory with the host copy
view[:] = 1.0
del view
mapping.release()
```

A mapping belongs to the thread that created it. While it lives, other access to that buffer from the
same thread raises `KoraError` (`INVALID_ARGUMENT`) instead of deadlocking; other threads wait.

## asyncio

```python
async def run():
    first = runtime.dispatch(k).global_size(n).read(a).write(b).submit()
    second = runtime.dispatch(k).global_size(n).read(c).write(d).submit()
    await asyncio.gather(first, second)
```

Awaiting registers a completion callback that resolves the future on the event loop; no thread is
blocked per await.

## Changes from 0.x

See [Migrating from 0.x](../project/migrating.md#python). In short: Python 3.11+, `RuntimeOptions`
fields default to `None`, mappings are per-thread, and a `DispatchBuilder` submits once.
