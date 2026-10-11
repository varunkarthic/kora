# Python package

KORA's Python package is a thin pybind11 extension linked to the C++ runtime. The `kora` package
contains no execution policy or duplicate runtime implementation; runtime decisions and GPU work
remain in C++. The optional binding provides runtime, buffer, kernel, dispatch, and event objects.

## Install from a source checkout

Python 3.9 or newer, a C++20 compiler, CMake, and a supported build backend are required. From the
repository root, create and activate a virtual environment, then install the local package:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
python -c "import kora; print(kora.version())"
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. A successful import prints the linked KORA
version, currently `0.1.0` for the default CMake version configuration. The wheel includes the
compiled native extension and the small Python import facade. The build uses scikit-build-core and
pybind11 from PEP 517 build isolation; Python dependencies are not required for the default native
CMake build.

To install the package and its test dependency, use `python -m pip install '.[test]'`, then run
`python -m pytest`. Tests cover package identity, mapped-buffer lifetime, repeated runtime creation,
and a real vector-add dispatch when Vulkan and `glslangValidator` are available. Device-dependent
tests skip only when no usable Vulkan implementation/device is present.

## Native CMake build

The bindings are optional in native builds. Enable them with the `python` CMake preset or configure
with `-DKORA_BUILD_PYTHON=ON`; CMake finds Python development headers and pybind11, fetching the
project's pinned pybind11 release only when no package configuration is available. The preset builds
the extension alongside the native library. Standard Debug, Release, sanitizer, and installed C++
consumer builds remain independent of Python.

## API reference

### Runtime and errors

`Runtime.create(options=RuntimeOptions())` initializes the native runtime. `RuntimeOptions` exposes
the native device policy, index/UUID selectors, and cache directory. Initialization releases the
GIL. A runtime owns the C++ runtime; buffers, kernels, dispatch builders, and events retain shared
Python-side ownership so the native runtime cannot be destroyed while an operation still refers to
it.

`KoraError` is raised for failed `Result<T>`/`Status` operations. Its `code` is an `ErrorCode` enum
value and `code_name` is the stable native spelling; applicable Vulkan result, device, kernel, task,
and recovery details are attached as attributes. Do not parse the human-readable message to branch
on failures. `version()` and `schema_version()` report identity from the linked C++ runtime.

### Buffers and the Python buffer protocol

`runtime.create_buffer(size_bytes, usage=BufferUsage.STORAGE)` creates a native byte buffer.
`buffer.write(data)` accepts any contiguous, one-dimensional Python buffer exporter (including
`bytes`, `bytearray`, `memoryview`, and NumPy arrays when NumPy is installed); the payload is copied
into KORA's host shadow. `buffer.read()` returns a copied `bytes` snapshot of that host shadow and
releases the GIL during the native copy. Device writes become visible in the host shadow after an
event completes or after `runtime.readback(buffer).wait()`.

`buffer.map(access=MapAccess.READ_WRITE)` returns a `MappedView` implementing Python's buffer
protocol. Use `memoryview(buffer.map())` to access bytes without an intermediate copy. The view
retains the native mapping and its buffer/runtime owners; keep the memoryview alive only as long as
the mapped access is needed, then release it. `MapAccess.READ` creates a read-only export. Mapping is
not a NumPy dependency; callers may use `numpy.frombuffer()` with an appropriately typed view if
NumPy is installed.

### Kernels and dispatch

Build `KernelMetadata` using `KernelBindingMetadata`, `PushConstantFieldMetadata`, and optional
`SpecializationConstantMetadata`. Their fields mirror the corresponding C++ metadata model. Load a
compiled SPIR-V module with `runtime.load_kernel(spirv_path, entry_point, metadata)`; file loading,
validation, and pipeline setup happen in C++ with the GIL released. A `Kernel` exposes its validated
`entry_point` and metadata.

Create work with `runtime.dispatch(kernel)`. The returned builder supports `global_size(x)` or
`global_size(x, y, z)`, then `.read(buffer)`, `.write(buffer)`, or `.read_write(buffer)` according to
the kernel's declared access, and `.push(data)` for contiguous push-constant bytes. Finish with
`.submit()` to enqueue native work and receive an `Event`. Buffers and kernels from a different
Runtime are rejected. Dispatch submission releases the GIL; the native builder and returned event
retain their referenced objects until it is safe to release them.

### Events and asynchronous completion

`event.wait()` blocks until completion and releases the GIL; `event.wait_for(timeout_seconds)` waits
up to the requested duration without cancelling pending work. `event.done` is a nonblocking status
check. Events are awaitable: `await event` registers a native completion callback and schedules
resolution onto the running asyncio loop with `loop.call_soon_threadsafe`, so the loop thread is not
blocked while native work completes. Awaiting propagates failures as `KoraError` with the original
error code. Keep the event alive while awaiting it; its dispatch resources are retained for the
event's lifetime.

### Task graphs and reusable plans

`TaskGraph` is a metadata-only builder; it does not schedule devices or execute GPU work. A
`TaskMetadata` describes a `TaskType`, optional kernel/CPU-function identifiers and scheduler hints,
plus zero or more `BufferAccess` entries. `BufferId(slot, generation)` supplies the same logical
generation identity used by the C++ graph API. `AccessMode` drives automatic RAW/WAR/WAW dependency
inference. `graph.add_task(metadata)` returns its `TaskId`; `graph.add_dependency(prerequisite,
dependent)` adds an explicit order. `graph.finalize()` returns immutable task nodes and a deterministic
topological order, or raises `KoraError` (including `ErrorCode.GRAPH_CYCLE` for a cycle).

`ExecutionPlan.create(graph)` snapshots the finalized metadata and dependency order. Calling
`plan.replay(executor)` invokes the Python callback once per `TaskNode` in that order. The callback
may return `None` or `True` to continue, or `False` to stop with `KoraError`/`PYTHON_BINDING_ERROR`;
exceptions raised by the callback are translated into that same KORA error category. Since Python
callbacks require the GIL, replay retains it while invoking the callback. Native-only callback
replay remains available in C++ for allocation-free/GIL-free hot paths. The plan is reusable after a
successful replay and contains no hidden backend work; callers provide any actual task execution in
the executor.

### Minimal example

```python
import asyncio
import kora

async def main():
    runtime = kora.Runtime.create()
    buffer = runtime.create_buffer(16)
    buffer.write(bytes(range(16)))
    assert buffer.read() == bytes(range(16))
    await runtime.readback(buffer)

asyncio.run(main())
```

For a metadata-only graph and replay:

```python
graph = kora.TaskGraph()
task_id = graph.add_task(kora.TaskMetadata())
plan = kora.ExecutionPlan.create(graph)
visited = []
plan.replay(lambda node: visited.append(node.id))
assert visited == [task_id]
```

The package version in `pyproject.toml` must be kept in step with the repository's release version
when publishing Python distributions.
