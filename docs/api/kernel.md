# Kernel and dispatch API

The runtime kernel API loads a SPIR-V compute module against caller-supplied, versioned
`KernelMetadata`, binds logical buffers in metadata order, and returns an `Event` for asynchronous
completion. The first implementation deliberately supports storage-buffer bindings only and
prioritizes a clear, correct host-shadow upload/dispatch/readback path.

## Invariants

- `load_kernel` validates the SPIR-V module structure and checks metadata when creating a pipeline.
- Each `.read`, `.write`, or `.read_write` call consumes exactly one binding, in the order recorded
  by `KernelMetadata::bindings`; the declared access mode must match.
- `global(x, y, z)` describes invocation extents. The runtime computes group counts by ceiling
  division using the reflected local size and checks Vulkan dimension and invocation limits.
- Push-constant bytes must exactly match the reflected/declared block size and fit the selected
  device's `maxPushConstantsSize`.
- Read/read-write host shadows are uploaded before compute. Write/read-write buffers are downloaded
  before the returned event completes. Shader bounds checks are required when global extents are not
  multiples of the local size.
- Invalid dimensions, bindings, accesses, and push-constant sizes fail as `Result` errors. The
  storage-buffer-only implementation reports `UnsupportedFeature` for other descriptor kinds.

## Example

```cpp
auto input = runtime.create_buffer<std::uint32_t>(count);
auto output = runtime.create_buffer<std::uint32_t>(count);
if (!input || !output) return /* propagate the allocation error */;

auto event = runtime.dispatch(kernel)
    .global(count)
    .read(input.value())
    .write(output.value())
    .push(count)
    .submit();
if (!event) return /* propagate the validation/submission error */;
auto completed = event.value().wait(); // Includes output readback.
```

The builder copies push data and retains buffer handles through asynchronous completion. Keep the
owning Runtime alive until all submitted events and builders are finished; device resources are owned
by that Runtime. Transfer upload is currently waited before queueing compute, and readback is waited
inside completion, so this baseline does not yet overlap transfers or preserve device-resident values
across dispatches.

## Function reference

- `Runtime::create_buffer<T>(count, usage)` allocates typed host-shadow storage after checking byte
  size arithmetic. `Buffer<T>::write` and `read` copy the complete allocation.
- `Runtime::load_kernel(path, entry_point, metadata)` reads at most 16 MiB of SPIR-V, validates its
  structure, and associates metadata with the selected runtime device.
- `Runtime::dispatch(kernel)` creates a fluent builder for that device.
- `DispatchBuilder::global(x)` sets a one-dimensional invocation extent; `global(x, y, z)` sets all
  three dimensions.
- `read(buffer)`, `write(buffer)`, and `read_write(buffer)` declare access for the next metadata
  binding. Typed `Buffer<T>` and `BufferHandle` are accepted.
- `push(bytes)` copies a byte span; `push(value)` copies the object representation of one
  trivially-copyable value.
- `submit()` returns `Result<Event>`; validation and synchronous setup errors are immediate, while
  execution/transfer failures are reported by `Event::wait()`/`status()`.

## Fixtures and verification

The integration target compiles `vector_add.comp` and `saxpy.comp` with `glslangValidator` when
available. The vector-add integration test runs once for each device enumerated on the local host (or
the configured test ICD) and compares all output elements. The fixtures include a push-constant count
guard so their final partial workgroup is safe.
