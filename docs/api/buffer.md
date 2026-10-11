# Logical buffers

`kora/buffer.hpp` provides the first logical-buffer API layer: reference-counted handles, typed
host-shadow storage, mapped ranges, generation-checked identities, and completion-gated reclamation.
The standalone `BufferManager` is CPU-side bookkeeping. When created through `Runtime`, a buffer can
be bound to a storage-buffer kernel: the first dispatch creates a Vulkan `VkBuffer` lazily and uses
the host shadow as its upload/readback source. This initial path synchronizes transfers and does not
yet retain authoritative multi-device replicas between dispatches.

## Create and map

`BufferManager::create(size_bytes, usage)` returns a `Result<BufferHandle>` for a nonzero byte size.
The typed overload `create<T>(element_count, usage)` checks `element_count * sizeof(T)` before
allocating and returns `Buffer<T>`. Typed elements must be trivially copyable and no more aligned
than `std::max_align_t`. Allocation failure is reported as `AllocationFailed`, arithmetic overflow
as `Overflow`, and zero size as `InvalidArgument`.

```cpp
kora::BufferManager buffers;
auto created = buffers.create<std::uint32_t>(256, kora::BufferUsage::Storage);
if (!created) return created.error();

auto values = std::move(created).value();
{
    auto mapped = values.map(kora::MapAccess::Write);
    if (!mapped) return mapped.error();
    mapped.value()[0] = 42;
}  // The map pin is released here.
```

`BufferHandle::map(access, offset, size)` maps a checked byte range; the default size selects the
remainder of the buffer. `Buffer<T>::map(access)` maps the complete typed allocation and returns a
move-only `MappedBuffer<T>` whose `span()` or indexed access is valid until that view is destroyed.
`Read` maps do not add dirty metadata. `Write` and `ReadWrite` maps record the mapped range as dirty
when the map succeeds (a conservative declaration, even if the caller does not modify bytes).
`MapAccess` is a caller contract; the current C++ view remains mutable for all modes, so callers must
not write through a `Read` mapping.
Dirty-range merging and non-coherent Vulkan flush alignment are separate follow-up behavior.

`Buffer<T>::write(span)` and `read(span)` provide checked whole-buffer copies without exposing a
long-lived mapping. The element count must exactly match the buffer capacity. `read()` returns a newly
allocated vector. After `Event::wait()` succeeds for a runtime dispatch, these methods observe its
GPU output copied back to the host shadow.

## References, use tracking, and deferred free

Copying `BufferHandle` or `Buffer<T>` retains the same slot/generation. Destroying a copy drops one
logical reference; after the final reference is released, the record enters the deferred queue with
its last-use timeline value. A mapped view pins the record independently, so releasing the final
buffer handle cannot invalidate an outstanding mapping.

`mark_used(value)` associates a monotonically nondecreasing completion value with the buffer. The
queue/completion layer calls `BufferManager::advance_completion(completed_value)` as timeline progress
is observed. A pending record is reclaimed only when its last-use value is complete and its pin count
is zero. Reclamation increments the slot generation before that slot can be reused. Completion values
must belong to one ordered timeline domain; values from unrelated queues or devices cannot be
compared by this manager.

`BufferId` is a non-owning slot/generation identity for diagnostics and submission records. It does
not keep storage alive. `inspect(id)` returns `StaleHandle` for a reclaimed generation. In Debug
builds, submitting a stale identity through `BufferManager::mark_used` asserts at the invariant
boundary; non-Debug builds return `StaleHandle`. `BufferInfo` exposes byte size, usage, current
residency placeholder, pin/reference counts, generation, last-use value, and pending-free state.

## Concurrency and invariants

Manager operations, handle reference updates, map pinning, last-use updates, and completion
reclamation are serialized by the manager state mutex. The mapping's byte span is caller-owned while
the view lives; callers must not concurrently read and write the same mapped bytes without their own
synchronization. This implementation keeps a host shadow and reports `HostOnly` residency. It does not
yet coordinate multiple timelines or GPU access.

The core lifecycle invariants are:

- a buffer is not reclaimed while any logical handle or mapping pin remains;
- a buffer is not reclaimed before its recorded completion value passes;
- a reused slot receives a new nonzero generation;
- stale IDs cannot inspect or mark a buffer as used;
- typed byte-size arithmetic and map ranges are checked before access.

Run the focused tests with:

```sh
cmake --preset debug
cmake --build --preset debug --parallel
ctest --preset debug --output-on-failure -R BufferManager
```
