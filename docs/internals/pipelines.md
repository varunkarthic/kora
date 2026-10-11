# Compute pipeline and descriptor management

This document covers the internal per-device pipeline manager used by the Runtime. It is the
compute-pipeline preparation layer; command recording and dispatch are introduced by the following
kernel-execution issue. The manager receives SPIR-V words plus validated version-1 kernel metadata,
then creates Vulkan resources only after the interface has been reflected and matched.

## Ownership and synchronization

`detail::kernel::PipelineManager` is owned by a Runtime device entry. Its `vk::Api` copy retains the
device dispatch-table state. Runtime declares the manager after its `vk::Device`, so manager-owned
Vulkan objects are destroyed before the logical device. The manager serializes pipeline creation,
descriptor pool operations, cache save/load, and statistics with one mutex. The pipeline cache is
created lazily on the first pipeline request; creating a Runtime without compiling a kernel does
not create cache directories or files.

Returned `PipelineRef` values are non-owning views into manager/device-owned Vulkan resources and
must not outlive the Runtime device. Cache eviction removes the manager's lookup reference. If a
caller still holds a `PipelineRef`, the pipeline is moved to a retired list and destroyed only after
the final reference is released. The caller must release descriptor sets only after they are no
longer referenced by in-flight GPU work.

## Cache keys and layout caches

The serialized pipeline key contains the exact SPIR-V words, selected entry point, specialization
IDs/types/default values, descriptor set/binding/kind/access declarations, push-constant extent,
local workgroup size, and device/driver identity. Its FNV-1a hash selects the unordered-map bucket;
full serialized key equality resolves hash collisions. Kernel ID and source-level names are excluded
because they do not change Vulkan pipeline behavior.

Descriptor set layouts are canonicalized by binding number, descriptor kind, count, and shader stage.
Identical layouts can be shared across sets and pipelines. Empty layouts are created for gaps in set
numbering so the Vulkan pipeline-layout set indices remain faithful to metadata. The supported set
index is bounded to 31 to avoid untrusted metadata causing unbounded layout allocation. Pipeline
layouts are keyed by their descriptor-layout handles and the push-constant extent; version-1 metadata
maps the block to one compute-stage range beginning at offset zero.

The compute-pipeline cache is an LRU bounded by `max_cached_pipelines` (128 by default). A successful
second request for the same key returns the same pipeline record and increments
`pipeline_cache_hits`. New validated keys increment misses. Eviction destroys unused pipeline
objects immediately and retires in-use ones until safe to reclaim. Descriptor layouts remain cached
for the manager lifetime, avoiding invalidating layouts referenced by retained pipeline records.

## Descriptor pool growth and recycling

Pools are grouped by descriptor-set layout. Each newly-created pool is sized for a fixed number of
sets (32 by default) and the aggregate descriptor counts needed by that layout. When every pool for
the layout reports `VK_ERROR_OUT_OF_POOL_MEMORY` or `VK_ERROR_FRAGMENTED_POOL`, the manager grows by
one pool, up to eight pools per layout by default. Limits are configurable for internal tests and
future runtime tuning.

`free_descriptor_set` returns a handle to a KORA-side free list instead of calling
`vkFreeDescriptorSets`, which would invalidate the handle that the free list is meant to recycle.
Reallocation reuses the same still-allocated Vulkan set. Vulkan frees all remaining sets when the
manager destroys their owning pools. A set must not be returned while any submitted GPU work can
still read or write it.

Specialization values are serialized in SpecId order and converted to tightly packed Vulkan map
entries using the metadata scalar width: bool uses `VkBool32`; i32/u32/f32 use four bytes; and
i64/u64/f64 use eight. `validate_and_reflect` ensures these IDs, types, and defaults match the SPIR-V
module before the map is passed to `vkCreateComputePipelines`.

## Persistent VkPipelineCache format

Each device gets a separate file beneath `<cache_dir>/pipelines/`. The 72-byte outer envelope is
encoded explicitly in little-endian order and contains:

| Offset | Size | Field |
| ---: | ---: | --- |
| 0 | 8 | `KORAPCH1` magic |
| 8 | 4 | KORA cache schema version |
| 12 | 4 | Vulkan vendor ID |
| 16 | 4 | Vulkan device ID |
| 20 | 4 | driver version |
| 24 | 16 | Vulkan device UUID |
| 40 | 16 | Vulkan driver UUID |
| 56 | 8 | payload byte count |
| 64 | 8 | FNV-1a payload checksum |

The payload is exactly the opaque byte sequence returned by `vkGetPipelineCacheData`. Loading is
bounded to 64 MiB and requires an exact envelope length, current schema, matching device/driver
identity, and matching checksum. Missing, truncated, corrupted, mismatched, or driver-rejected data
is discarded and an empty Vulkan pipeline cache is created. A successful save writes a sibling
temporary file and renames it into place. The cache is rebuildable and contains no application
buffers or shader source.

## Function reference

- `make_pipeline_key`: serializes the module, selected entry point, specialization values, interface
  layout, local size, and device identity; computes a stable 64-bit lookup hash.
- `PipelineManager::create`: validates limits and required Vulkan entry points and binds the manager
  to one live device; it does not create disk state yet.
- `PipelineManager::ensure_pipeline_cache`: validates the outer cache envelope, imports usable
  driver data, retries with an empty cache when persisted bytes are rejected, and initializes the
  `VkPipelineCache` lazily.
- `PipelineManager::descriptor_layouts`: converts metadata to canonical per-set
  `VkDescriptorSetLayout` objects, inserting empty set layouts for gaps and recording pool sizes.
- `PipelineManager::pipeline_layout`: reuses or creates a `VkPipelineLayout` with one optional
  compute-stage push range.
- `PipelineManager::get_or_create`: reflects and validates SPIR-V, builds the exact key, returns a
  cache hit when present, or invokes pipeline creation and applies LRU eviction.
- `PipelineManager::create_pipeline`: creates a temporary shader module, packs specialization
  values, creates the `VkComputePipeline`, destroys the temporary module, and stores the pipeline.
- `PipelineManager::allocate_descriptor_set`: reuses a free handle or allocates from an existing
  pool, growing the per-layout pool chain only when required.
- `PipelineManager::free_descriptor_set`: returns an owned set to its layout-specific KORA free list
  and rejects unknown or already-free handles.
- `PipelineManager::save_pipeline_cache`: retrieves opaque driver bytes, wraps them with identity
  and integrity metadata, and publishes the file via temporary-file rename.
- `PipelineManager::destroy_pipeline_objects`: destroys pipelines before layouts, pools, and cache,
  preserving Vulkan object lifetime order.

## Tests

`tests/unit/test_pipeline_cache.cpp` verifies key determinism and that module, entry point, metadata,
specialization value, and device identity affect the serialized key. The lavapipe integration compiles
the shared GLSL fixture at build time, verifies a repeated request is a hit, exercises the bounded
LRU and descriptor free list, reopens a persisted driver cache, and verifies corrupted-file recovery.
The integration is skipped when no usable Vulkan compute device or `glslangValidator` is available.
