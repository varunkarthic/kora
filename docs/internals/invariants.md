# Internal invariant audit

This page records twelve cross-layer correctness invariants audited against the current source and
tests. It is a verification map, not a promise that every impossible state has a recoverable API
response. Public or input-dependent failures use `Result<T>`/`Status`; `KORA_ASSERT` is reserved for
internal programming errors and is active only when standard assertions are enabled (normally
non-Release builds). Its expression must have no side effects and is not evaluated with `NDEBUG`.

## Invariant and test map

| # | Invariant | Enforcement point | Test evidence |
| ---: | --- | --- | --- |
| 1 | Size, byte-count, range-end, and alignment arithmetic never wraps into a valid smaller value. | `kora::checked::{add,mul,align_up,in_range,range_end}` return an error/empty result on invalid input; callers validate before Vulkan operations. | `tests/unit/test_checked.cpp`: `Checked.AddBoundaries`, `Checked.MulBoundaries`, `Checked.AlignUp`, `Checked.InRange`, `Checked.RangeEnd`, `Checked.ByteSize`, `Checked.Volume`. |
| 2 | Optional device behavior is enabled only when the selected `DeviceInfo` capability snapshot supports it; Vulkan vendor identity is not a capability. | `DeviceCaps` maps queried feature/extension structures; feature use is checked at pipeline/device setup. | `tests/unit/test_device_caps.cpp`: `DeviceCaps.StartsAtBaselineAndDoesNotRequireHigherTierFeatures`, `DeviceCaps.AssignsAdditiveCapabilityTiers`, `DeviceCaps.MapsLimitsFeaturesQueuesAndMemoryWithoutVulkanTypes`. |
| 3 | A kernel module must be structurally valid, reflect the supported interface, and agree with metadata before pipeline creation. | `detail::spirv::validate_and_reflect`, `ModuleParser`, and `PipelineManager::get_or_create`. | `tests/unit/test_kernel_metadata.cpp`: `KernelMetadata.RejectsUnknownFieldsAndInvalidMetadataPrecisely`, `KernelMetadata.RejectsOverflowOverlapAndTruncatedDocuments`; `tests/unit/test_spirv.cpp`: `SpirvReflection.RejectsTruncatedAndCorruptModulesWithoutTools`, `SpirvReflection.ReflectsAndChecksComputeInterfaceAgainstMetadata`, `SpirvReflection.RejectsCapabilitiesNotPresentInDeviceSnapshot`. |
| 4 | Task access declarations induce all RAW/WAR/WAW dependencies, and a cyclic graph is never exposed as a valid execution plan. | `TaskGraph::finalize` applies the writer/active-reader model and deterministic cycle detection. | `tests/unit/test_task_graph.cpp`: `TaskGraph.InfersRawWarWawAndReadWriteEdges`, `TaskGraph.RandomAccessSequencesMatchIndependentHazardOracle`, `TaskGraph.ReportsExplicitSelfAndMultiTaskCycles`. |
| 5 | A recycled buffer slot cannot make an old handle identify a new allocation. | `BufferManager` checks slot and generation on public operations; internal command use has a `KORA_ASSERT` in assertion-enabled builds and returns `StaleHandle` otherwise. | `tests/unit/test_buffer_manager.cpp`: `BufferManager.CopyingHandlesRetainsOneLogicalAllocation`, `BufferManager.DefersReclamationUntilTheLastUseCompletes` (also checks stale generation and Debug death behavior). |
| 6 | A buffer slot is not reclaimed while a submitted operation may still use it. | `BufferManager::collect` compares each device's last-use completion value before reclaiming. | `tests/unit/test_buffer_manager.cpp`: `BufferManager.DefersReclamationUntilTheLastUseCompletes`; `tests/integration/test_runtime.cpp`: `Runtime.DispatchesVectorAddOnEveryDeviceWithTimelineAndFenceFallback`. |
| 7 | A writable mapping or in-flight pin prevents storage reuse; publishing a write advances host content state. | Buffer mapping guards, pin counts, and content versions; impossible pin/reference states use `KORA_ASSERT`. | `tests/unit/test_buffer_manager.cpp`: `BufferManager.MappedPinPreventsReuseAndWritableMapsTrackDirtyRange`, `BufferManager.HostContentVersionAdvancesForWritesAndWritableMappings`. |
| 8 | Per-heap reserved bytes never exceed the effective admission limit, and failed checked reservations do not mutate the ledger. | `MemoryManager::reserve/release` use checked arithmetic; post-mutation snapshots are guarded by `KORA_ASSERT`. | `tests/unit/test_memory_manager.cpp`: `BudgetLedger.EnforcesLimitAndRecoversAfterRelease`, `BudgetLedger.CheckedArithmeticAndInvalidOperationsDoNotMutateState`, `BudgetLedger.RefreshClampsToHeapAndPreservesExistingReservations`. |
| 9 | Suballocator regions cover the arena without overlap; free-list and bitmap membership agree, and adjacent free regions are coalesced. | `Suballocator::validate_invariants` checks structure; mutation paths invoke `KORA_ASSERT(validate_invariants())` in assertion-enabled builds. | `tests/unit/test_suballocator.cpp`: `Suballocator.CoalescesBothNeighbours`, `Suballocator.RandomizedOperationsMatchIntervalOracle`, `Suballocator.SplitsAndTracksFragmentationStatistics`. |
| 10 | Host access to one Vulkan queue is serialized, and queue completion values preserve dependent submission order. | `QueueContext::submit` enforces the single-owner submission guard with `KORA_ASSERT`; timeline/fence completion is published only after the corresponding submission completes. | `tests/integration/test_queue_context.cpp`: `QueueContext.CompletesConcurrentRealVulkanSubmissions`, `QueueContext.WaitsForTimelineValueAndOrdersDependentSubmissions`. |
| 11 | A command buffer is acquired only from its owning thread and is not reset/reused before its signaled batch completes. | `CommandPoolSet::check_thread` asserts in Debug and returns `InternalInvariantViolation`; ring acquisition checks completion values. | `tests/integration/test_command_pool_set.cpp`: `CommandPoolSet.RepeatsTenThousandRecordSubmitAndReuseCycles`; pool exhaustion is exercised by the timeline-wait path in that test. |
| 12 | A staging slice remains unavailable until the oldest queue timeline value using it has completed. | `StagingRing::allocate` waits for or reports the oldest in-flight range; reclamation is timeline-gated. | `tests/integration/test_staging_ring.cpp`: `StagingRing.RandomRangesWaitForOldestTimelineAndNeverOverlapInFlight`; `tests/integration/test_runtime.cpp`: `Runtime.StagesOutputsAcrossEverySelectedDevicePairThroughHostMemory`. |

`KORA_ASSERT` is not a security boundary, input validator, cancellation mechanism, or substitute for
recoverable error handling. Debug-only assertions may abort on internal corruption; Release builds
retain runtime validation for externally reachable invalid states. The macro itself is tested in
`tests/unit/test_assert.cpp` in Debug and Release configurations.

## Macro and function reference

| Symbol | Contract |
| --- | --- |
| `KORA_ASSERT(expression)` | Project-wide spelling for a standard assertion; assertion-enabled builds evaluate and check the expression, while `NDEBUG` builds do not evaluate it. Never place state changes in the expression. |
| `checked::{add,mul,align_up,in_range,range_end}` | Checked size operations return empty/error on overflow, invalid alignment, or an out-of-bounds range. |
| `BufferManager::{inspect,mark_used,collect}` | Enforce generation identity and completion-gated slot reclamation. |
| `Suballocator::validate_invariants` | Verifies physical coverage, free-list/bitmap consistency, coalescing, and live-allocation records. |
| `MemoryManager::{reserve,release}` | Maintain a checked per-heap reservation ledger and preserve state on rejected operations. |
| `QueueContext::submit` | Serializes host access to one queue and assigns completion state to successfully submitted work. |
| `CommandPoolSet::{check_thread,acquire}` | Enforce owning-thread use and reuse command buffers only after their previous completion value. |
| `StagingRing::allocate` | Avoids overlap with slices whose completion timeline value has not been observed. |

Run the whole unit and integration suite with `ctest --preset debug --output-on-failure`. The Vulkan
tests skip when no usable implementation is available; the skip is reported separately from a pass.
