# Invariants

Cross-layer correctness rules and the tests that enforce them. Expected failures are values
(`Result`/`Status`); `debug_assert!` guards internal states that would indicate a KORA bug.

| # | Invariant | Enforced by | Tests |
| ---: | --- | --- | --- |
| 1 | Size, range, and alignment arithmetic never wraps into a smaller valid value. | `kora::checked` | `checked::tests::{add_and_mul_boundaries, align_up_cases, in_range_cases, range_end_cases, byte_size_and_volume}` |
| 2 | Optional device behavior is enabled only when the device reports it; vendor identity is never a capability. | `vulkan::caps`, `vulkan::device` | `caps::tests::{starts_at_baseline_and_does_not_require_higher_tier_features, assigns_additive_capability_tiers, maps_limits_features_queues_and_memory}` |
| 3 | A kernel is structurally valid SPIR-V and agrees with its manifest before a pipeline is created. | `kernel::spirv`, `PipelineManager` | `spirv::tests::{rejects_truncated_and_corrupt_modules, reflects_and_checks_compute_interface_against_metadata, rejects_capabilities_not_present_in_the_device_snapshot}`; `tests/pipelines.rs::rejects_invalid_modules_before_reaching_the_driver` |
| 4 | Declared accesses induce every RAW/WAR/WAW dependency, and a cyclic graph never becomes a plan. | `TaskGraph::finalize` | `task_graph::tests::{infers_raw_war_waw_and_read_write_edges, random_access_sequences_match_independent_hazard_oracle, reports_explicit_self_and_multi_task_cycles}` |
| 5 | A reused buffer slot cannot make an old handle name a new buffer. | `BufferId` generations | `buffer::tests::cloned_handles_share_one_allocation_and_slots_are_reused_with_new_generations` |
| 6 | A device result is committed only if the host still holds the version the work started from; versions are unique across buffers. | `BufferHandle::write_bytes_if_version` | `buffer::tests::{conditional_writes_only_accept_the_expected_version, versions_are_unique_across_buffers}` |
| 7 | Mappings are exclusive and never race host writes; a re-entering thread gets an error, not a deadlock. | Mapping guards | `buffer::tests::writable_mappings_track_dirty_ranges_and_advance_versions` and the mapping re-entry tests in `buffer.rs` |
| 8 | Reserved bytes never exceed a heap's admission limit, and rejected reservations change nothing. | `MemoryManager` ledger | `manager::tests::{ledger_enforces_limit_and_recovers_after_release, ledger_checked_arithmetic_and_invalid_operations_do_not_mutate, refresh_clamps_to_heap_and_preserves_reservations}` |
| 9 | Suballocator regions cover the arena without overlap, and adjacent free regions coalesce. | `Suballocator::validate_invariants` | `suballocator::tests::{randomized_operations_match_interval_oracle, coalesces_no_left_right_and_both_neighbours, largest_free_block_matches_a_full_scan}` |
| 10 | Submissions to one queue are serialized and completion values preserve order; device loss fails queued work and blocks further use. | `QueueContext` | `queue::tests::{serializes_sixteen_concurrent_producers_and_completes_callbacks, device_lost_fails_queued_submissions_and_prevents_further_queue_use}`; `tests/queues_transfers.rs::waits_for_timeline_value_and_orders_dependent_submissions` |
| 11 | A command buffer is not reset or reused before its batch completes. | `CommandPoolSet` | `command_pool::tests::{reuses_ring_without_steady_state_allocation_or_free, full_ring_reports_out_of_memory_budget_until_work_completes}`; `tests/queues_transfers.rs::command_pool_set_repeats_ten_thousand_record_submit_reuse_cycles` |
| 12 | A staging range is not reused until the queue timeline value that used it has completed. | `StagingRing` | `tests/memory_pools.rs::staging_ring_waits_for_oldest_timeline_and_never_overlaps_in_flight` |
| 13 | A remote result is validated completely before any buffer changes, and every remote failure re-runs the job locally. | `cluster::link` | `tests/cluster.rs::{random_disconnects_never_corrupt_or_lose_work, split_survives_worker_chaos}` |

Run them with `cargo test --workspace`. Vulkan tests skip (with a printed reason) when no device is
available; concurrency and `unsafe` modules are also exercised under Miri (see
[Building and testing](../project/building.md)).
