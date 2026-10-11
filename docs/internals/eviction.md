# Buffer-eviction policy internals

KORA's eviction policy ranks reclaimable buffer copies when an admission/recovery caller needs
additional capacity. It uses a bounded LRU-2 history rather than raw LRU: a single accidental access
does not earn the recency bonus, while two or more recent references do. The policy is deterministic
for a fixed candidate list, history, and planning tick.

## Keep value

Each eligible candidate receives this score; lower values are evicted first:

```text
KeepValue =
    reuse_probability * estimated_reload_cost_ms
  + recent_access_bonus
  + user_priority_bonus
  + expensive_recreation_bonus
  + required_writeback_cost_ms
  - idle_age_penalty
```

`recent_access_bonus` is zero until the LRU-2 history has two accesses. Thereafter it is
`recent_access_weight / (1 + age_of_second_most_recent_access)`. The idle penalty is
`idle_age_penalty_per_tick * age_since_most_recent_access`; callers use one monotonic tick domain for
both `record_access()` and planning. A candidate without recorded accesses uses its supplied
`last_access_tick`. When device data is newer and no other current copy exists, estimated writeback
cost is included in the keep value and the returned action is `WritebackThenEvict`. Disposable data
or a valid alternate current copy can be evicted directly.

## Eligibility and safety

Candidates with a nonzero pin count, in-flight work, or active submission are never selected. Plans
are complete-or-error: if all eligible bytes cannot satisfy `requested_bytes - available_bytes`, the
planner returns `OutOfMemoryBudget` without returning a partial eviction plan. Each decision carries
the full `BufferId` slot and generation, so a caller must revalidate identity and eligibility before
performing writeback or freeing storage. Planning itself performs no I/O, changes no residency, and
does not free memory. Runtime dispatch records generation-aware buffer touches in this policy, but
selection remains separate from the queue/timeline-specific operation that must preserve buffer
lifetime. Runtime admission/recovery integration is handled separately from this policy component.

## Function reference

| Function | Contract |
| --- | --- |
| `AccessHistory::record(tick)` | Records monotonic access time in a fixed-size LRU-2 ring; backward time returns `InvalidArgument` without replacing the history. |
| `AccessHistory::kth_recent_age(now_tick)` | Returns the age of the second-most-recent reference only after two references have been observed. |
| `EvictionPolicy::record_access(id, tick)` | Tracks access history by the complete slot/generation identity and reports allocation or time-order errors. |
| `EvictionPolicy::forget(id)` | Removes history for a reclaimed or reused identity. |
| `EvictionPolicy::keep_value(candidate, now_tick)` | Computes the documented score and rejects invalid, non-finite, or temporally inconsistent inputs. |
| `EvictionPolicy::plan(candidates, available, requested, now_tick)` | Returns the lowest-value eligible decisions until the byte deficit is covered, or an error with no partial plan. |

`EvictionPolicy` is a policy object and is not internally synchronized. The caller serializes access
history updates, snapshots, and plan application, and must re-check pins, submission state, timeline
completion, and generation immediately before eviction. Unit tests cover synthetic score ordering,
LRU-2 behavior, protected candidates, dirty-copy writeback decisions, insufficient capacity, and
low-budget planning with generation-aware IDs.

Run the focused tests with:

```sh
cmake --preset debug
cmake --build --preset debug --parallel
ctest --preset debug --output-on-failure -R 'EvictionPolicy|AccessHistory'
```
