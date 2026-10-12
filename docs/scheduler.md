# Scheduler cost models

`kora::detail::TransferModelSet` supplies direction-specific transfer estimates for future placement
and scheduling decisions. It is a policy/model component only: Runtime dispatch does not yet consult it,
and callers must not interpret an unmeasured path as free. An estimate for a path with no observations
returns `InvalidArgument`.

`estimate_calibration_cost()` adapts a complete, identity-matched startup calibration record into
separate workload-sized nanosecond components. It is an optional input adapter, not a selector: it
does not mutate `SchedulerCostInput`, `TransferModelSet`, queue state, or Runtime placement. A caller
may use its values when constructing candidate costs and must retain its static/online estimates when
`complete_record` is false. Runtime auto-placement remains tracked separately by #185.

## Logarithmic size buckets

`transfer_size_bucket(bytes)` uses `floor(log2(bytes))`, with zero and one byte in bucket zero. Bucket
`i` covers `[2^i, 2^(i+1)-1]` for `i` from 1 to 62; bucket zero covers zero and one, and bucket 63
extends through `UINT64_MAX`. This yields 64 stable buckets over the complete `uint64_t` range. Bucket
statistics retain the EWMA byte count, EWMA duration, and observation count independently for each
transfer path.

## Paths and latency/bandwidth fitting

The model distinguishes host-to-device, device-to-host, device copy, cross-device via host, and UMA
direct movement. The cross-device path is measured end-to-end; it is not implicitly treated as a peer
copy. Each observation is a positive byte count and a finite nonnegative duration in nanoseconds.
Within a bucket, byte count and time use the configured EWMA alpha (default 0.2). For each path, the
current EWMA pair from every populated bucket is fitted to `time = latency + ns_per_byte * bytes` by
least squares. The slope and intercept are clamped at zero if noisy or insufficient data would make
them negative. One populated bucket therefore gives a zero-latency proportional estimate. Since the
slope is nonnegative, estimated time is monotone with size.

UMA-direct penalty multiplies only the size-dependent byte cost, not fixed latency. The default is
1.0 (no penalty); a caller with evidence that direct host-visible access contends with application CPU
work can set a finite penalty of at least 1.0. This term is explicit and does not branch on device IDs,
vendor identity, or topology labels. It is not currently auto-calibrated.

## Function reference

| API | Contract |
| --- | --- |
| `transfer_size_bucket(bytes)` | Returns logarithmic bucket index and inclusive byte bounds, including a defined zero-byte bucket. |
| `TransferModelSet(options)` | Creates empty per-path bucket state; invalid alpha or direct penalty selects safe defaults. |
| `observe(path, bytes, duration_ns)` | Validates and adds an EWMA sample to the path/size bucket, then refits its latency and byte-time curve. |
| `estimate_ns(path, bytes)` | Returns a monotone modeled duration or an error when the path has no measurements. |
| `model(path)` | Returns the current observation count, fitted latency, and nanoseconds per byte. |
| `bucket_stats(path, bucket)` | Returns the bucket count and EWMA byte/time pair; an empty bucket returns zero statistics. |
| `set_uma_direct_penalty(value)` | Sets a finite multiplier of at least one for UMA-direct byte cost. |

## Calibration-to-cost adapter

The adapter accepts only a record whose calibration schema, public KORA schema, device UUID, and
driver version exactly match the requested device. Its conservative completeness threshold requires
every current calibration metric to be nonzero; any absent metric returns `complete_record == false`
and empty estimate optionals, which callers must treat as “use fallback,” not as free work. KORA's
calibration writer accepts compute metrics only after output verification. This gate is a completeness
and identity check, not a statistical confidence interval: the calibration format does not persist a
variance estimate.

Input/output/device-copy byte counts map only to their corresponding transfer fields. Kernel work is
explicitly classified as tiny dispatch count, total streaming bytes (read plus write), or ALU operation
count; ambiguous/zero kernel work is rejected. The returned terms remain separate so callers can
choose an appropriate model. In particular, `submit_completion_ns` overlaps with host-observed transfer
and kernel timings; it is populated only for a submit-only workload and is empty when any transfer or
kernel work was requested, preventing accidental double counting.

| Workload input | Calibration metric | Returned unit |
| --- | --- | --- |
| `input_bytes` | H2D bytes/second | nanoseconds |
| `output_bytes` | D2H bytes/second | nanoseconds |
| `device_copy_bytes` | device-copy bytes/second | nanoseconds |
| tiny dispatch count | tiny-dispatch nanoseconds/dispatch | nanoseconds |
| total streaming bytes | streaming bytes/second | nanoseconds |
| ALU operation count | ALU operations/second | nanoseconds |
| one standalone submission | submit/completion nanoseconds | nanoseconds |

This adapter does not establish cross-device confidence or select a device. Runtime placement continues
to honor the loaded kernel's device; P9 issue #185 owns any future automatic-placement integration.

Unit tests cover all bucket boundaries, separate per-path models, fitting, monotonicity from zero through
`UINT64_MAX`, EWMA aggregation, direct penalty behavior, and invalid inputs. The model is not yet wired
to `PerformanceDatabase` persistence or scheduler decisions; those integrations remain explicit future
work.

## Candidate cost and score breakdown

`evaluate_scheduler_cost(input)` implements the estimated-finish-time model used to compare feasible
candidate backends. `dependency_ready_ns` and `backend_available_ns` are absolute times relative to one
common scheduler clock; their maximum is represented additively as `ready_delay_ns + queue_delay_ns`.
Remaining terms are input transfer, execution, required-now output transfer, and synchronization.
Outputs that stay resident for a downstream device consumer should not be charged as immediate
readback.

Memory penalty is zero through 70% of effective budget, then grows quadratically toward the budget:
`scale * ((usage_ratio - 0.70) / 0.30)^2`. A candidate at or above 100% is infeasible and returns an
error so callers can first apply an explicit eviction/admission plan. Uncertainty adds
`lambda * sigma`; locality subtracts estimated avoided transfer cost, capped by `locality_cap_ns`.
Graph look-ahead separately adds `lookahead_beta * estimated_downstream_movement_ns` (default beta
0.25). That estimate represents movement a downstream consumer would incur if this candidate places
the current node away from its future consumer; it is distinct from the current node's input-locality
bonus. The graph planner supplies this per-candidate estimate from its known successors and measured
transfer models. All costs and beta are finite and nonnegative, and a locality benefit that would make
predicted finish negative is rejected. `ScoreBreakdown` exposes each term, including the weighted
downstream-movement penalty and final predicted finish; the terms sum exactly to that value.

For example, suppose the current kernel takes 97 ms on GPU 0 and 100 ms on GPU 1, while its 2 GiB
intermediate would cost an estimated 200 ms to move from GPU 0 to the downstream-preferred GPU 1. With
the default beta, adjusted costs are 147 ms on GPU 0 and 100 ms on GPU 1, so selection preserves the
chain locality despite GPU 0's 3% immediate advantage. This is a bounded heuristic: beta zero restores
greedy immediate-cost behavior, and the model does not itself execute or inspect a task graph.

`QueueAvailability` accumulates estimated pending work per queue. On real completion, the caller passes
the estimated and observed duration; a configurable EWMA correction learns the estimate error while
subtracting the completed estimated work. This is a scheduling primitive, not an executor or a source
of wall-clock timestamps; the owner supplies durations from a consistent measurement domain.

| API or field | Contract |
| --- | --- |
| `evaluate_scheduler_cost(input)` | Returns immediate EFT components plus `lookahead_beta * estimated_downstream_movement_ns`; rejects invalid or infeasible estimates. |
| `SchedulerCostInput::estimated_downstream_movement_ns` | Caller-supplied measured-model estimate of bytes/time movement induced in known downstream consumers for this candidate device. |
| `SchedulerCostInput::lookahead_beta` | Multiplier constrained to `[0, 1]`, default `0.25`; set to zero for immediate-only selection. |
| `ScoreBreakdown::downstream_movement_penalty_ns` | Exposes the weighted downstream movement added to the final predicted finish. |

Unit tests use hand-computed EFT examples to validate the additive breakdown, the 70% memory knee,
infeasibility, invalid-input rejection, pending-work accounting, and completion correction. A simulated
2 GiB chain uses the measured transfer-model interface and deterministic candidate selection to verify
the look-ahead term can outweigh a small immediate execution-time advantage. Candidate evaluation
remains independent of Runtime placement until the future scheduler combines measured transfer,
execution, queue, and memory observations.

## Candidate selection policy

`select_scheduler_candidate()` applies eligibility constraints before ranking or exploration. A
candidate must be supported, healthy, within memory budget, allowed by user policy, and match task
affinity; exploration can never bypass these checks. If none qualify, selection returns
`NoEligibleDevice`.

Candidates with observations use their supplied cost estimate; unobserved candidates use their
coarse estimate when present, otherwise `cold_start_estimate_ns`. Equal unobserved estimates prefer a
caller-marked known-safe candidate, followed by ascending candidate ID. This establishes a predictable
cold-start fallback without encoding device vendor/type preferences.

Exploration is caller-randomized through a draw in `[0,1)`: probability decays linearly from 5% to 1%
as the least-observed eligible candidate reaches the configured stable-observation threshold. Exploration
is limited to candidates within both the configured relative and absolute plausibility window of the
current best score. Determinism mode sets probability to zero and always uses stable score/ID order.

Hysteresis retains the current feasible candidate unless an alternative improves its score by at least
both the absolute and relative margins. Priority aging discounts cost with wait duration and priority
weight, capped by `maximum_aging_discount_ns`; it changes ranking hints only and makes no hard
real-time guarantee. Unsupported or unhealthy candidates remain excluded regardless of priority.

The policy accepts candidate estimates rather than invoking Runtime, executing work, or generating
random values internally. This makes the decision boundary replayable and testable; the future runtime
scheduler is responsible for constructing candidates, sampling measurements, and supplying a random
draw only when nondeterministic mode is enabled.

## Small-dispatch coalescing

Runtime compute queues enable a bounded queue-level coalescing window (100 microseconds, at most eight
dispatch records). Dispatches are eligible only when they share the exact cached pipeline/layout key,
declare no explicit dependencies, have identical external semaphore waits, and have no cross-dispatch
read/write conflict on a generation-aware buffer identity. Identical waits avoid delaying an otherwise
independent dispatch behind a later transfer prerequisite. Compatible command buffers are recorded
independently and submitted in one Vulkan queue-submit call with a shared completion timeline value;
every dispatch still receives its own Event completion callback. Read/read sharing is allowed, while
any shared write forces separate submissions.

High and real-time-hint priorities bypass the waiting window and are never merged. Low/normal work may
wait up to the bounded window for compatible peers; an incompatible or elevated-priority queue head
ends collection. Fence-only queue contexts disable coalescing because dispatch Events rely on a shared
timeline completion. The optimization does not delay work beyond the configured window and does not
batch work with unresolved task dependencies.

Queue-level fault-injection coverage submits 32 compatible tiny-dispatch records and verifies that
fewer queue-submit calls carry all 32 command buffers and callbacks, with shared timeline values within
each batch. Additional tests ensure high-priority and conflicting writes remain separate. Coalescing is
an overhead optimization only; it does not merge task semantics or promise lower latency for large work.
