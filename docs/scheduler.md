# Scheduler cost models

`kora::detail::TransferModelSet` supplies direction-specific transfer estimates for future placement
and scheduling decisions. It is a policy/model component only: Runtime dispatch does not yet consult it,
and callers must not interpret an unmeasured path as free. An estimate for a path with no observations
returns `InvalidArgument`.

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
All inputs are finite and nonnegative, and a locality benefit that would make predicted finish
negative is rejected. `ScoreBreakdown` exposes each term, including the final predicted finish; the
terms sum exactly to that value.

`QueueAvailability` accumulates estimated pending work per queue. On real completion, the caller passes
the estimated and observed duration; a configurable EWMA correction learns the estimate error while
subtracting the completed estimated work. This is a scheduling primitive, not an executor or a source
of wall-clock timestamps; the owner supplies durations from a consistent measurement domain.

Unit tests use hand-computed EFT examples to validate the additive breakdown, the 70% memory knee,
infeasibility, invalid-input rejection, pending-work accounting, and completion correction. Candidate
evaluation remains independent of Runtime placement until the future scheduler combines measured
transfer, execution, queue, and memory observations.

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
