# Performance database

`kora::detail::PerformanceDatabase` is an internal, thread-safe store for scheduler measurements
keyed by semantic operation strings. A sample is a finite, nonnegative scalar in the caller's chosen
unit; the database does not infer units or combine unlike measurements. The bounded online workgroup
autotuner also uses it to persist GPU dispatch durations for `LocalSizeId` variants; the scheduler
does not yet use these records for automatic placement decisions.

## Statistics and robustness

Each key has an independent mutex, so updates and snapshots for unrelated keys do not serialize on
one global lock. A shared registry lock protects key lookup/insertion. The EWMA uses
`mean += alpha * (sample - mean)`. Its exponentially weighted population-variance recurrence is
`variance = (1 - alpha) * (variance + alpha * delta^2)`, where `delta` is measured against the prior
mean. Alpha must be in `(0, 1]`; invalid configuration resets to conservative defaults.

Each key retains a bounded ring represented by the most recent `sample_window` accepted values (256 by
default). Median and p95 use nearest-rank order statistics over that window; these are exact for the
retained window, not estimates over lifetime history. Once at least five samples exist, an incoming
sample is rejected when its distance from the window median exceeds six robust scales. Robust scale is
`max(1.4826 * MAD, max(1, abs(median) * 0.01))`; MAD is the median absolute deviation. Rejected samples
increment `rejected_outliers`, but do not alter the accepted count, EWMA, variance, range, quantiles,
or last-accepted-observation timestamp. The first five accepted observations bootstrap the filter.

When a new observation arrives more than `maximum_age_ms` after the last observation, stale history for
that key is reset before the sample is processed. Load drops expired rows using the supplied current
Unix-millisecond timestamp. The default age is 30 days.

## Persistence and failure policy

`save(path)` writes a deterministic, schema-versioned JSON snapshot to a unique sibling temporary file
and atomically renames it into place. Windows uses replace-existing/write-through semantics; POSIX
uses atomic rename and best-effort directory synchronization. The bounded reader rejects malformed,
oversized, duplicate-key, incompatible-schema, non-finite, or invalid sample data. A failed `load()`
returns `IoError` and leaves the current in-memory database unchanged. Startup code treats the file as
an optional cache and must ignore a failed load rather than fail Runtime creation.

Workgroup autotuning keys include device identity, driver version, kernel ID plus a module fingerprint,
entry point, and all three local dimensions. This prevents stale measurements from crossing a driver
update or a changed SPIR-V module that reused a kernel ID.

## Function reference

| API | Contract |
| --- | --- |
| `PerformanceDatabase(options)` | Creates an empty store. Invalid EWMA alpha or zero sample-window size selects defaults. |
| `observe(key, sample, now_unix_ms)` | Validates input, applies per-key aging and MAD rejection, and updates statistics. |
| `query(key)` | Returns a consistent per-key statistics snapshot or `InvalidArgument` for an unknown key. |
| `size()` | Returns the number of known keys. |
| `save(path)` | Writes a sorted-key schema-1 snapshot using an atomic replacement. |
| `load(path, now_unix_ms)` | Replaces current entries only after complete validation; expired rows are omitted. |

Unit tests cover EWMA/variance math, outlier filtering, bounded-window quantiles, aging, atomic
round-tripping with escaped keys, corruption recovery, expired rows, and concurrent per-key updates.
