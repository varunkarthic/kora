# Workgroup autotuning

KORA can tune the X dimension of a compute workgroup when a SPIR-V entry point uses
`LocalSizeId` specialization constants. Fixed `LocalSize` entry points remain unchanged. The feature is
vendor-neutral: candidates are derived from the selected device's queried compute limits and, when
available, subgroup size. KORA does not infer behavior from vendor or device IDs.

## Candidate and measurement policy

The declared workgroup size is always included as a fallback. Additional X sizes are subgroup-size
multiples (or powers of two when subgroup information is unavailable), with the Y and Z dimensions
held at their declared defaults. Candidates that exceed a per-dimension workgroup limit or the total
invocation limit are discarded. At most eight candidates are considered.

When GPU timestamps are supported and tuning has not been disabled, KORA evaluates each candidate
with two warmup dispatches followed by five measured dispatches. These are ordinary user-requested
dispatches; KORA does not replay kernels or create hidden work. Dispatch duration samples are collected
only after the queue completion timeline has advanced. A candidate is rejected when its measured
range is more than 50% of its median. Once all candidates have enough observations, the stable
candidate with the lowest median duration is selected; if all candidates are unstable, the declared
size remains in use.

## Persistence and controls

Mature measurements are stored in `performance-db.json` under the configured cache directory. Keys
include the device identity, Vulkan driver version, kernel ID, entry point, and workgroup dimensions, so
records from another driver version are not reused. Records that are not yet mature do not bypass
warmups or measurements.

Set `KORA_DISABLE_AUTOTUNE=1` to disable runtime tuning. The setting can also be supplied through
`RuntimeOptions::disable_autotune`. With tuning disabled, without timestamp profiling, or for kernels
that do not use `LocalSizeId`, KORA uses the declared workgroup size. No tuning-specific command-line
option is required.

## Limitations

- Only the X dimension is varied; Y and Z remain at the kernel's declared sizes.
- The X specialization ID must be independent of the Y and Z IDs; otherwise the declared size is kept.
- Kernels without `LocalSizeId` cannot be specialized by the runtime and are not tuned.
- If the application does not submit enough dispatches to evaluate every candidate, the declared size
  remains active for the unevaluated candidates.
- Timing variance from unrelated GPU work can make a candidate unstable; unstable results are excluded
  rather than treated as a reliable ranking.

See [GPU timestamp profiling](profiling.md) for timestamp availability and the completion-based sample
collection model.
