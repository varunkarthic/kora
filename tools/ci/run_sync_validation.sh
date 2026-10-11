#!/usr/bin/env bash
set -euo pipefail

log_file="$(mktemp)"
trap 'rm -f "$log_file"' EXIT

# The complete suite is also run by the preceding CI step. Re-run the queue and Runtime integration
# paths with synchronization validation enabled. Large memory-volume transfer coverage remains in
# the ordinary full suite; the full sync-validation suite is exercised on the maintainer host.
if ! ctest --preset debug --verbose \
  -R '^(Runtime\.DispatchesVectorAddOnEveryDeviceWithTimelineAndFenceFallback|QueueContext\.(SelectsSynchronization2OnlyWhenEnabledOnLogicalDevice|CompletesConcurrentRealVulkanSubmissions|WaitsForTimelineValueAndOrdersDependentSubmissions))$' \
  2>&1 | tee "$log_file"; then
  exit 1
fi

if grep -E 'Validation Error|VUID-[A-Za-z0-9-]+' "$log_file"; then
  echo "Vulkan validation reported an error." >&2
  exit 1
fi
