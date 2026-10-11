#!/usr/bin/env bash
set -euo pipefail

log_file="$(mktemp)"
trap 'rm -f "$log_file"' EXIT

# The complete suite is also run by the preceding CI step. Re-run the integration layer with sync
# validation enabled; the 64 MiB transfer stress test is intentionally reserved for the ordinary
# suite because validation-layer tracking can make that memory-volume test excessively slow.
if ! ctest --preset debug --verbose -L integration \
  -E '^TransferEngine\.RoundTripsOddOffsetsAndSizesThroughEveryLocalVulkanDevice$' \
  2>&1 | tee "$log_file"; then
  exit 1
fi

if grep -E 'Validation Error|VUID-[A-Za-z0-9-]+' "$log_file"; then
  echo "Vulkan validation reported an error." >&2
  exit 1
fi
