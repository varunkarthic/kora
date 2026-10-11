#!/usr/bin/env bash
set -euo pipefail

log_file="$(mktemp)"
trap 'rm -f "$log_file"' EXIT

if ! ctest --preset debug --verbose 2>&1 | tee "$log_file"; then
  exit 1
fi

if grep -E 'Validation Error|VUID-[A-Za-z0-9-]+' "$log_file"; then
  echo "Vulkan validation reported an error." >&2
  exit 1
fi
