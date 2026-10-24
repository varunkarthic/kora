#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$repo_root"

# Lines tagged `kora:identity-check` compare persisted cache identity, not runtime behavior.
# Vulkan (`vendorID`) and Rust (`vendor_id`) spellings; `match` covers Rust pattern dispatch.
id='(vendorID|vendor_id)'
pattern="(if|switch|while|match)[[:space:]]*\\(?[^){]*${id}|${id}(\\(\\))?[[:space:]]*(==|!=|<=|>=|<|>|\\?|&&|\\|\\|)|((==|!=|<=|>=|<|>)[[:space:]]*[^;]*${id})"
set +e
matches=$(grep -RInE --include='*.rs' "$pattern" crates | grep -v 'kora:identity-check')
grep_status=$?
set -e

if [ "$grep_status" -gt 1 ]; then
    echo "Unable to scan sources for vendor-dependent branches." >&2
    exit "$grep_status"
fi
if [ -n "$matches" ]; then
    printf '%s\n' "Vendor-dependent branches are not allowed in runtime sources:" "$matches" >&2
    exit 1
fi
