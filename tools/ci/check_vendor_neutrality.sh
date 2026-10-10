#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$repo_root"

pattern='(if|switch|while)[[:space:]]*\([^)]*vendorID|vendorID[[:space:]]*(==|!=|<=|>=|<|>|\?|&&|\|\|)|((==|!=|<=|>=|<|>)[[:space:]]*[^;]*vendorID)'
set +e
matches=$(grep -RInE --include='*.cpp' --include='*.hpp' "$pattern" src)
grep_status=$?
set -e

if [ "$grep_status" -gt 1 ]; then
    echo "Unable to scan src/ for vendor-dependent branches." >&2
    exit "$grep_status"
fi
if [ -n "$matches" ]; then
    printf '%s\n' "Vendor-dependent branches are not allowed in src/:" "$matches" >&2
    exit 1
fi
