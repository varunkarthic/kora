#!/bin/sh
# Builds libkora (kora-capi), installs it with include/kora.h into a temporary prefix, compiles the
# C example against the installed copy, and runs vector add on the first device.
# Expected result: "vector add of 4099 elements: 0 mismatches", or "skipped: ..." (exit 0) when no
# Vulkan device is available.
set -eu
root=$(cd "$(dirname "$0")/../.." && pwd)
prefix=$(mktemp -d)
trap 'rm -rf "$prefix"' EXIT
cargo build --release --locked -p kora-capi --manifest-path "$root/Cargo.toml"
mkdir -p "$prefix/include" "$prefix/lib"
cp "$root/crates/kora-capi/include/kora.h" "$prefix/include/"
for library in libkora.so libkora.dylib libkora.a; do
    if [ -f "$root/target/release/$library" ]; then cp "$root/target/release/$library" "$prefix/lib/"; fi
done
# Link the shared library (the static one would also need Rust's native dependencies).
rm -f "$prefix/lib/libkora.a"
${CC:-cc} -std=c11 -Wall -Wextra -Werror -I "$prefix/include" "$root/crates/kora-capi/examples/vector_add.c" \
    -L "$prefix/lib" -lkora -Wl,-rpath,"$prefix/lib" -o "$prefix/vector_add"
status=0
"$prefix/vector_add" "$root/crates/kora/tests/fixtures/vector_add.spv" || status=$?
if [ "$status" -eq 77 ]; then exit 0; fi
exit "$status"
