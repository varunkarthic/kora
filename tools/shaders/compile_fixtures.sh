#!/bin/sh
# Regenerates the committed SPIR-V test fixtures under crates/kora/tests/fixtures and the runtime's
# built-in shaders under crates/kora/shaders.
# Requires glslangValidator and spirv-as (Vulkan SDK or Homebrew glslang/spirv-tools).
# The binaries are committed so `cargo test` needs no shader toolchain.
set -eu
root=$(cd "$(dirname "$0")/../.." && pwd)
out="$root/crates/kora/tests/fixtures"
mkdir -p "$out"
for shader in "$out"/*.comp; do
    name=$(basename "$shader" .comp)
    # autotune.comp documents autotune.spvasm, which adds LocalSizeId (GLSL cannot express it).
    [ "$name" = autotune ] && continue
    glslangValidator -V --target-env vulkan1.0 -o "$out/$name.spv" "$shader" >/dev/null
done
glslangValidator -V --target-env vulkan1.0 -o "$root/crates/kora/shaders/calibration_probe.spv" \
    "$root/crates/kora/shaders/calibration_probe.comp" >/dev/null
spirv-val "$root/crates/kora/shaders/calibration_probe.spv"
spirv-as --target-env vulkan1.2 -o "$out/autotune.spv" "$out/autotune.spvasm"
for module in "$out"/*.spv; do
    spirv-val --target-env vulkan1.2 --allow-localsizeid "$module"
done
echo "fixtures written to $out"
