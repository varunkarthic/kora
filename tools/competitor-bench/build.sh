#!/bin/sh
# Builds the standalone macOS comparison harness (Metal, OpenCL, OpenGL, Accelerate) into
# target/competitor-bench. It never links KORA, so these frameworks stay out of KORA itself.
set -eu
root=$(cd "$(dirname "$0")/../.." && pwd)
revision=$(git -C "$root" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
mkdir -p "$root/target"
clang++ -std=c++20 -O2 -fobjc-arc \
    -DCOMPETITOR_BENCH_SOURCE_REVISION="\"$revision\"" \
    "$root/tools/competitor-bench/main.mm" \
    -framework Foundation -framework Metal -framework OpenCL -framework OpenGL -framework Accelerate \
    -o "$root/target/competitor-bench"
echo "built $root/target/competitor-bench"
