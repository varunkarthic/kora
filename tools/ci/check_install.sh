#!/usr/bin/env bash
# Install a Release build and compile a standalone find_package(kora) consumer.
set -euo pipefail
cd "$(dirname "$0")/../.."

install_dir=$(mktemp -d)
consumer_build=$(mktemp -d)
trap 'rm -rf "$install_dir" "$consumer_build"' EXIT

cmake --preset release
cmake --build --preset release --parallel
cmake --install build/release --prefix "$install_dir"
cmake -S tests/package -B "$consumer_build" -G Ninja -DCMAKE_PREFIX_PATH="$install_dir"
cmake --build "$consumer_build" --parallel
"$consumer_build/kora_package_consumer"
