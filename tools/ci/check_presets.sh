#!/usr/bin/env bash
# Configure (and optionally build) every configure preset. Usage: check_presets.sh [--build]
set -euo pipefail
cd "$(dirname "$0")/../.."
build=0; [[ "${1:-}" == "--build" ]] && build=1
for p in $(cmake --list-presets=configure | sed -n 's/^ *"\([^"]*\)".*/\1/p'); do
  echo "== preset $p"
  cmake --preset "$p" >/dev/null
  if [[ $build == 1 ]]; then cmake --build --preset "$p"; fi
done
echo "all presets OK"
