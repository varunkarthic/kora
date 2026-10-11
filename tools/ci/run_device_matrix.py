#!/usr/bin/env python3
"""Run the CTest suite once per locally available KORA reference device."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


TARGETS = (
    ("KosmicKrisp", lambda device: "kosmickrisp" in device["driver_name"].lower()),
    ("llvmpipe", lambda device: "llvmpipe" in device["driver_name"].lower()),
    ("MoltenVK", lambda device: device.get("moltenvk") is True),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("build/debug"),
        help="CMake build directory containing CTestTestfile.cmake (default: build/debug)",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="fail unless all three reference driver types are present",
    )
    args = parser.parse_args()

    build_dir = args.build_dir.resolve()
    if not (build_dir / "CTestTestfile.cmake").is_file():
        parser.error(f"{build_dir} is not a configured CMake test directory")

    suffix = ".exe" if sys.platform == "win32" else ""
    kora_info = build_dir / "tools" / f"kora-info{suffix}"
    if not kora_info.is_file():
        parser.error(f"kora-info was not found at {kora_info}; configure with KORA_BUILD_TOOLS=ON")
    ctest = shutil.which("ctest")
    if ctest is None:
        parser.error("ctest was not found on PATH")

    probe = subprocess.run(
        [str(kora_info), "--json"],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        sys.stderr.write(probe.stderr)
        return probe.returncode
    try:
        devices = json.loads(probe.stdout)["devices"]
        if not isinstance(devices, list):
            raise ValueError("devices is not an array")
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        print(f"kora-info returned invalid device JSON: {error}", file=sys.stderr)
        return 2

    selected: list[tuple[str, dict[str, object]]] = []
    missing: list[str] = []
    for label, matches in TARGETS:
        device = next(
            (candidate for candidate in devices if isinstance(candidate, dict) and matches(candidate)),
            None,
        )
        if device is None:
            missing.append(label)
        else:
            selected.append((label, device))

    for label in missing:
        print(f"SKIP {label}: no matching Vulkan device was reported", file=sys.stderr)
    if args.require_all and missing:
        return 1
    if not selected:
        print("No reference devices are available; no CTest suite was run", file=sys.stderr)
        return 1

    for label, device in selected:
        index = device.get("index")
        if not isinstance(index, int) or index < 0:
            print(f"Device entry for {label} has no valid enumeration index", file=sys.stderr)
            return 2
        name = device.get("name", "unknown device")
        print(f"\n=== {label}: device {index} ({name}) ===", flush=True)
        environment = os.environ.copy()
        environment["KORA_DEVICE"] = str(index)
        result = subprocess.run(
            [ctest, "--test-dir", str(build_dir), "--output-on-failure"],
            check=False,
            env=environment,
        )
        if result.returncode != 0:
            print(f"CTest failed for {label} (device {index})", file=sys.stderr)
            return result.returncode
        print(f"PASS {label}: full CTest suite completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
