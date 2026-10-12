#!/usr/bin/env python3
"""Render matched CPU, KORA/Vulkan, and raw Vulkan vector-add rows from a kora-bench report."""

import argparse
import json
from pathlib import Path


def format_ns(value: int) -> str:
    if value < 1_000:
        return f"{value} ns"
    return f"{value / 1_000:,.3f} µs" if value < 1_000_000 else f"{value / 1_000_000:,.3f} ms"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="schema-3 kora-bench JSON report")
    parser.add_argument("--sizes", nargs="+", type=int, default=[4096, 1048576, 4194240])
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("schema_version") != 3:
        parser.error("expected kora-bench JSON schema version 3")
    if "gpu_measurements" not in report:
        parser.error("report must include --gpu measurements")
    if not report["gpu_measurements"].get("same_physical_device_for_kora_and_raw_vulkan"):
        parser.error("KORA and raw Vulkan must identify the same physical device")
    results = {item["name"]: item for item in report["results"]}

    print(
        "| Elements | CPU median | CPU p95 | KORA default median | KORA default p95 | "
        "KORA deferred median | KORA deferred p95 | Raw host median | Raw host p95 | "
        "KORA GPU median | KORA GPU p95 | Raw GPU median | Raw GPU p95 |"
    )
    print("| " + " | ".join(["---:"] * 13) + " |")
    for size in args.sizes:
        names = (
            f"cpu_vector_add_{size}",
            f"gpu_vector_add_{size}_dispatch_wait",
            f"gpu_vector_add_{size}_deferred_readback_dispatch_wait",
            f"raw_vulkan_vector_add_{size}_dispatch_wait",
        )
        try:
            cpu, kora, kora_deferred, raw = (results[name] for name in names)
        except KeyError as error:
            parser.error(f"report is missing matched workload {error.args[0]}")
        gpu_names = (
            f"gpu_vector_add_{size}_gpu_execution",
            f"raw_vulkan_vector_add_{size}_gpu_execution",
        )
        gpu_results = [results.get(name) for name in gpu_names]
        values = [
            str(size),
            format_ns(cpu["median"]),
            format_ns(cpu["p95"]),
            format_ns(kora["median"]),
            format_ns(kora["p95"]),
            format_ns(kora_deferred["median"]),
            format_ns(kora_deferred["p95"]),
            format_ns(raw["median"]),
            format_ns(raw["p95"]),
            *[
                format_ns(result[stat]) if result is not None else "unavailable"
                for result in gpu_results
                for stat in ("median", "p95")
            ],
        ]
        print("| " + " | ".join(values) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
