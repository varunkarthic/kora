#!/usr/bin/env python3
"""Compare KORA with Metal, OpenCL, OpenGL, and Accelerate on the same machine.

Takes interleaved kora-bench reports and competitor-bench reports (tools/competitor-bench) and
prints Markdown tables of the median across runs of each run's p50, then writes SVG charts.
Every timing domain stays separate: host submit-to-completion, device timestamps, CPU execution.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import statistics
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


charts = _load("render_charts")
comparison = _load("render_comparison")

APIS = ("metal", "opencl", "opengl")
API_LABELS = {"metal": "Metal", "opencl": "OpenCL 1.2", "opengl": "OpenGL 4.1"}
VECTOR_SIZES = ((4096, "4K"), (65536, "64K"), (1048576, "1M"), (4194240, "4M"))


def cases() -> list[dict[str, Any]]:
    """Matched cases with the result name each tool uses for them."""
    rows = []
    for group, label, case in charts.COMPUTE_CASES:
        upload = case.endswith("_upload")
        workload, _, size_text = case.removesuffix("_upload").rpartition("_")
        size = int(size_text)
        suffix = "_upload_dispatch_wait" if upload else "_dispatch_wait"
        rows.append({"group": group, "label": label, "workload": workload, "size": size, "case": case, "suffix": suffix})
    for size, label in VECTOR_SIZES:
        rows.append({"group": "Vector add", "label": label, "workload": "vector_add", "size": size,
                     "case": f"vector_add_{size}", "suffix": "_dispatch_wait"})
    return rows


def load(paths: list[Path], schemas: tuple[int, ...]) -> list[dict[str, dict[str, Any]]]:
    runs = []
    for path in paths:
        data = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
        report = json.loads(data.decode("utf-8"))
        if report.get("schema_version") not in schemas:
            raise ValueError(f"{path} must use schema version {' or '.join(map(str, schemas))}")
        if report.get("diagnostic_profile") is True:
            raise ValueError(f"{path} is a diagnostic profile, not a comparison report")
        runs.append({item["name"]: item for item in report["results"]})
    if not runs:
        raise ValueError("at least one report is required")
    return runs


def summary(runs: list[dict[str, dict[str, Any]]], name: str, domain: str) -> dict[str, float] | None:
    found = [run[name] for run in runs if name in run]
    if len(found) != len(runs):
        return None  # Missing from some run (e.g. timer unavailable): report as a gap, never estimate.
    for item in found:
        if item.get("timing_domain") != domain:
            raise ValueError(f"{name} must use {domain}")
    return {stat: statistics.median(item[stat] for item in found) for stat in ("median", "p95")}


def cell(value: dict[str, float] | None) -> str:
    return comparison.format_ns(int(value["median"])) if value else "—"


def render(kora_paths: list[Path], competitor_paths: list[Path], output_dir: Path | None) -> str:
    kora = load(kora_paths, (3, 4))  # Schema 4 (1.0) adds fields to schema 3 (0.x).
    competitors = load(competitor_paths, (1,))
    host_domain, device_domain, cpu_domain = "host_submit_completion", "gpu_execution_timestamp", "host_cpu_execution"
    lines = [
        f"{len(kora_paths)} kora-bench and {len(competitor_paths)} competitor-bench runs, interleaved; each value is "
        "the median across runs of the per-run p50.",
        "",
        "### Host submit to completion",
        "",
        "| Workload | Size | KORA default | KORA deferred | Raw Vulkan | Metal | OpenCL 1.2 | OpenGL 4.1 | "
        "KORA deferred vs fastest other GPU path |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    series_names = ("KORA default readback", "KORA deferred readback", "Raw Vulkan", *API_LABELS.values())
    host_points: dict[str, list[dict[str, Any]]] = {name: [] for name in series_names}
    device_points: dict[str, list[dict[str, Any]]] = {name: [] for name in ("KORA Runtime", "Raw Vulkan", "Metal", "OpenGL 4.1")}
    device_lines = [
        "",
        "### Device time",
        "",
        "Apple's OpenCL profiling counters report physically impossible durations on this hardware, so OpenCL has "
        "no device-time column. OpenGL timer queries return no value for many small draws; those are gaps.",
        "",
        "| Workload | Size | KORA | Raw Vulkan | Metal | OpenGL 4.1 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in cases():
        key = f"{row['group']}:{row['label']}"
        size = row["size"]
        values = {
            "KORA default readback": summary(kora, f"gpu_{row['case']}_dispatch_wait", host_domain),
            "KORA deferred readback": summary(kora, f"gpu_{row['case']}_deferred_readback_dispatch_wait", host_domain),
            "Raw Vulkan": summary(kora, f"raw_vulkan_{row['case']}_dispatch_wait", host_domain),
        }
        for api, label in API_LABELS.items():
            values[label] = summary(competitors, f"{api}_{row['workload']}_{size}{row['suffix']}", host_domain)
        others = [values[name] for name in ("Raw Vulkan", *API_LABELS.values()) if values[name]]
        deferred = values["KORA deferred readback"]
        ratio = (
            f"{deferred['median'] / min(item['median'] for item in others):.2f}×" if deferred and others else "—"
        )
        lines.append(
            "| " + " | ".join([row["group"], row["label"], *(cell(values[name]) for name in series_names), ratio]) + " |"
        )
        for name in series_names:
            if values[name]:
                host_points[name].append({"size": key, **values[name]})
        device = {
            "KORA Runtime": summary(kora, f"gpu_{row['workload']}_{size}_gpu_execution", device_domain),
            "Raw Vulkan": summary(kora, f"raw_vulkan_{row['case']}_gpu_execution", device_domain),
            "Metal": summary(competitors, f"metal_{row['workload']}_{size}_gpu_execution", device_domain),
            "OpenGL 4.1": summary(competitors, f"opengl_{row['workload']}_{size}_gpu_execution", device_domain),
        }
        device_lines.append(
            "| " + " | ".join([row["group"], row["label"], *(cell(device[name]) for name in device_points)]) + " |"
        )
        for name in device_points:
            if device[name]:
                device_points[name].append({"size": key, **device[name]})

    cpu_lines = [
        "",
        "### CPU execution",
        "",
        "Accelerate is Apple's tuned CPU library and may use several cores; KORA's CPU rows are its portable "
        "single-thread and four-worker baselines. Accelerate has no 64-wide tiled integer scan.",
        "",
        "| Workload | Size | KORA CPU (1 thread) | KORA CPU (4 workers) | Accelerate |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in cases():
        workload, size = row["workload"], row["size"]
        serial = summary(kora, f"cpu_{workload}_{size}", cpu_domain)
        parallel = summary(kora, f"cpu_{workload}_parallel_{size}", cpu_domain)
        accelerate = summary(competitors, f"accelerate_{workload}_{size}", cpu_domain)
        if serial or parallel or accelerate:
            cpu_lines.append(
                "| " + " | ".join([row["group"], row["label"], cell(serial), cell(parallel), cell(accelerate)]) + " |"
            )
    lines += device_lines + cpu_lines
    lines += [
        "",
        "Metal, OpenCL, and raw Vulkan keep outputs in device-visible memory, like KORA deferred; KORA default also "
        "copies outputs back into host buffers (the vector-add rows read back a 16 MiB buffer at every size). "
        "OpenGL 4.1 has no compute shaders: its rows run vertex shaders with transform feedback, and its reduction, "
        "scan, and matmul loop over inputs without shared memory, so they are not algorithm-matched.",
    ]

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        keys = tuple(f"{row['group']}:{row['label']}" for row in cases())
        labels = {f"{row['group']}:{row['label']}": row["label"] for row in cases()}
        groups = {f"{row['group']}:{row['label']}": row["group"] for row in cases()}
        colors = {
            "KORA default readback": "#c2410c",
            "KORA deferred readback": "#2563eb",
            "KORA Runtime": "#2563eb",
            "Raw Vulkan": "#15803d",
            "Metal": "#7c3aed",
            "OpenCL 1.2": "#0891b2",
            "OpenGL 4.1": "#a16207",
        }
        for filename, title, points in (
            ("competitors-host.svg", "KORA vs Metal, OpenCL, OpenGL — host submit to completion", host_points),
            ("competitors-device.svg", "KORA vs Metal and OpenGL — device timestamp", device_points),
        ):
            series = [{"label": name, "color": colors[name], "points": items} for name, items in points.items()]
            svg = charts.render_svg(
                f"{title} · Apple M4",
                f"{title}. Median across interleaved runs of each run's p50 (dot) and p95 (whisker). OpenGL rows for "
                "reduction, scan, and matmul are emulations without shared memory.",
                series,
                keys,
                x_label="Workload · problem size (elements, or matrix dimension²)",
                labels=labels,
                groups=groups,
                width=1520,
            )
            (output_dir / filename).write_text(svg, encoding="utf-8", newline="\n")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kora", type=Path, nargs="+", required=True, help="kora-bench --gpu reports")
    parser.add_argument("--competitors", type=Path, nargs="+", required=True, help="competitor-bench reports")
    parser.add_argument("--output-dir", type=Path, help="write competitors-host.svg and competitors-device.svg here")
    args = parser.parse_args()
    try:
        print(render(args.kora, args.competitors, args.output_dir), end="")
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
