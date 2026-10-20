#!/usr/bin/env python3
"""Render accessible, timing-domain-separated SVG charts from schema-3 reports."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any


SIZES = (4096, 1048576, 4194240)
SERIES = {
    "cpu": (("CPU arithmetic", "#6b7280", "cpu_vector_add_{size}", "host_cpu_execution"),),
    "host": (
        ("KORA default readback", "#c2410c", "gpu_vector_add_{size}_dispatch_wait", "host_submit_completion"),
        (
            "KORA deferred readback",
            "#2563eb",
            "gpu_vector_add_{size}_deferred_readback_dispatch_wait",
            "host_submit_completion",
        ),
        (
            "Raw Vulkan",
            "#15803d",
            "raw_vulkan_vector_add_{size}_dispatch_wait",
            "host_submit_completion",
        ),
    ),
    "gpu": (
        ("KORA Runtime", "#7c3aed", "gpu_vector_add_{size}_gpu_execution", "gpu_execution_timestamp"),
        (
            "Raw Vulkan",
            "#0891b2",
            "raw_vulkan_vector_add_{size}_gpu_execution",
            "gpu_execution_timestamp",
        ),
    ),
    "compute": (
        ("KORA default readback", "#c2410c", "gpu_{size}_dispatch_wait", "host_submit_completion"),
        (
            "KORA deferred readback",
            "#2563eb",
            "gpu_{size}_deferred_readback_dispatch_wait",
            "host_submit_completion",
        ),
        ("Raw Vulkan", "#15803d", "raw_vulkan_{size}_dispatch_wait", "host_submit_completion"),
    ),
}

# Matched compute cases: (workload group, tick label, result-name case).
COMPUTE_CASES = tuple(
    (group, label, case)
    for group, workload, suffix, tiers in (
        ("SAXPY", "saxpy", "_upload", ((4096, "4K"), (65536, "64K"), (1048576, "1M"))),
        ("Reduction", "reduction", "", ((4096, "4K"), (65536, "64K"), (1048576, "1M"))),
        ("Scan", "tiled_scan", "", ((4096, "4K"), (65536, "64K"), (1048576, "1M"))),
        ("Stencil", "stencil", "", ((4096, "4K"), (65536, "64K"), (1048576, "1M"))),
        ("Tiled matmul", "tiled_matmul", "", ((16, "16²"), (64, "64²"), (128, "128²"))),
    )
    for size, label in tiers
    for case in (f"{workload}_{size}{suffix}",)
)


def _valid_ns(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def report_series(report: dict[str, Any], kind: str, sizes: tuple[int | str, ...] = SIZES) -> list[dict[str, Any]]:
    """Validate report identity/schema and return only available, correctly-domain-tagged values."""
    if report.get("diagnostic_profile") is True:
        raise ValueError("diagnostic profile reports include telemetry overhead and are not comparison charts")
    if report.get("schema_version") != 3:
        raise ValueError("expected kora-bench JSON schema version 3")
    measurements = report.get("gpu_measurements")
    if not isinstance(measurements, dict) or not measurements.get("same_physical_device_for_kora_and_raw_vulkan"):
        raise ValueError("report must confirm a shared physical device for KORA and raw Vulkan")
    results = report.get("results")
    if not isinstance(results, list):
        raise ValueError("report results must be an array")
    by_name: dict[str, dict[str, Any]] = {}
    for item in results:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError("report contains a malformed result")
        if item["name"] in by_name:
            raise ValueError(f"duplicate result name: {item['name']}")
        by_name[item["name"]] = item

    expected_series = SERIES[kind]
    chart_series = []
    for label, color, name_template, domain in expected_series:
        points = []
        for size in sizes:
            result = by_name.get(name_template.format(size=size))
            if result is None:
                continue  # Missing GPU timer samples are deliberate gaps, never zeroes.
            if result.get("timing_domain") != domain or result.get("unit") != "ns":
                raise ValueError(f"{result.get('name')} must use {domain} in ns")
            if not _valid_ns(result.get("median")) or not _valid_ns(result.get("p95")):
                raise ValueError(f"{result.get('name')} must have finite positive median and p95 values")
            samples = result.get("samples")
            if not isinstance(samples, list) or not samples or any(not _valid_ns(sample) for sample in samples):
                raise ValueError(f"{result.get('name')} must retain finite positive raw samples")
            if result["p95"] < result["median"]:
                raise ValueError(f"{result.get('name')} p95 must not be below its median")
            points.append({"size": size, "median": result["median"], "p95": result["p95"]})
        chart_series.append({"label": label, "color": color, "points": points})
    return chart_series


def _size_label(size: int | str) -> str:
    return f"{size:,}" if isinstance(size, int) else size


def render_svg(
    title: str,
    description: str,
    series: list[dict[str, Any]],
    sizes: tuple[int | str, ...],
    *,
    x_label: str = "Vector elements · problem size",
    labels: dict[Any, str] | None = None,
    groups: dict[Any, str] | None = None,
    width: int = 1040,
) -> str:
    """Render p50/p95 series; ``groups`` labels contiguous categories and breaks lines between them."""
    points = [point for item in series for point in item["points"]]
    if not points:
        raise ValueError("cannot render a chart without measurements")
    labels = labels or {}
    groups = groups or {}

    height = 660
    left, right, top, bottom = 112, 44, 112, 180
    plot_width, plot_height = width - left - right, height - top - bottom
    minimum = min(point["median"] for point in points)
    maximum = max(point["p95"] for point in points)
    min_exp = math.floor(math.log10(minimum))
    max_exp = math.ceil(math.log10(maximum))
    if min_exp == max_exp:
        min_exp -= 1
        max_exp += 1
    lower, upper = 10.0**min_exp, 10.0**max_exp
    unit_scale, unit = (1_000_000.0, "ms") if maximum >= 1_000_000 else (1_000.0, "µs") if maximum >= 1_000 else (1.0, "ns")

    def y(value: float) -> float:
        fraction = (math.log10(value) - math.log10(lower)) / (math.log10(upper) - math.log10(lower))
        return top + plot_height * (1.0 - fraction)

    escaped_title = html.escape(title)
    escaped_description = html.escape(description)
    heading, separator, detail = title.partition(" — ")
    heading = html.escape(heading)
    detail = html.escape(detail if separator else "")
    timing_label = "DEVICE TIMESTAMP" if "timestamp" in title.lower() else "HOST SUBMIT + COMPLETION" if "completion" in title.lower() else "CPU HOST EXECUTION"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{escaped_title}</title><desc id="desc">{escaped_description}</desc>',
        '<style>text{font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;fill:#17243b}.muted{fill:#607089}.grid{stroke:#dbe3ef;stroke-width:1}.axis{stroke:#8b9bb1;stroke-width:1.25}.whisker{stroke-width:2.5;stroke-linecap:round}.median{stroke:#fff;stroke-width:2}.series-line{fill:none;stroke-width:2.5;stroke-linecap:round;stroke-linejoin:round;opacity:.72}.card{fill:#fff;stroke:#dce5f1;stroke-width:1.3}.plot-bg{fill:#f8fafd}.pill{fill:#eaf0f8}.tick{font-variant-numeric:tabular-nums}</style>',
        f'<rect x="1" y="1" width="{width-2}" height="{height-2}" rx="18" fill="#f2f6fb" stroke="#dce5f1"/>',
        f'<rect class="card" x="18" y="18" width="{width-36}" height="{height-36}" rx="14"/>',
        f'<text x="{left}" y="55" font-size="27" font-weight="700">{heading}</text>',
        f'<text class="muted" x="{left}" y="82" font-size="15">{detail}</text>',
        f'<rect class="pill" x="{width-right-312}" y="37" width="312" height="30" rx="15"/>',
        f'<text x="{width-right-156}" y="57" text-anchor="middle" font-size="11" font-weight="700" letter-spacing="1.1">{timing_label} · P50 / P95</text>',
        f'<rect class="plot-bg" x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" rx="8"/>',
    ]
    for exponent in range(min_exp, max_exp + 1):
        value = 10.0**exponent
        y_pos = y(value)
        parts.append(f'<line class="grid" x1="{left}" y1="{y_pos:.1f}" x2="{width-right}" y2="{y_pos:.1f}"/>')
        parts.append(f'<text class="muted tick" x="{left-14}" y="{y_pos+5:.1f}" text-anchor="end" font-size="13">{value/unit_scale:g}</text>')
    parts.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}"/>',
            f'<text class="muted" x="29" y="{top + plot_height/2:.1f}" transform="rotate(-90 29 {top + plot_height/2:.1f})" text-anchor="middle" font-size="13">Latency ({unit}, logarithmic scale)</text>',
            f'<text class="muted" x="{left + plot_width/2:.1f}" y="{height-bottom+(72 if groups else 57)}" text-anchor="middle" font-size="13">{html.escape(x_label)}</text>',
        ]
    )

    centers = {size: left + plot_width * (index + 0.5) / len(sizes) for index, size in enumerate(sizes)}
    group_width = plot_width / max(1, len(sizes))
    for size in sizes:
        parts.append(f'<text class="muted tick" x="{centers[size]:.1f}" y="{height-bottom+26}" text-anchor="middle" font-size="13">{html.escape(labels.get(size, _size_label(size)))}</text>')
    group_members: dict[str, list[Any]] = {}
    for size in sizes:
        if size in groups:
            group_members.setdefault(groups[size], []).append(size)
    for index, (group, members) in enumerate(group_members.items()):
        center = sum(centers[size] for size in members) / len(members)
        parts.append(f'<text x="{center:.1f}" y="{height-bottom+47}" text-anchor="middle" font-size="13" font-weight="600">{html.escape(group)}</text>')
        if index:
            divider = centers[members[0]] - group_width / 2
            parts.append(f'<line class="grid" x1="{divider:.1f}" y1="{top}" x2="{divider:.1f}" y2="{height-bottom}"/>')
    for series_index, item in enumerate(series):
        count = len(series)
        offset = (series_index - (count - 1) / 2) * min(20, group_width / max(2, count + 1))
        size_indices = {size: index for index, size in enumerate(sizes)}
        ordered = sorted(item["points"], key=lambda point: size_indices[point["size"]])
        commands = []
        previous_index = None
        previous_size = None
        for point in ordered:
            current_index = size_indices[point["size"]]
            contiguous = previous_index is not None and current_index == previous_index + 1
            same_group = groups.get(point["size"]) == groups.get(previous_size)
            command = "L" if contiguous and same_group else "M"
            previous_size = point["size"]
            commands.append(command + f" {centers[point['size']] + offset:.1f} {y(point['median']):.1f}")
            previous_index = current_index
        line_path = " ".join(commands)
        if len(ordered) > 1:
            parts.append(f'<path class="series-line" stroke="{item["color"]}" d="{line_path}"/>')
        for point in ordered:
            x_pos = centers[point["size"]] + offset
            median_y, p95_y = y(point["median"]), y(point["p95"])
            parts.append(f'<line class="whisker" stroke="{item["color"]}" x1="{x_pos:.1f}" y1="{p95_y:.1f}" x2="{x_pos:.1f}" y2="{median_y:.1f}"/>')
            parts.append(f'<line class="whisker" stroke="{item["color"]}" x1="{x_pos-6:.1f}" y1="{p95_y:.1f}" x2="{x_pos+6:.1f}" y2="{p95_y:.1f}"/>')
            parts.append(f'<circle class="median" fill="{item["color"]}" cx="{x_pos:.1f}" cy="{median_y:.1f}" r="7"><title>{html.escape(item["label"])}: median {point["median"]:g} ns, p95 {point["p95"]:g} ns at {html.escape(" ".join(filter(None, (groups.get(point["size"]), labels.get(point["size"], _size_label(point["size"]))))))}{"" if groups else " elements"}</title></circle>')

    legend_y = height - 79
    total_width = plot_width
    start_x = left
    for index, item in enumerate(series):
        x_pos = start_x + index * (total_width / max(1, len(series)))
        parts.append(f'<line stroke="{item["color"]}" stroke-width="3" x1="{x_pos:.1f}" y1="{legend_y}" x2="{x_pos+20:.1f}" y2="{legend_y}"/>')
        parts.append(f'<circle fill="{item["color"]}" stroke="#fff" stroke-width="1.5" cx="{x_pos+10:.1f}" cy="{legend_y}" r="5"/>')
        parts.append(f'<text x="{x_pos+30:.1f}" y="{legend_y+5}" font-size="13" font-weight="600">{html.escape(item["label"])}</text>')
    parts.append(f'<text class="muted" x="{left}" y="{height-53}" font-size="12">Line + dot: median (p50) · whisker: p95 · missing points: timer unavailable</text>')
    parts.append(f'<text class="muted" x="{left}" y="{height-32}" font-size="12">Log scale; lower is faster. Compare only within this timing domain and the same recorded configuration.</text>')
    parts.append('</svg>')
    return '\n'.join(parts) + '\n'


def render_report(report: dict[str, Any], output_dir: Path, sizes: tuple[int, ...] = SIZES) -> list[Path]:
    if not isinstance(report, dict):
        raise ValueError("benchmark report must be an object")
    measurements = report.get("gpu_measurements")
    if not isinstance(measurements, dict):
        raise ValueError("report must include GPU measurements")
    environment = report.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("report environment must be an object")
    device = measurements.get("device_name", "selected Vulkan device")
    driver = measurements.get("driver_name", "Vulkan driver")
    platform = f'{environment.get("os", "unknown")} {environment.get("architecture", "unknown")}'
    configuration = f'{device} / {driver} · {platform}'
    chart_specs = (
        ("cpu", "CPU arithmetic — host execution", "Single-thread CPU vector-add arithmetic; not GPU work or full application latency."),
        ("host", "Vulkan dispatch — host submit to completion", "Host-observed latency includes KORA/raw Vulkan submission and completion. KORA default includes output readback; deferred KORA and raw Vulkan keep output device-resident."),
        ("gpu", "Vulkan dispatch — device timestamp", "Device timestamp interval where supported; missing KORA sizes are unavailable, not zero. This is not shader active-cycle accounting."),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for kind, title, explanation in chart_specs:
        series = report_series(report, kind, sizes)
        svg = render_svg(f"{title} · {device} / {driver}", f"{title}. {explanation} Reference configuration: {configuration}.", series, sizes)
        filename = {"cpu": "cpu-arithmetic.svg", "host": "host-latency.svg", "gpu": "gpu-timestamps.svg"}[kind]
        output = output_dir / filename
        output.write_text(svg, encoding="utf-8", newline="\n")
        outputs.append(output)
    cases = tuple(case for _, _, case in COMPUTE_CASES)
    compute_series = report_series(report, "compute", cases)
    if any(item["points"] for item in compute_series):
        title = "Matched compute — host submit to completion"
        svg = render_svg(
            f"{title} · {device} / {driver}",
            f"{title}. SAXPY, reduction, scan, stencil, and tiled matmul at three sizes each; KORA default includes "
            f"output readback, deferred KORA and raw Vulkan keep output device-resident. Reference configuration: {configuration}.",
            compute_series,
            cases,
            x_label="Workload · problem size (elements, or matrix dimension²)",
            labels={case: label for _, label, case in COMPUTE_CASES},
            groups={case: group for group, _, case in COMPUTE_CASES},
            width=1400,
        )
        output = output_dir / "compute-host-latency.svg"
        output.write_text(svg, encoding="utf-8", newline="\n")
        outputs.append(output)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="schema-3 kora-bench JSON report")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/charts"))
    args = parser.parse_args()
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        outputs = render_report(report, args.output_dir)
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as error:
        parser.error(str(error))
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
