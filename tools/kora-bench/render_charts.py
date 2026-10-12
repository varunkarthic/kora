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
}


def _valid_ns(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def report_series(report: dict[str, Any], kind: str, sizes: tuple[int, ...] = SIZES) -> list[dict[str, Any]]:
    """Validate report identity/schema and return only available, correctly-domain-tagged values."""
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


def _size_label(size: int) -> str:
    return f"{size:,}"


def render_svg(title: str, description: str, series: list[dict[str, Any]], sizes: tuple[int, ...]) -> str:
    points = [point for item in series for point in item["points"]]
    if not points:
        raise ValueError("cannot render a chart without measurements")

    width, height = 920, 470
    left, right, top, bottom = 100, 32, 70, 105
    plot_width, plot_height = width - left - right, height - top - bottom
    minimum = min(point["median"] for point in points)
    maximum = max(point["p95"] for point in points)
    min_exp = math.floor(math.log10(minimum))
    max_exp = math.ceil(math.log10(maximum))
    if min_exp == max_exp:
        min_exp -= 1
        max_exp += 1
    lower, upper = 10.0**min_exp, 10.0**max_exp

    def y(value: float) -> float:
        fraction = (math.log10(value) - math.log10(lower)) / (math.log10(upper) - math.log10(lower))
        return top + plot_height * (1.0 - fraction)

    escaped_title = html.escape(title)
    escaped_description = html.escape(description)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{escaped_title}</title><desc id="desc">{escaped_description}</desc>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1f2937}.muted{fill:#4b5563}.grid{stroke:#d1d5db;stroke-width:1}.axis{stroke:#374151;stroke-width:1.4}.whisker{stroke-width:2}.median{stroke:#fff;stroke-width:1.5}</style>',
        f'<text x="{left}" y="30" font-size="20" font-weight="700">{escaped_title}</text>',
    ]
    for exponent in range(min_exp, max_exp + 1):
        value = 10.0**exponent
        y_pos = y(value)
        parts.append(f'<line class="grid" x1="{left}" y1="{y_pos:.1f}" x2="{width-right}" y2="{y_pos:.1f}"/>')
        parts.append(f'<text class="muted" x="{left-12}" y="{y_pos+5:.1f}" text-anchor="end" font-size="12">{value:g}</text>')
    parts.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}"/>',
            f'<text class="muted" x="24" y="{top + plot_height/2:.1f}" transform="rotate(-90 24 {top + plot_height/2:.1f})" text-anchor="middle" font-size="13">Latency (ns, log scale)</text>',
            f'<text class="muted" x="{left + plot_width/2:.1f}" y="{height-18}" text-anchor="middle" font-size="13">Vector elements</text>',
        ]
    )

    centers = {size: left + plot_width * (index + 0.5) / len(sizes) for index, size in enumerate(sizes)}
    group_width = plot_width / max(1, len(sizes))
    for size in sizes:
        parts.append(f'<text class="muted" x="{centers[size]:.1f}" y="{height-bottom+24}" text-anchor="middle" font-size="12">{_size_label(size)}</text>')
    for series_index, item in enumerate(series):
        count = len(series)
        offset = (series_index - (count - 1) / 2) * min(20, group_width / max(2, count + 1))
        for point in item["points"]:
            x_pos = centers[point["size"]] + offset
            median_y, p95_y = y(point["median"]), y(point["p95"])
            parts.append(f'<line class="whisker" stroke="{item["color"]}" x1="{x_pos:.1f}" y1="{p95_y:.1f}" x2="{x_pos:.1f}" y2="{median_y:.1f}"/>')
            parts.append(f'<line class="whisker" stroke="{item["color"]}" x1="{x_pos-5:.1f}" y1="{p95_y:.1f}" x2="{x_pos+5:.1f}" y2="{p95_y:.1f}"/>')
            parts.append(f'<circle class="median" fill="{item["color"]}" cx="{x_pos:.1f}" cy="{median_y:.1f}" r="6"><title>{html.escape(item["label"])}: median {point["median"]:g} ns, p95 {point["p95"]:g} ns at {_size_label(point["size"])} elements</title></circle>')

    legend_y = height - 55
    total_width = min(plot_width, 210 * len(series))
    start_x = left + max(0, (plot_width - total_width) / 2)
    for index, item in enumerate(series):
        x_pos = start_x + index * (total_width / max(1, len(series)))
        parts.append(f'<circle fill="{item["color"]}" cx="{x_pos:.1f}" cy="{legend_y}" r="5"/>')
        parts.append(f'<text x="{x_pos+12:.1f}" y="{legend_y+4}" font-size="12">{html.escape(item["label"])}</text>')
    parts.append(f'<text class="muted" x="{left}" y="{height-20}" font-size="11">Dots: median (p50); whiskers extend to p95. Logarithmic vertical axis.</text>')
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
