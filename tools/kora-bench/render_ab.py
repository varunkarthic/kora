#!/usr/bin/env python3
"""Summarize interleaved before/after kora-bench runs as a Markdown table and SVG charts.

Runs are paired by position (before[i] ran next to after[i]). Each case reports the median across
runs of the per-run p50 (and p95) for the charts, plus the median of the per-pair after/before p50
ratios and how many pairs the after build won. Pairing adjacent runs keeps host-load and GPU clock
drift, which can move a single run's median several-fold on some devices, from favoring either side.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import re
import statistics
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def _load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


charts = _load_module("render_charts")
comparison = _load_module("render_comparison")

VECTOR_SIZES = (4096, 65536, 1048576, 4194240)


def cases() -> list[tuple[str, str, str, str, str]]:
    """Return (group, tick label, KORA default, KORA deferred, raw Vulkan) result names."""
    labels = {case: (group, label) for group, label, case in charts.COMPUTE_CASES}
    rows = []
    for _, default, deferred, raw, _, _ in comparison.matched_compute_rows():
        case = raw.removeprefix("raw_vulkan_").removesuffix("_dispatch_wait")
        rows.append((*labels[case], default, deferred, raw))
    for size, label in zip(VECTOR_SIZES, ("4K", "64K", "1M", "4M")):
        rows.append(
            (
                "Vector add",
                label,
                f"gpu_vector_add_{size}_dispatch_wait",
                f"gpu_vector_add_{size}_deferred_readback_dispatch_wait",
                f"raw_vulkan_vector_add_{size}_dispatch_wait",
            )
        )
    return rows


def read_report(path: Path) -> dict[str, Any]:
    """Reads a JSON report, transparently decompressing retained `.json.gz` files."""
    data = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    return json.loads(data.decode("utf-8"))


def load_runs(paths: list[Path]) -> tuple[list[dict[str, dict[str, Any]]], str]:
    runs, revisions = [], set()
    for path in paths:
        report = read_report(path)
        if report.get("schema_version") not in (3, 4) or report.get("diagnostic_profile") is True:
            raise ValueError(f"{path} is not a schema-3 or schema-4 comparison report")
        if not report.get("gpu_measurements", {}).get("same_physical_device_for_kora_and_raw_vulkan"):
            raise ValueError(f"{path} must confirm a shared physical device")
        revisions.add(report["environment"]["source_revision"])
        runs.append({item["name"]: item for item in report["results"]})
    if not runs:
        raise ValueError("at least one report is required per side")
    if len(revisions) != 1:
        raise ValueError("all reports on one side must come from the same source revision")
    return runs, revisions.pop()


def summarize(runs: list[dict[str, dict[str, Any]]], name: str) -> dict[str, float]:
    missing = [index for index, run in enumerate(runs) if name not in run]
    if missing:
        raise ValueError(f"{name} is missing from run(s) {missing}")
    return {
        stat: statistics.median(run[name][stat] for run in runs) for stat in ("median", "p95")
    }


def paired_change(before: list[dict[str, dict[str, Any]]], after: list[dict[str, dict[str, Any]]], name: str) -> str:
    ratios = [new[name]["median"] / old[name]["median"] for old, new in zip(before, after)]
    wins = sum(ratio < 1 for ratio in ratios)
    return f"{100 * (statistics.median(ratios) - 1):+.0f}% ({wins}/{len(ratios)})"


def family(name: str) -> str:
    """A result name without its size tokens, e.g. gpu_vector_add_dispatch_wait."""
    return re.sub(r"_\d+(?=_|$)", "", name)


def render_families(before_paths: list[Path], after_paths: list[Path]) -> tuple[str, list[dict[str, Any]]]:
    """Per-family medians of every per-pair after/before p50 ratio across the family's cases.

    Returns the Markdown table and the rows (family, cases, pairs, ratio, wins, comparisons).
    """
    before, before_revision = load_runs(before_paths)
    after, after_revision = load_runs(after_paths)
    if len(before) != len(after):
        raise ValueError("before and after runs must be paired one-to-one")
    names = set.intersection(*(set(run) for run in before + after))
    grouped: dict[str, list[str]] = {}
    for name in sorted(names):
        if name.startswith("profile_"):
            continue
        grouped.setdefault(family(name), []).append(name)
    rows = []
    for key, members in sorted(grouped.items()):
        ratios = [
            new[name]["median"] / old[name]["median"]
            for name in members
            for old, new in zip(before, after)
            if old[name]["median"] > 0
        ]
        if not ratios:
            continue
        rows.append(
            {
                "family": key,
                "cases": len(members),
                "pairs": len(before),
                "ratio": statistics.median(ratios),
                "wins": sum(ratio < 1 for ratio in ratios),
                "comparisons": len(ratios),
            }
        )
    lines = [
        f"Before: `{before_revision}` ({len(before)} runs); after: `{after_revision}` ({len(after)} runs). "
        "Ratio is the median of every per-pair after/before p50 ratio across the family's sizes "
        "(below 1.00 means the after build is faster); wins counts the comparisons the after build won.",
        "",
        "| Family | Sizes | Ratio | Wins |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['family']}` | {row['cases']} | {row['ratio']:.2f} | {row['wins']}/{row['comparisons']} |"
        )
    return "\n".join(lines) + "\n", rows


def render(
    before_paths: list[Path],
    after_paths: list[Path],
    output_dir: Path,
    title_suffix: str,
    change: str = "#227",
    prefix: str = "uma-direct",
) -> str:
    before, before_revision = load_runs(before_paths)
    after, after_revision = load_runs(after_paths)
    if len(before) != len(after):
        raise ValueError("before and after runs must be paired one-to-one")
    rows = cases()
    lines = [
        f"Before: `{before_revision}` ({len(before)} runs); after: `{after_revision}` ({len(after)} runs). "
        "Latencies are the median across runs of the per-run p50. Change is the median of per-pair "
        "after/before p50 ratios, with the number of pairs the after build won.",
        "",
        "| Workload | Size | KORA default before | after | change | KORA deferred before | after | change | "
        "Raw Vulkan (after runs) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    series = {"default": ([], [], []), "deferred": ([], [], [])}
    keys = []
    for group, label, default, deferred, raw in rows:
        key = f"{group}:{label}"
        keys.append((key, group, label))
        cells = [group, label]
        for kind, name in (("default", default), ("deferred", deferred)):
            old, new = summarize(before, name), summarize(after, name)
            cells += [
                comparison.format_ns(int(old["median"])),
                comparison.format_ns(int(new["median"])),
                paired_change(before, after, name),
            ]
            series[kind][0].append({"size": key, **old})
            series[kind][1].append({"size": key, **new})
        raw_after = summarize(after, raw)
        for kind in series:
            series[kind][2].append({"size": key, **raw_after})
        cells.append(comparison.format_ns(int(raw_after["median"])))
        lines.append("| " + " | ".join(cells) + " |")

    output_dir.mkdir(parents=True, exist_ok=True)
    sizes = tuple(key for key, _, _ in keys)
    labels = {key: label for key, _, label in keys}
    groups = {key: group for key, group, _ in keys}
    for kind, heading in (("default", "KORA default readback"), ("deferred", "KORA deferred readback")):
        old, new, raw = series[kind]
        chart_series = [
            {"label": f"{heading} · before", "color": "#c2410c", "points": old},
            {"label": f"{heading} · after", "color": "#2563eb", "points": new},
            {"label": "Raw Vulkan", "color": "#15803d", "points": raw},
        ]
        title = f"{heading} before and after {change} — host submit to completion · {title_suffix}"
        svg = charts.render_svg(
            title,
            f"{title}. Median across interleaved runs of each run's p50 (dot) and p95 (whisker).",
            chart_series,
            sizes,
            x_label="Workload · problem size (elements, or matrix dimension²)",
            labels=labels,
            groups=groups,
            width=1520,
        )
        (output_dir / f"{prefix}-{kind}.svg").write_text(svg, encoding="utf-8", newline="\n")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, nargs="+", required=True)
    parser.add_argument("--after", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/charts"))
    parser.add_argument("--title-suffix", default="Apple M4 / MoltenVK")
    parser.add_argument("--change", default="#227", help="change name used in chart titles")
    parser.add_argument("--prefix", default="uma-direct", help="chart file name prefix")
    parser.add_argument(
        "--families", action="store_true", help="print the per-family summary instead of the compute table"
    )
    args = parser.parse_args()
    try:
        if args.families:
            print(render_families(args.before, args.after)[0], end="")
            return 0
        print(render(args.before, args.after, args.output_dir, args.title_suffix, args.change, args.prefix), end="")
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
