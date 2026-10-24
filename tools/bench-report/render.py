#!/usr/bin/env python3
"""Render the README's benchmark charts and tables from retained kora-bench reports.

Every number comes from the committed JSON in benchmarks/results/rust-vs-cpp-apple-m4: for each
case, the median across runs of each run's p50 (the run's `median` field). Charts are written as
light and dark SVGs under docs/assets/charts; tables replace the regions between
`<!-- bench:NAME -->` and `<!-- /bench:NAME -->` in README.md.

    python3 tools/bench-report/render.py           # regenerate charts and README tables
    python3 tools/bench-report/render.py --check   # fail if the README tables are stale
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks" / "results" / "rust-vs-cpp-apple-m4"
CHARTS = ROOT / "docs" / "assets" / "charts"
README = ROOT / "README.md"

# (label, size label, result-name stem); the stem gains `_deferred_readback_dispatch_wait` for KORA
# and `_dispatch_wait` for raw Vulkan.
WORKLOADS = [
    ("Vector add", "4K", "vector_add_4096"),
    ("Vector add", "1M", "vector_add_1048576"),
    ("Vector add", "4M", "vector_add_4194240"),
    ("SAXPY", "64K", "saxpy_65536_upload"),
    ("SAXPY", "1M", "saxpy_1048576_upload"),
    ("Reduction", "1M", "reduction_1048576"),
    ("Tiled scan", "1M", "tiled_scan_1048576"),
    ("Stencil", "1M", "stencil_1048576"),
    ("Tiled matmul", "128²", "tiled_matmul_128"),
]
HOST_CASES = [
    ("Task graph: finalize 64 tasks", "task_graph_finalize_64"),
    ("Task graph: build, plan, replay 64", "task_graph_replay_64"),
    ("Staging ring: reserve with 512 live ranges", "staging_ring_reserve_fragmented_512"),
    ("Allocator: 256 allocations and frees", "allocator_allocate_free_256"),
    ("CPU stencil, 1M elements", "cpu_stencil_1048576"),
]
CLUSTER_CASES = [
    ("4K", "saxpy_4096"),
    ("64K", "saxpy_65536"),
    ("1M", "saxpy_1048576"),
]
NOISE_BAND = 0.10

THEMES = {
    "light": {
        "fg": "#1f2937",
        "muted": "#6b7280",
        "grid": "#e5e7eb",
        "rust": "#0f766e",
        "cpp": "#9ca3af",
        "raw": "#d97706",
        "faster": "#0f766e",
        "slower": "#d97706",
        "band": "#0f766e1f",
    },
    "dark": {
        "fg": "#e5e7eb",
        "muted": "#9ca3af",
        "grid": "#30363d",
        "rust": "#2dd4bf",
        "cpp": "#6b7280",
        "raw": "#fbbf24",
        "faster": "#2dd4bf",
        "slower": "#fbbf24",
        "band": "#2dd4bf26",
    },
}


@dataclass
class Side:
    """All runs of one build on one device, keyed by result name."""

    runs: list[dict[str, dict]]
    revision: str

    def median(self, name: str) -> float:
        """Median across runs of each run's p50, in nanoseconds."""
        return statistics.median(run[name]["median"] for run in self.runs)

    def quartiles(self, name: str) -> tuple[float, float]:
        values = sorted(run[name]["median"] for run in self.runs)
        q = statistics.quantiles(values, n=4, method="inclusive")
        return q[0], q[2]

    def has(self, name: str) -> bool:
        return all(name in run for run in self.runs)


def load_side(device: str, side: str) -> Side:
    paths = sorted((RESULTS / device).glob(f"{side}-*.json.gz"))
    if not paths:
        raise SystemExit(f"no reports in {RESULTS / device} for side {side!r}")
    reports = [json.loads(gzip.decompress(path.read_bytes())) for path in paths]
    revisions = {report["environment"]["source_revision"] for report in reports}
    if len(revisions) != 1:
        raise SystemExit(f"{device}/{side}: reports come from several revisions: {sorted(revisions)}")
    return Side([{item["name"]: item for item in report["results"]} for report in reports], revisions.pop())


def paired_ratio(before: Side, after: Side, name: str) -> float:
    """Median of per-pair after/before p50 ratios (pairs are matched by file order)."""
    return statistics.median(new[name]["median"] / old[name]["median"] for old, new in zip(before.runs, after.runs))


def family(name: str) -> str:
    return re.sub(r"_\d+(?=_|$)", "", name)


def family_ratios(before: Side, after: Side) -> list[tuple[str, float, int, int]]:
    """(family, median ratio, wins, comparisons) for every family present on both sides."""
    names = set.intersection(*(set(run) for run in before.runs + after.runs))
    grouped: dict[str, list[str]] = {}
    for name in sorted(names):
        if not name.startswith("profile_"):
            grouped.setdefault(family(name), []).append(name)
    rows = []
    for key, members in grouped.items():
        ratios = [
            new[n]["median"] / old[n]["median"]
            for n in members
            for old, new in zip(before.runs, after.runs)
            if old[n]["median"] > 0
        ]
        rows.append((key, statistics.median(ratios), sum(r < 1 for r in ratios), len(ratios)))
    return rows


def fmt_ns(ns: float) -> str:
    if ns >= 1_000_000:
        return f"{ns / 1_000_000:.2f} ms"
    if ns >= 10_000:
        return f"{ns / 1_000:.0f} µs"
    if ns >= 1_000:
        return f"{ns / 1_000:.1f} µs"
    return f"{ns:.0f} ns"


def fmt_change(ratio: float) -> str:
    return f"{(ratio - 1) * 100:+.0f}%"


def dispatch_table(cpp: Side, rust: Side) -> str:
    lines = [
        "| Workload | Size | KORA 1.0 | KORA 0.3 | 1.0 vs 0.3 | Raw Vulkan |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, size, stem in WORKLOADS:
        kora = f"gpu_{stem}_deferred_readback_dispatch_wait"
        raw = f"raw_vulkan_{stem}_dispatch_wait"
        lines.append(
            f"| {label} | {size} | {fmt_ns(rust.median(kora))} | {fmt_ns(cpp.median(kora))} | "
            f"{fmt_change(paired_ratio(cpp, rust, kora))} | {fmt_ns(rust.median(raw))} |"
        )
    return "\n".join(lines)


def host_table(cpp: Side, rust: Side) -> str:
    lines = ["| Operation | KORA 1.0 | KORA 0.3 | Change |", "| --- | ---: | ---: | ---: |"]
    for label, name in HOST_CASES:
        lines.append(
            f"| {label} | {fmt_ns(rust.median(name))} | {fmt_ns(cpp.median(name))} | "
            f"{fmt_change(paired_ratio(cpp, rust, name))} |"
        )
    return "\n".join(lines)


def cluster_table(rust: Side) -> str:
    lines = ["| SAXPY | Local | Remote worker (loopback) |", "| ---: | ---: | ---: |"]
    for size, stem in CLUSTER_CASES:
        lines.append(
            f"| {size} | {fmt_ns(rust.median(f'cluster_local_{stem}_dispatch_wait'))} | "
            f"{fmt_ns(rust.median(f'cluster_remote_loopback_{stem}_dispatch_wait'))} |"
        )
    return "\n".join(lines)


def family_summary(rows: list[tuple[str, float, int, int]], device: str) -> str:
    faster = sum(r < 1 - NOISE_BAND for _, r, _, _ in rows)
    slower = [f for f, r, _, _ in rows if r > 1 + NOISE_BAND]
    within = len(rows) - faster - len(slower)
    return (
        f"{device}: {len(rows)} benchmark families — {faster} faster than 0.3 by more than "
        f"{NOISE_BAND:.0%}, {within} within ±{NOISE_BAND:.0%}, {len(slower)} slower by more than "
        f"{NOISE_BAND:.0%}" + (f" ({', '.join(f'`{s}`' for s in slower)})" if slower else "") + "."
    )


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "svg.fonttype": "none",
            "svg.hashsalt": "kora",
            "font.family": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 10,
        }
    )
    return plt


def save(fig, stem: str, theme: str) -> None:
    CHARTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARTS / f"{stem}-{theme}.svg", transparent=True, bbox_inches="tight", metadata={"Date": None})


def style_axes(ax, colors) -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(colors["grid"])
    ax.tick_params(colors=colors["muted"])
    ax.xaxis.label.set_color(colors["muted"])
    ax.yaxis.label.set_color(colors["muted"])
    ax.title.set_color(colors["fg"])


def dispatch_chart(plt, cpp: Side, rust: Side, device: str, theme: str) -> None:
    import numpy as np

    colors = THEMES[theme]
    labels = [f"{label}\n{size}" for label, size, _ in WORKLOADS]
    series = [
        ("KORA 1.0 (Rust)", rust, "gpu_{}_deferred_readback_dispatch_wait", colors["rust"]),
        ("KORA 0.3 (C++)", cpp, "gpu_{}_deferred_readback_dispatch_wait", colors["cpp"]),
        ("Raw Vulkan", rust, "raw_vulkan_{}_dispatch_wait", colors["raw"]),
    ]
    x = np.arange(len(WORKLOADS))
    width = 0.26
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for index, (name, side, pattern, color) in enumerate(series):
        medians = np.array([side.median(pattern.format(stem)) for _, _, stem in WORKLOADS]) / 1000
        quartiles = np.array([side.quartiles(pattern.format(stem)) for _, _, stem in WORKLOADS]) / 1000
        errors = np.vstack([medians - quartiles[:, 0], quartiles[:, 1] - medians])
        ax.bar(x + (index - 1) * width, medians, width, label=name, color=color, zorder=3)
        ax.errorbar(
            x + (index - 1) * width, medians, yerr=errors, fmt="none", ecolor=colors["fg"], elinewidth=0.8,
            capsize=2, alpha=0.55, zorder=4,
        )
    from matplotlib.ticker import FuncFormatter

    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f} µs"))
    ax.set_ylabel("Host submit → completion")
    ax.set_xticks(x, labels)
    ax.grid(axis="y", color=colors["grid"], linewidth=0.6, zorder=0)
    ax.set_title(
        f"Dispatch latency on Apple M4 / {device} — median of 10 interleaved runs (bars: interquartile range)",
        loc="left", fontsize=10.5, fontweight="bold", color=colors["fg"],
    )
    style_axes(ax, colors)
    legend = ax.legend(frameon=False, ncols=3, loc="upper left", bbox_to_anchor=(0, -0.2))
    for text in legend.get_texts():
        text.set_color(colors["fg"])
    save(fig, f"readme-dispatch-{device.lower()}", theme)
    plt.close(fig)


def family_chart(plt, rows: list[tuple[str, float, int, int]], device: str, theme: str) -> None:
    import numpy as np

    colors = THEMES[theme]
    rows = sorted((r for r in rows if r[0] != "host_allocation_4k"), key=lambda r: r[1])
    names = [r[0] for r in rows]
    ratios = np.array([r[1] for r in rows])
    fig, ax = plt.subplots(figsize=(10, max(4, 0.16 * len(rows))))
    ax.axvspan(1 - NOISE_BAND, 1 + NOISE_BAND, color=colors["band"], zorder=0, label=f"±{NOISE_BAND:.0%} noise band")
    bar_colors = [
        colors["faster"] if r < 1 - NOISE_BAND else colors["slower"] if r > 1 + NOISE_BAND else colors["cpp"]
        for r in ratios
    ]
    y = np.arange(len(rows))
    ax.barh(y, ratios - 1, left=1, color=bar_colors, height=0.7, zorder=3)
    ax.axvline(1, color=colors["muted"], linewidth=0.8, zorder=2)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 2)
    from matplotlib.ticker import NullFormatter

    ax.set_xticks([0.125, 0.25, 0.5, 1, 1.5, 2], ["0.125×", "0.25×", "0.5×", "1×", "1.5×", "2×"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(y, names, fontsize=7.5, family="monospace")
    ax.set_xlabel("KORA 1.0 / KORA 0.3, median per-pair ratio (left of 1× = 1.0 is faster)")
    ax.set_title(
        f"Every benchmark family, 1.0 vs 0.3 — Apple M4 / {device}, 10 interleaved pairs (shaded: ±10% noise)",
        loc="left", fontsize=10.5, fontweight="bold", color=colors["fg"],
    )
    style_axes(ax, colors)
    ax.tick_params(axis="y", colors=colors["fg"])
    save(fig, f"readme-families-{device.lower()}", theme)
    plt.close(fig)


def replace_block(text: str, name: str, body: str) -> str:
    pattern = re.compile(rf"(<!-- bench:{name} -->\n)(?:.*?\n)?(<!-- /bench:{name} -->)", re.S)
    if not pattern.search(text):
        raise SystemExit(f"README.md has no <!-- bench:{name} --> block")
    return pattern.sub(lambda m: m.group(1) + body + "\n" + m.group(2), text)


def tables() -> dict[str, str]:
    cpp, rust = load_side("moltenvk", "a"), load_side("moltenvk", "b")
    kk_cpp, kk_rust = load_side("kosmickrisp", "a"), load_side("kosmickrisp", "b")
    return {
        "dispatch": dispatch_table(cpp, rust),
        "host": host_table(cpp, rust),
        "cluster": cluster_table(rust),
        "families": "\n\n".join(
            [
                family_summary(family_ratios(cpp, rust), "MoltenVK"),
                family_summary(family_ratios(kk_cpp, kk_rust), "KosmicKrisp"),
            ]
        ),
        "revisions": f"C++ 0.3 `{cpp.revision[:7]}` and Rust 1.0 `{rust.revision[:7]}`",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="fail if README.md tables are out of date")
    parser.add_argument("--no-charts", action="store_true", help="update tables only")
    args = parser.parse_args()
    readme = README.read_text(encoding="utf-8")
    updated = readme
    for name, body in tables().items():
        updated = replace_block(updated, name, body)
    if args.check:
        if updated != readme:
            print("README.md benchmark tables are stale; run tools/bench-report/render.py", file=sys.stderr)
            return 1
        return 0
    README.write_text(updated, encoding="utf-8")
    if not args.no_charts:
        plt = setup_matplotlib()
        for device in ("moltenvk", "kosmickrisp"):
            cpp, rust = load_side(device, "a"), load_side(device, "b")
            label = "MoltenVK" if device == "moltenvk" else "KosmicKrisp"
            for theme in THEMES:
                dispatch_chart(plt, cpp, rust, label, theme)
                family_chart(plt, family_ratios(cpp, rust), label, theme)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
