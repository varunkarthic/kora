"""The README's benchmark numbers equal the medians in the retained JSON reports.

    python3 -m unittest discover -s tests/bench_report

Needs only the standard library (charts are not rendered).
"""

import gzip
import importlib.util
import json
import re
import statistics
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks" / "results" / "rust-vs-cpp-apple-m4"
spec = importlib.util.spec_from_file_location("render", ROOT / "tools" / "bench-report" / "render.py")
render = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = render  # dataclasses look up their module while the class is created.
spec.loader.exec_module(render)


def medians(device: str, side: str) -> dict[str, float]:
    """Median across runs of each run's p50, computed without render.py."""
    runs = [
        {item["name"]: item["median"] for item in json.loads(gzip.decompress(path.read_bytes()))["results"]}
        for path in sorted((RESULTS / device).glob(f"{side}-*.json.gz"))
    ]
    return {name: statistics.median(run[name] for run in runs) for name in runs[0]}


def table_rows(block: str) -> list[list[str]]:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    body = re.search(rf"<!-- bench:{block} -->\n(.*?)\n<!-- /bench:{block} -->", readme, re.S).group(1)
    return [[cell.strip() for cell in line.strip("|").split("|")] for line in body.splitlines()[2:]]


def to_ns(text: str) -> float:
    value, unit = text.split()
    return float(value) * {"ns": 1, "µs": 1e3, "ms": 1e6}[unit]


class ReadmeNumbersMatchJson(unittest.TestCase):
    def assert_close(self, shown: str, expected_ns: float) -> None:
        # The README rounds to its displayed precision; allow half a unit of the last digit.
        value, unit = shown.split()
        step = 10 ** -len(value.split(".")[1]) if "." in value else 1
        scale = {"ns": 1, "µs": 1e3, "ms": 1e6}[unit]
        self.assertLessEqual(abs(to_ns(shown) - expected_ns), step * scale / 2 + 1e-6, f"{shown} vs {expected_ns} ns")

    def test_dispatch_table(self) -> None:
        rust, cpp = medians("moltenvk", "b"), medians("moltenvk", "a")
        rows = table_rows("dispatch")
        self.assertEqual(len(rows), len(render.WORKLOADS))
        for row, (label, size, stem) in zip(rows, render.WORKLOADS):
            self.assertEqual(row[:2], [label, size])
            kora = f"gpu_{stem}_deferred_readback_dispatch_wait"
            self.assert_close(row[2], rust[kora])
            self.assert_close(row[3], cpp[kora])
            self.assert_close(row[5], rust[f"raw_vulkan_{stem}_dispatch_wait"])

    def test_host_table(self) -> None:
        rust, cpp = medians("moltenvk", "b"), medians("moltenvk", "a")
        for row, (label, name) in zip(table_rows("host"), render.HOST_CASES, strict=True):
            self.assertEqual(row[0], label)
            self.assert_close(row[1], rust[name])
            self.assert_close(row[2], cpp[name])

    def test_cluster_table(self) -> None:
        rust = medians("moltenvk", "b")
        for row, (size, stem) in zip(table_rows("cluster"), render.CLUSTER_CASES, strict=True):
            self.assertEqual(row[0], size)
            self.assert_close(row[1], rust[f"cluster_local_{stem}_dispatch_wait"])
            self.assert_close(row[2], rust[f"cluster_remote_loopback_{stem}_dispatch_wait"])

    def test_readme_is_current(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "bench-report" / "render.py"), "--check"],
            capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
