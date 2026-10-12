import importlib.util
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree


ROOT = Path(__file__).parents[2]
SPEC = importlib.util.spec_from_file_location(
    "render_charts", ROOT / "tools/kora-bench/render_charts.py"
)
charts = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(charts)


def result(name, domain, median=1000, p95=1500):
    return {
        "name": name,
        "timing_domain": domain,
        "unit": "ns",
        "median": median,
        "p95": p95,
        "samples": [median, p95],
    }


def report(results=None):
    return {
        "schema_version": 3,
        "environment": {"os": "macos", "architecture": "arm64"},
        "gpu_measurements": {
            "same_physical_device_for_kora_and_raw_vulkan": True,
            "device_name": "Apple M4",
            "driver_name": "MoltenVK",
        },
        "results": results or [
            result("cpu_vector_add_4096", "host_cpu_execution"),
            result("gpu_vector_add_4096_dispatch_wait", "host_submit_completion"),
            result("gpu_vector_add_4096_deferred_readback_dispatch_wait", "host_submit_completion"),
            result("raw_vulkan_vector_add_4096_dispatch_wait", "host_submit_completion"),
            result("gpu_vector_add_4096_gpu_execution", "gpu_execution_timestamp"),
            result("raw_vulkan_vector_add_4096_gpu_execution", "gpu_execution_timestamp"),
        ],
    }


class BenchmarkChartTests(unittest.TestCase):
    def test_unavailable_timestamp_is_a_gap_not_a_zero(self):
        series = charts.report_series(report(), "gpu", (4096, 1048576))
        self.assertEqual(series[0]["points"], [{"size": 4096, "median": 1000, "p95": 1500}])
        self.assertEqual(series[1]["points"], [{"size": 4096, "median": 1000, "p95": 1500}])
        svg = charts.render_svg("GPU <timer>", "Unavailable sizes are gaps.", series, (4096, 1048576))
        self.assertIn("GPU &lt;timer&gt;", svg)
        self.assertNotIn("at 1,048,576 elements", svg)
        self.assertIn("role=\"img\"", svg)
        self.assertIn("median 1000 ns, p95 1500 ns", svg)
        ElementTree.fromstring(svg)

    def test_domain_mismatch_fails_instead_of_mixing_timings(self):
        bad = report([result("gpu_vector_add_4096_dispatch_wait", "gpu_execution_timestamp")])
        with self.assertRaisesRegex(ValueError, "must use host_submit_completion"):
            charts.report_series(bad, "host", (4096,))

    def test_nonfinite_or_inconsistent_summary_is_rejected(self):
        bad = report([result("cpu_vector_add_4096", "host_cpu_execution", float("nan"), 1500)])
        with self.assertRaisesRegex(ValueError, "finite positive"):
            charts.report_series(bad, "cpu", (4096,))
        bad = report([result("cpu_vector_add_4096", "host_cpu_execution", 2000, 1500)])
        with self.assertRaisesRegex(ValueError, "p95 must not be below"):
            charts.report_series(bad, "cpu", (4096,))

    def test_shared_device_and_schema_are_required(self):
        bad = report()
        bad["gpu_measurements"]["same_physical_device_for_kora_and_raw_vulkan"] = False
        with self.assertRaisesRegex(ValueError, "shared physical device"):
            charts.report_series(bad, "host", (4096,))
        bad = report()
        bad["schema_version"] = 2
        with self.assertRaisesRegex(ValueError, "schema version 3"):
            charts.report_series(bad, "host", (4096,))

    def test_generation_is_deterministic_and_separates_domains(self):
        data = report()
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            paths_a = charts.render_report(data, Path(first), (4096, 1048576))
            paths_b = charts.render_report(data, Path(second), (4096, 1048576))
            self.assertEqual([path.name for path in paths_a], [path.name for path in paths_b])
            for path_a, path_b in zip(paths_a, paths_b):
                self.assertEqual(path_a.read_bytes(), path_b.read_bytes())
            self.assertEqual({path.name for path in paths_a}, {"cpu-arithmetic.svg", "host-latency.svg", "gpu-timestamps.svg"})
            self.assertIn("Apple M4 / MoltenVK", (Path(first) / "gpu-timestamps.svg").read_text())


if __name__ == "__main__":
    unittest.main()
