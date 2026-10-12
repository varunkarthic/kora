import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).parents[2]
SPEC = importlib.util.spec_from_file_location(
    "render_comparison", ROOT / "tools/kora-bench/render_comparison.py"
)
comparison = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(comparison)


def kora_report():
    return {
        "schema_version": 3,
        "diagnostic_profile": False,
        "gpu_measurements": {"same_physical_device_for_kora_and_raw_vulkan": True},
        "results": [
            {"name": "cpu_vector_add_4096", "timing_domain": "host_cpu_execution", "median": 90, "p95": 120},
            {"name": "gpu_vector_add_4096_dispatch_wait", "timing_domain": "host_submit_completion", "median": 400, "p95": 500},
            {
                "name": "gpu_vector_add_4096_deferred_readback_dispatch_wait",
                "timing_domain": "host_submit_completion",
                "median": 300,
                "p95": 350,
            },
            {
                "name": "raw_vulkan_vector_add_4096_dispatch_wait",
                "timing_domain": "host_submit_completion",
                "median": 250,
                "p95": 280,
            },
            {
                "name": "gpu_vector_add_4096_gpu_execution",
                "timing_domain": "gpu_execution_timestamp",
                "median": 210,
                "p95": 250,
            },
            {
                "name": "raw_vulkan_vector_add_4096_gpu_execution",
                "timing_domain": "gpu_execution_timestamp",
                "median": 190,
                "p95": 230,
            },
        ],
    }


def cuda_report():
    return {
        "schema_version": 1,
        "backend": "cuda",
        "environment": {
            "os": "linux",
            "architecture": "x86_64",
            "compiler": "nvcc 13.0",
            "build_type": "Release",
            "source_revision": "a" * 40,
        },
        "device": {"name": "NVIDIA Example", "driver_version": "600.1", "runtime_version": "13.0"},
        "device_match": {
            "same_physical_device_as_kora_vulkan": True,
            "verification_method": "cuda_uuid_equals_vk_device_uuid",
        },
        "results": [
            {
                "workload_id": "vector_add",
                "size": 4096,
                "precision": "fp32",
                "semantic_contract": "kora-bench.vector-add.f32.v1",
                "comparison_class": "algorithm_matched",
                "implementation": "one-thread-per-element",
                "measurement_boundary": "kernel_dispatch",
                "kora_result_name": "gpu_vector_add_4096_gpu_execution",
                "raw_vulkan_result_name": "raw_vulkan_vector_add_4096_gpu_execution",
                "correctness_method": "elementwise exact comparison before timed iterations",
                "correctness_checked": True,
                "correctness_passed": True,
                "timing_domain": "gpu_execution_event",
                "unit": "ns",
                "iterations": 4,
                "warmup": 2,
                "samples": [100, 200, 300, 400],
                "median": 200,
                "p95": 400,
            }
        ],
    }


class CudaBenchmarkContractTests(unittest.TestCase):
    def test_accepts_versioned_same_device_report_and_preserves_domain(self):
        rows = comparison.validate_cuda_report(kora_report(), cuda_report())
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["timing_domain"], "gpu_execution_event")
        self.assertEqual(rows[0]["kora_median"], 210)
        self.assertEqual(rows[0]["raw_vulkan_median"], 190)

    def test_unavailable_kora_timer_remains_unavailable_not_zero(self):
        report = kora_report()
        report["results"].clear()
        rows = comparison.validate_cuda_report(report, cuda_report())
        self.assertIsNone(rows[0]["kora_median"])
        self.assertIsNone(rows[0]["raw_vulkan_median"])

    def test_rejects_unverified_or_mismatched_device(self):
        bad = cuda_report()
        bad["device_match"]["same_physical_device_as_kora_vulkan"] = False
        with self.assertRaisesRegex(ValueError, "same-physical-device"):
            comparison.validate_cuda_report(kora_report(), bad)
        bad = cuda_report()
        bad["device_match"]["verification_method"] = "device-name-only"
        with self.assertRaisesRegex(ValueError, "UUIDs"):
            comparison.validate_cuda_report(kora_report(), bad)

    def test_rejects_domain_or_boundary_mismatch(self):
        bad = cuda_report()
        bad["results"][0]["timing_domain"] = "host_submit_completion"
        with self.assertRaisesRegex(ValueError, "GPU event timing"):
            comparison.validate_cuda_report(kora_report(), bad)
        bad = cuda_report()
        bad["results"][0]["measurement_boundary"] = "host_submit_and_wait"
        with self.assertRaisesRegex(ValueError, "GPU event timing"):
            comparison.validate_cuda_report(kora_report(), bad)

    def test_rejects_unchecked_results_and_inconsistent_raw_samples(self):
        bad = cuda_report()
        bad["results"][0]["correctness_passed"] = False
        with self.assertRaisesRegex(ValueError, "correctness check"):
            comparison.validate_cuda_report(kora_report(), bad)
        bad = cuda_report()
        bad["results"][0]["p95"] = 399
        with self.assertRaisesRegex(ValueError, "nearest-rank"):
            comparison.validate_cuda_report(kora_report(), bad)

    def test_rejects_diagnostic_kora_report_and_invalid_revision(self):
        bad_kora = kora_report()
        bad_kora["diagnostic_profile"] = True
        with self.assertRaisesRegex(ValueError, "telemetry overhead"):
            comparison.validate_cuda_report(bad_kora, cuda_report())
        bad_cuda = deepcopy(cuda_report())
        bad_cuda["environment"]["source_revision"] = "not-a-commit"
        with self.assertRaisesRegex(ValueError, "Git commit hash"):
            comparison.validate_cuda_report(kora_report(), bad_cuda)

    def test_cli_renders_optional_cuda_table_after_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kora_path = root / "kora.json"
            cuda_path = root / "cuda.json"
            kora_path.write_text(json.dumps(kora_report()), encoding="utf-8")
            cuda_path.write_text(json.dumps(cuda_report()), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools/kora-bench/render_comparison.py"),
                    str(kora_path),
                    "--sizes",
                    "4096",
                    "--cuda-report",
                    str(cuda_path),
                    "--kora-revision",
                    "b" * 40,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Optional same-device CUDA reference", result.stdout)
        self.assertIn("KORA source revision: " + "b" * 40, result.stdout)
        self.assertIn("200 ns", result.stdout)
        self.assertIn("0.952×", result.stdout)

    def test_cli_requires_kora_revision_for_cuda_comparisons(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kora_path = root / "kora.json"
            cuda_path = root / "cuda.json"
            kora_path.write_text(json.dumps(kora_report()), encoding="utf-8")
            cuda_path.write_text(json.dumps(cuda_report()), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools/kora-bench/render_comparison.py"),
                    str(kora_path),
                    "--sizes",
                    "4096",
                    "--cuda-report",
                    str(cuda_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires --kora-revision", result.stderr)


if __name__ == "__main__":
    unittest.main()
