#!/usr/bin/env python3
"""Validate tiny benchmark reports and their percentile/sample invariants."""

import json
from pathlib import Path
import re
import subprocess
import sys


arguments = [sys.argv[1], *sys.argv[2:]]
if "--smoke" not in arguments:
    arguments.append("--smoke")
completed = subprocess.run(arguments, text=True, capture_output=True)
if completed.returncode == 3:
    print("GPU benchmark skipped: " + completed.stderr.strip())
    sys.exit(77)
completed.check_returncode()
report = json.loads(completed.stdout)
assert report["schema_version"] == 3
benchmarking_doc = Path(__file__).resolve().parents[1] / "docs" / "benchmarking.md"
documentation = benchmarking_doc.read_text(encoding="utf-8")
assert re.search(r"valid JSON with schema version `3`", documentation), (
    "docs/benchmarking.md must state the schema version emitted by kora-bench"
)
assert report["clock"] == "steady_clock"
assert report["iterations"] == 3
assert report["warmup"] == 1
assert report["environment"]["os"] in {"windows", "macos", "linux", "unknown"}
assert report["environment"]["architecture"] in {"arm64", "x86_64", "x86", "unknown"}
assert report["environment"]["compiler"].startswith(("MSVC ", "Clang ", "AppleClang ", "GCC ", "unknown"))
assert report["environment"]["build_type"]
assert isinstance(report["environment"]["cpu_model"], str) and report["environment"]["cpu_model"]
assert isinstance(report["environment"]["build_flags"], str)
assert re.fullmatch(r"[0-9a-f]{40}|unknown", report["environment"]["source_revision"])
assert isinstance(report["environment"]["logical_cpu_count"], int)
expected_names = {
    "scheduler_cost",
    "task_graph_finalize_64",
    "task_graph_replay_64",
    "task_graph_replay_64_reuse_plan",
    "allocator_allocate_free_256",
    "allocator_fragmentation_128",
    "host_allocation_4k",
}
vector_sizes = (1, 64, 256, 1024, 4096, 65536, 262144, 1048576, 4194240)
expected_names |= {f"cpu_vector_add_{count}" for count in vector_sizes}
compute_sizes = (4096, 65536, 1048576)
expected_names |= {
    f"cpu_{workload}_{count}"
    for workload in ("saxpy", "reduction", "stencil")
    for count in compute_sizes
}
expected_names |= {f"cpu_tiled_matmul_{dimension}" for dimension in (16, 64, 128)}
expected_names |= {
    f"cpu_{workload}_parallel_{count}"
    for workload in ("vector_add", "saxpy", "reduction", "stencil")
    for count in compute_sizes
}
expected_names |= {f"cpu_tiled_matmul_parallel_{dimension}" for dimension in (16, 64, 128)}
if "--gpu" in arguments:
    gpu = report["gpu_measurements"]
    assert gpu["backend"] == "vulkan"
    assert isinstance(gpu["device_index"], int) and gpu["device_index"] >= 0
    assert gpu["device_name"] and gpu["api_version"] and gpu["driver_name"]
    assert gpu["device_kind"] in {"discrete_gpu", "integrated_gpu", "virtual_gpu", "cpu_vulkan", "other"}
    assert isinstance(gpu["compute_queue_family"], int)
    assert gpu["same_physical_device_for_kora_and_raw_vulkan"] is True
    assert isinstance(gpu["kora_gpu_timestamps_available"], bool)
    assert isinstance(gpu["raw_gpu_timestamps_available"], bool)
    assert gpu["kora_gpu_timestamps_available"] == bool(gpu["kora_gpu_timestamp_sizes"])
    assert gpu["raw_gpu_timestamps_available"] == bool(gpu["raw_gpu_timestamp_sizes"])
    assert set(gpu["paths"]) == {"kora_runtime", "kora_transfer_engine", "raw_vulkan"}
    assert not {"uuid", "device_uuid", "vendor_id", "device_id"}.intersection(gpu)
    expected_names |= {
        "gpu_tiny_dispatch_1",
        "gpu_tiny_dispatch_batch_10",
        "gpu_saxpy_4096_upload_dispatch_wait",
    }
    expected_names |= {f"gpu_vector_add_{count}_dispatch_wait" for count in vector_sizes[1:]}
    expected_names |= {
        f"gpu_vector_add_{count}_deferred_readback_dispatch_wait" for count in vector_sizes[1:]
    }
    expected_names |= {
        "gpu_reduction_4096_dispatch_wait",
        "gpu_tiled_scan_4096_dispatch_wait",
        "gpu_tiled_matmul_64_dispatch_wait",
        "gpu_stencil_4096_dispatch_wait",
        "command_pool_reuse_steady_state",
        "command_pool_cold_create_record_destroy",
    }
    expected_names |= {
        f"staging_ring_reserve_{layout}_{count}"
        for layout in ("dense", "fragmented")
        for count in (0, 1, 8, 32, 128, 512)
    }
    expected_names |= {
        f"raw_vulkan_vector_add_{count}_dispatch_wait" for count in vector_sizes[1:]
    }
    expected_names |= {
        f"gpu_vector_add_{count}_gpu_execution" for count in gpu["kora_gpu_timestamp_sizes"]
    }
    expected_names |= {
        f"raw_vulkan_vector_add_{count}_gpu_execution" for count in gpu["raw_gpu_timestamp_sizes"]
    }
    for count in (4096, 65536, 1048576):
        expected_names |= {
            f"transfer_direct_upload_{count}",
            f"transfer_staged_upload_{count}",
            f"transfer_direct_download_{count}",
            f"transfer_staged_download_{count}",
            f"raw_vulkan_copy_{count}",
            f"kora_transfer_copy_{count}",
        }
assert {item["name"] for item in report["results"]} == expected_names
for result in report["results"]:
    assert result["unit"] == "ns"
    assert result["timing_domain"]
    assert len(result["samples"]) == 3
    assert result["median"] <= result["p95"]
    if "_parallel_" in result["name"]:
        assert result["thread_count"] == 4
    if result["name"].startswith(("transfer_", "raw_vulkan_copy_", "kora_transfer_copy_")):
        assert result["bytes_transferred"] in (4096, 65536, 1048576)
        assert result["throughput_bytes_per_second"] > 0
    if "--gpu" in arguments and result["name"].startswith(("gpu_", "transfer_", "raw_vulkan_", "kora_transfer_copy_", "command_pool_")):
        assert result["device_index"] == report["gpu_measurements"]["device_index"]
        assert result["path"] in report["gpu_measurements"]["paths"]
