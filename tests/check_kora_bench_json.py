#!/usr/bin/env python3
"""Validate tiny benchmark reports and their percentile/sample invariants."""

import json
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
assert report["schema_version"] == 2
assert report["clock"] == "steady_clock"
assert report["iterations"] == 3
assert report["warmup"] == 1
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
if "--gpu" in arguments:
    gpu = report["gpu_measurements"]
    assert gpu["backend"] == "vulkan"
    assert isinstance(gpu["device_index"], int) and gpu["device_index"] >= 0
    assert gpu["device_name"] and gpu["api_version"] and gpu["driver_name"]
    assert isinstance(gpu["compute_queue_family"], int)
    assert gpu["same_physical_device_for_kora_and_raw_vulkan"] is True
    assert set(gpu["paths"]) == {"kora_runtime", "kora_transfer_engine", "raw_vulkan"}
    assert not {"uuid", "device_uuid", "vendor_id", "device_id"}.intersection(gpu)
    expected_names |= {
        "gpu_tiny_dispatch_1",
        "gpu_tiny_dispatch_batch_10",
        "gpu_saxpy_4096_upload_dispatch_wait",
    }
    expected_names |= {f"gpu_vector_add_{count}_dispatch_wait" for count in vector_sizes[1:]}
    expected_names |= {
        "gpu_reduction_4096_dispatch_wait",
        "gpu_tiled_scan_4096_dispatch_wait",
        "gpu_tiled_matmul_64_dispatch_wait",
        "gpu_stencil_4096_dispatch_wait",
        "raw_vulkan_vector_add_4096_dispatch_wait",
        "command_pool_reuse_steady_state",
        "command_pool_cold_create_record_destroy",
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
    assert len(result["samples"]) == 3
    assert result["median"] <= result["p95"]
    if result["name"].startswith(("transfer_", "raw_vulkan_copy_", "kora_transfer_copy_")):
        assert result["bytes_transferred"] in (4096, 65536, 1048576)
        assert result["throughput_bytes_per_second"] > 0
    if "--gpu" in arguments and result["name"].startswith(("gpu_", "transfer_", "raw_vulkan_", "kora_transfer_copy_", "command_pool_")):
        assert result["device_index"] == report["gpu_measurements"]["device_index"]
        assert result["path"] in report["gpu_measurements"]["paths"]
