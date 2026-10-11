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
assert report["schema_version"] == 1
assert report["clock"] == "steady_clock"
assert report["iterations"] == 3
assert report["warmup"] == 1
expected_names = {
    "scheduler_cost",
    "task_graph_finalize_64",
    "task_graph_replay_64",
    "allocator_allocate_free_256",
    "allocator_fragmentation_128",
    "host_allocation_4k",
}
expected_names |= {f"cpu_vector_add_{count}" for count in (1, 64, 256, 1024, 4096)}
if "--gpu" in arguments:
    expected_names |= {
        "gpu_tiny_dispatch_1",
        "gpu_saxpy_4096_end_to_end",
    }
    expected_names |= {f"gpu_vector_add_{count}_end_to_end" for count in (64, 256, 1024, 4096)}
    expected_names |= {
        "gpu_reduction_4096_end_to_end",
        "gpu_tiled_scan_4096_end_to_end",
        "gpu_tiled_matmul_64_end_to_end",
        "gpu_stencil_4096_end_to_end",
    }
assert {item["name"] for item in report["results"]} == expected_names
for result in report["results"]:
    assert result["unit"] == "ns"
    assert len(result["samples"]) == 3
    assert result["median"] <= result["p95"]
