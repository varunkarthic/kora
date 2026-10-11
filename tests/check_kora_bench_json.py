#!/usr/bin/env python3
"""Validate the tiny benchmark smoke report and its percentile/sample invariants."""

import json
import subprocess
import sys


report = json.loads(subprocess.check_output([sys.argv[1], "--smoke"], text=True))
assert report["schema_version"] == 1
assert report["clock"] == "steady_clock"
assert report["iterations"] == 3
assert report["warmup"] == 1
assert {item["name"] for item in report["results"]} == {
    "scheduler_cost",
    "task_graph_finalize_64",
    "task_graph_replay_64",
    "allocator_allocate_free_256",
    "allocator_fragmentation_128",
    "host_allocation_4k",
}
for result in report["results"]:
    assert result["unit"] == "ns"
    assert len(result["samples"]) == 3
    assert result["median"] <= result["p95"]
