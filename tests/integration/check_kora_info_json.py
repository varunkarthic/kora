#!/usr/bin/env python3
import json
import subprocess
import sys


result = subprocess.run([sys.argv[1], "--json"], capture_output=True, text=True, check=False)
if result.returncode != 0:
    if "NoVulkanImplementation" in result.stderr:
        raise SystemExit(77)
    raise SystemExit(f"kora-info --json failed: {result.stderr}")

report = json.loads(result.stdout)
human = subprocess.run([sys.argv[1]], capture_output=True, text=True, check=True).stdout
assert report["schema_version"] == 2
assert isinstance(report["kora_version"], str)
assert report["pipeline_cache_state"] == "not_queried"
assert 'Pipeline cache state: not queried' in human
assert "kernel cache is not implemented" not in human
assert isinstance(report["devices"], list) and report["devices"]
for device in report["devices"]:
    assert isinstance(device["name"], str) and device["name"]
    assert "cache_state" not in device
    assert isinstance(device["features"]["synchronization2"], bool)
    assert isinstance(device["features"]["buffer_device_address"], bool)
    assert isinstance(device["queue_families"], list) and device["queue_families"]
    assert isinstance(device["memory"]["heaps"], list) and device["memory"]["heaps"]
    assert isinstance(device["memory"]["types"], list) and device["memory"]["types"]
