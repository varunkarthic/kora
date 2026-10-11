import json
import pathlib
import subprocess
import sys

script = pathlib.Path(sys.argv[1])
extension = pathlib.Path(sys.argv[2])
completed = subprocess.run(
    [sys.executable, str(script), "--iterations", "3", "--warmup", "1", "--tasks", "2",
     "--extension", str(extension)],
    check=True,
    capture_output=True,
    text=True,
)
report = json.loads(completed.stdout)
assert report["schema_version"] == 1
assert report["iterations"] == 3
assert report["warmup"] == 1
assert len(report["results"]) == 1
result = report["results"][0]
assert result["name"] == "python_execution_plan_replay_2"
assert len(result["samples"]) == 3
assert result["callbacks_per_iteration"] == 2
assert result["total_callbacks"] == 8
assert result["median"] <= result["p95"]
