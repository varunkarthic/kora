#!/usr/bin/env python3
"""Measure Python callback overhead for replaying a fixed KORA execution plan."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pathlib
import sys
import time
import types


def nearest_rank(samples: list[int], percentile: int) -> int:
    ordered = sorted(samples)
    return ordered[max(math.ceil(percentile * len(ordered) / 100), 1) - 1]


def load_kora(repo: pathlib.Path, extension: pathlib.Path) -> types.ModuleType:
    package_dir = repo / "python" / "kora"
    package = types.ModuleType("kora")
    package.__path__ = [str(package_dir)]
    package.__package__ = "kora"
    sys.modules["kora"] = package

    spec = importlib.util.spec_from_file_location("kora._core", extension)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load KORA extension: {extension}")
    core = importlib.util.module_from_spec(spec)
    sys.modules["kora._core"] = core
    spec.loader.exec_module(core)
    init_file = package_dir / "__init__.py"
    exec(compile(init_file.read_bytes(), str(init_file), "exec"), package.__dict__)
    return package


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--tasks", type=int, default=64)
    parser.add_argument("--extension", type=pathlib.Path)
    args = parser.parse_args()
    if min(args.iterations, args.tasks) <= 0 or args.warmup < 0:
        parser.error("iterations/tasks must be positive and warmup must be nonnegative")

    repo = pathlib.Path(__file__).resolve().parents[2]
    extensions = [args.extension] if args.extension else sorted(
        (repo / "build" / "python" / "bindings" / "python").glob("_core.*")
    )
    if not extensions or not extensions[0].is_file():
        print("Python benchmark unavailable: build the 'python' CMake preset first", file=sys.stderr)
        return 3

    kora = load_kora(repo, extensions[0])
    graph = kora.TaskGraph()
    for _ in range(args.tasks):
        metadata = kora.TaskMetadata()
        metadata.type = kora.TaskType.HOST_CALLBACK
        graph.add_task(metadata)
    plan = kora.ExecutionPlan.create(graph)
    invoked = 0

    def python_callback(_task: object) -> None:
        nonlocal invoked
        invoked += 1

    def replay() -> None:
        plan.replay(python_callback)

    for _ in range(args.warmup):
        replay()
    samples = []
    for _ in range(args.iterations):
        start = time.perf_counter_ns()
        replay()
        samples.append(time.perf_counter_ns() - start)

    report = {
        "schema_version": 1,
        "clock": "time.perf_counter_ns",
        "iterations": args.iterations,
        "warmup": args.warmup,
        "results": [
            {
                "name": f"python_execution_plan_replay_{args.tasks}",
                "unit": "ns",
                "median": nearest_rank(samples, 50),
                "p95": nearest_rank(samples, 95),
                "samples": samples,
                "callbacks_per_iteration": args.tasks,
                "total_callbacks": invoked,
            }
        ],
        "runtime_version": kora.version(),
        "note": "Includes Python callback crossings and interpreter scheduling; compare only on the same build and host.",
    }
    print(json.dumps(report, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
