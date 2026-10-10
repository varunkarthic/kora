#!/usr/bin/env python3
"""Perform offline syntax and contract checks for KORA workflow files."""

from pathlib import Path

import yaml


def main() -> None:
    workflows = sorted(Path(".github/workflows").glob("*.yml"))
    if not workflows:
        raise SystemExit("no workflow files found")
    for path in workflows:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or "jobs" not in document:
            raise SystemExit(f"{path}: expected a workflow mapping with jobs")
        if "on" not in document and True not in document:  # PyYAML 1.1 parses the key `on` as true.
            raise SystemExit(f"{path}: missing event trigger")
        for name, job in document["jobs"].items():
            if not isinstance(job, dict) or not ("steps" in job or "uses" in job):
                raise SystemExit(f"{path}: job {name!r} has no steps or reusable workflow")
        print(f"validated {path}")


if __name__ == "__main__":
    main()
