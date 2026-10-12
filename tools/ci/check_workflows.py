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

    release = Path(".github/workflows/release.yml").read_text(encoding="utf-8")
    for contract in ("ilammy/msvc-dev-cmd@v1", "Verify MSVC selection", "cl.exe"):
        if contract not in release:
            raise SystemExit(f"release workflow is missing Windows package contract: {contract}")
    print("validated Windows release package compiler contract")

    ci = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    for contract in (
        "fetch-depth: 0",
        "Check pull request contains the latest base branch",
        "refs/pull/${PR_NUMBER}/head",
        "tools/ci/check_pr_branch_freshness.py",
    ):
        if contract not in ci:
            raise SystemExit(f"CI workflow is missing stale-branch protection: {contract}")
    print("validated pull-request base freshness contract")


if __name__ == "__main__":
    main()
