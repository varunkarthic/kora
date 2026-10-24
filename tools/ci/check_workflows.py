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
    for contract in ("Verify MSVC selection", "x86_64-pc-windows-msvc"):
        if contract not in release:
            raise SystemExit(f"release workflow is missing Windows package contract: {contract}")
    if "ilammy/msvc-dev-cmd" in release:
        raise SystemExit("release workflow must not put MSVC on PATH: Git Bash's link would shadow link.exe")
    print("validated Windows release package compiler contract")

    release_document = yaml.safe_load(release)
    package_job = release_document["jobs"].get("package", {})
    if package_job.get("timeout-minutes") != 180:
        raise SystemExit("release package jobs must have a 180-minute job timeout")
    test_steps = [step for step in package_job.get("steps", []) if step.get("name") == "Test"]
    if (
        len(test_steps) != 1
        or "cargo test --workspace --locked" not in test_steps[0].get("run", "")
        or test_steps[0].get("timeout-minutes") != 60
    ):
        raise SystemExit("release tests must run the Cargo workspace with a 60-minute step timeout")
    publish_job = release_document["jobs"].get("publish", {})
    publish_condition = "${{ needs.package.result == 'success' }}"
    if "package" not in publish_job.get("needs", []) or publish_job.get("if") != publish_condition:
        raise SystemExit("release publishing must require success from the complete package matrix")
    print("validated bounded release tests and all-platform publication gate")

    ci = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    for contract in (
        "fetch-depth: 0",
        "Check pull request contains the latest base branch",
        "refs/pull/${PR_NUMBER}/head",
        "tools/ci/check_pr_branch_freshness.py",
        "tools/ci/check_changelog_sections.py",
        "Require a new Unreleased changelog entry",
        "--require-unreleased-entry",
        "--allow-version 0.2.5",
        "issue/176-changelog-tag-alignment",
    ):
        if contract not in ci:
            raise SystemExit(f"CI workflow is missing stale-branch protection: {contract}")
    print("validated pull-request freshness and changelog immutability contracts")


if __name__ == "__main__":
    main()
