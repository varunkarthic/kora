#!/usr/bin/env python3
"""Fail a pull request check when its head does not contain the current base branch tip."""

from __future__ import annotations

import argparse
import subprocess


def branch_contains_base(base_ref: str, head_ref: str, repo: str = ".") -> bool:
    result = subprocess.run(
        ["git", "-C", repo, "merge-base", "--is-ancestor", base_ref, head_ref],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"git merge-base failed with exit code {result.returncode}")
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_ref", help="freshly fetched target branch ref")
    parser.add_argument("head_ref", help="freshly fetched pull-request head ref")
    args = parser.parse_args()
    if not branch_contains_base(args.base_ref, args.head_ref):
        print(
            "Pull request branch is behind its target. Fetch the latest base branch and merge or rebase it "
            "before requesting review; this keeps changelog entries attached to the release that contains them."
        )
        return 1
    print("Pull request branch contains the latest target branch tip.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
