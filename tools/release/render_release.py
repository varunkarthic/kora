#!/usr/bin/env python3
"""Render milestone-backed GitHub release notes and Beta/Stable metadata."""

from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path


def api_get(repo: str, path: str, token: str) -> object:
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repo}{path}",
        headers={"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def milestone_issues(repo: str, milestone_title: str, token: str) -> list[dict[str, object]]:
    milestones = api_get(repo, "/milestones?state=all&per_page=100", token)
    milestone = next((item for item in milestones if item["title"] == milestone_title), None)
    if milestone is None:
        raise RuntimeError(f"missing release-validation milestone {milestone_title}")
    query = urllib.parse.urlencode({"state": "all", "milestone": milestone["number"], "per_page": 100})
    issues = api_get(repo, f"/issues?{query}", token)
    return sorted((item for item in issues if "pull_request" not in item), key=lambda item: item["number"])


def render(repo: str, version: str, changelog: str, issues: list[dict[str, object]]) -> tuple[str, bool]:
    closed = sum(issue["state"] == "closed" for issue in issues)
    stable = bool(issues) and closed == len(issues)
    channel = "Stable" if stable else "Beta"
    badge_color = "brightgreen" if stable else "orange"
    state_badge = f"{closed}%20of%20{len(issues)}"
    warning = (
        f"> [!NOTE]\n> **Stable.** All {len(issues)} release-validation issues are closed."
        if stable
        else f"> [!WARNING]\n> **Beta.** {len(issues) - closed} of {len(issues)} release-validation issues remain open. "
        "Evaluate this release before relying on it for production workloads."
    )
    rows = "\n".join(
        f"| #{issue['number']} | [{issue['title']}]({issue['html_url']}) | "
        f"{'closed' if issue['state'] == 'closed' else 'open'} |"
        for issue in issues
    )
    body = f"""![release {channel}](https://img.shields.io/badge/release-{channel.lower()}-{badge_color})
![validation issues](https://img.shields.io/badge/validation-{state_badge}-blue)

{warning}

## Install and use

Download the archive for your platform, extract it, and point `CMAKE_PREFIX_PATH` at the extracted
installation prefix:

```cmake
find_package(kora CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE kora::kora)
```

Or build from the tag:

```sh
git clone --branch v{version} https://github.com/{repo}.git
cd kora
cmake --preset release -DKORA_VERSION_PATCH={version.rsplit('.', 1)[1]}
cmake --build --preset release --parallel
ctest --preset release --output-on-failure
```

## Release-validation issues

| No. | Issue | Status |
| --- | --- | --- |
{rows}

## Changelog

{changelog.strip()}

Release status is refreshed automatically when a tracked issue changes.
"""
    return body, stable


def release_changelog(text: str, version: str) -> str:
    marker = f"## [{version}]"
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"CHANGELOG.md has no {marker} section")
    candidates = [
        position
        for position in (
            text.find("\n## [", start + len(marker)),
            text.find("\n[Unreleased]:", start + len(marker)),
        )
        if position >= 0
    ]
    end = min(candidates) if candidates else len(text)
    return text[start : end if end >= 0 else len(text)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN is required")
    base = ".".join(args.version.split(".")[:2])
    issues = milestone_issues(args.repo, f"v{base}", token)
    changelog = release_changelog(Path("CHANGELOG.md").read_text(encoding="utf-8"), args.version)
    body, stable = render(args.repo, args.version, changelog, issues)
    args.output.write_text(body, encoding="utf-8")
    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as output:
            output.write(f"stable={'true' if stable else 'false'}\n")
            output.write(f"channel={'Stable' if stable else 'Beta'}\n")


if __name__ == "__main__":
    main()
