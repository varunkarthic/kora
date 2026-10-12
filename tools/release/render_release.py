#!/usr/bin/env python3
"""Render milestone-backed GitHub release notes and Beta/Stable metadata."""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path


def _pagination_links(link_header: str | None) -> tuple[str | None, str | None]:
    next_url = None
    last_url = None
    if not link_header:
        return next_url, last_url
    for entry in link_header.split(","):
        match = re.fullmatch(r'\s*<([^>]+)>\s*;\s*rel="?([^";]+)"?\s*', entry)
        if not match:
            if "rel=" in entry:
                raise RuntimeError("GitHub API returned a malformed pagination link")
            continue
        if match.group(2) == "next":
            if next_url is not None:
                raise RuntimeError("GitHub API returned duplicate next-page links")
            next_url = match.group(1)
        elif match.group(2) == "last":
            if last_url is not None:
                raise RuntimeError("GitHub API returned duplicate last-page links")
            last_url = match.group(1)
    return next_url, last_url


def _same_collection_path(path: str, base_path: str) -> bool:
    if path == base_path:
        return True
    collection = base_path.rsplit("/", 1)[-1]
    return re.fullmatch(rf"/repositories/\d+/{re.escape(collection)}", path) is not None


def api_get_pages(repo: str, path: str, token: str) -> list[dict[str, object]]:
    """Fetch every page; incomplete/malformed pagination is an error, never a partial success."""
    base_url = f"https://api.github.com/repos/{repo}{path}"
    parsed_base = urllib.parse.urlparse(base_url)
    query = urllib.parse.parse_qs(parsed_base.query)
    query["per_page"] = ["100"]
    query["page"] = ["1"]

    def page_url(page: int) -> str:
        page_query = {**query, "page": [str(page)]}
        return urllib.parse.urlunparse(parsed_base._replace(query=urllib.parse.urlencode(page_query, doseq=True)))

    current_url = page_url(1)
    expected_page = 1
    known_last_page: int | None = None
    promised_next_page = False
    expected_query = {key: value for key, value in query.items() if key != "page"}

    def pagination_query_matches(actual: dict[str, list[str]]) -> bool:
        stable_filters = {key: value for key, value in actual.items() if key not in {"page", "after", "before"}}
        cursors = {key: value for key, value in actual.items() if key in {"after", "before"}}
        return (
            stable_filters == expected_query
            and all(len(values) == 1 and values[0] for values in cursors.values())
            and not {"after", "before"}.issubset(cursors)
        )
    visited_urls: set[str] = set()
    seen_ids: set[int] = set()
    records: list[dict[str, object]] = []

    while current_url:
        if current_url in visited_urls:
            raise RuntimeError("GitHub API pagination loop detected")
        visited_urls.add(current_url)
        request = urllib.request.Request(
            current_url,
            headers={"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            page_records = json.load(response)
            next_url, last_url = _pagination_links(response.headers.get("Link"))
        if not isinstance(page_records, list) or any(not isinstance(item, dict) for item in page_records):
            raise RuntimeError("GitHub API returned a malformed paginated response")
        for item in page_records:
            item_id = item.get("number", item.get("id"))
            if type(item_id) is not int or item_id in seen_ids:
                raise RuntimeError("GitHub API pagination returned a missing or duplicate item identity")
            seen_ids.add(item_id)
        records.extend(page_records)

        if last_url:
            parsed_last = urllib.parse.urlparse(last_url)
            last_query = urllib.parse.parse_qs(parsed_last.query)
            last_page = last_query.get("page", [])
            if (
                parsed_last.scheme != "https"
                or parsed_last.netloc != parsed_base.netloc
                or not _same_collection_path(parsed_last.path, parsed_base.path)
                or len(last_page) != 1
                or not last_page[0].isdigit()
                or int(last_page[0]) < expected_page
                or not pagination_query_matches(last_query)
            ):
                raise RuntimeError("GitHub API returned an invalid last-page link")
            page_count = int(last_page[0])
            if known_last_page is not None and page_count != known_last_page:
                raise RuntimeError("GitHub API pagination changed its last-page count while reading")
            known_last_page = page_count
        if not page_records and (
            promised_next_page or (known_last_page is not None and expected_page <= known_last_page)
        ):
            raise RuntimeError("GitHub API returned an empty page before the declared last page")
        promised_next_page = False
        if not next_url:
            last_page_number = known_last_page if known_last_page is not None else expected_page
            more_pages = last_page_number > expected_page
            ambiguous_full_page = len(page_records) >= 100 and known_last_page is None
            if more_pages or ambiguous_full_page:
                current_query = urllib.parse.parse_qs(urllib.parse.urlparse(current_url).query)
                if any(key in current_query for key in ("after", "before")) or (
                    last_url and any(key in last_query for key in ("after", "before"))
                ):
                    raise RuntimeError("GitHub API omitted a required cursor pagination link")
                expected_page += 1
                current_url = page_url(expected_page)
                if expected_page > 10_000:
                    raise RuntimeError("GitHub API pagination exceeded the safety limit")
                continue
            break
        parsed_next = urllib.parse.urlparse(next_url)
        next_query = urllib.parse.parse_qs(parsed_next.query)
        page_values = next_query.get("page", [])
        if (
            parsed_next.scheme != "https"
            or parsed_next.netloc != parsed_base.netloc
            or not _same_collection_path(parsed_next.path, parsed_base.path)
            or len(page_values) != 1
            or not page_values[0].isdigit()
            or int(page_values[0]) != expected_page + 1
            or not pagination_query_matches(next_query)
        ):
            raise RuntimeError("GitHub API returned an invalid or non-sequential next-page link")
        if known_last_page is not None and int(page_values[0]) > known_last_page:
            raise RuntimeError("GitHub API next-page link exceeds the declared last page")
        expected_page += 1
        current_url = next_url
        promised_next_page = True
        if expected_page > 10_000:
            raise RuntimeError("GitHub API pagination exceeded the safety limit")
    return records


def milestone_issues(repo: str, milestone_title: str, token: str) -> list[dict[str, object]]:
    milestones = api_get_pages(repo, "/milestones?state=all", token)
    matches = [item for item in milestones if item.get("title") == milestone_title]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one release-validation milestone {milestone_title}")
    milestone_number = matches[0].get("number")
    if type(milestone_number) is not int:
        raise RuntimeError(f"malformed release-validation milestone {milestone_title}")
    query = urllib.parse.urlencode(
        {"state": "all", "milestone": milestone_number, "sort": "created", "direction": "asc"}
    )
    issues = api_get_pages(repo, f"/issues?{query}", token)
    result = [item for item in issues if "pull_request" not in item]
    for issue in result:
        if (
            type(issue.get("number")) is not int
            or not isinstance(issue.get("title"), str)
            or issue.get("state") not in ("open", "closed")
            or not isinstance(issue.get("html_url"), str)
        ):
            raise RuntimeError("GitHub API returned a malformed issue while rendering release status")
    return sorted(result, key=lambda item: item["number"])


def _escape_table_text(value: str) -> str:
    """Escape untrusted text for a Markdown table cell or link label."""
    return (
        value.replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def release_highlights(changelog: str, limit: int = 3) -> list[str]:
    """Return the first categorized changelog bullets as concise, source-derived highlights."""
    highlights: list[str] = []
    for line in changelog.splitlines():
        stripped = line.strip()
        if re.match(r"[-*+]\s+\S", stripped):
            highlights.append(re.sub(r"^[-*+]\s+", "", stripped))
            if len(highlights) == limit:
                break
    return highlights


def release_assets(repo: str, version: str) -> list[tuple[str, str, str, str]]:
    """The release workflow uploads exactly these three package assets before publishing notes."""
    base = f"https://github.com/{repo}/releases/download/v{version}/"
    return [
        ("Linux", "x86-64", f"kora-{version}-linux-x86_64.tar.gz", base + f"kora-{version}-linux-x86_64.tar.gz"),
        ("macOS", "ARM64", f"kora-{version}-macos-arm64.tar.gz", base + f"kora-{version}-macos-arm64.tar.gz"),
        ("Windows", "x86-64", f"kora-{version}-windows-x86_64.zip", base + f"kora-{version}-windows-x86_64.zip"),
    ]


def render(repo: str, version: str, changelog: str, issues: list[dict[str, object]]) -> tuple[str, bool]:
    for issue in issues:
        if (
            not isinstance(issue, dict)
            or type(issue.get("number")) is not int
            or not isinstance(issue.get("title"), str)
            or issue.get("state") not in ("open", "closed")
            or not isinstance(issue.get("html_url"), str)
        ):
            raise RuntimeError("cannot determine release stability from malformed issue data")
    closed = sum(issue["state"] == "closed" for issue in issues)
    stable = bool(issues) and closed == len(issues)
    channel = "Stable" if stable else "Beta"
    badge_color = "brightgreen" if stable else "orange"
    state_badge = f"{closed}%20of%20{len(issues)}"
    if stable:
        warning = f"> [!NOTE]\n> **Stable.** All {len(issues)} release-validation issues are closed."
    elif not issues:
        warning = "> [!WARNING]\n> **Beta.** No release-validation issues are associated with this milestone; an empty milestone is not eligible for Stable."
    else:
        warning = (
            f"> [!WARNING]\n> **Beta.** {len(issues) - closed} of {len(issues)} release-validation issues remain open. "
            "Evaluate this release before relying on it for production workloads."
        )
    open_issues = [issue for issue in issues if issue["state"] == "open"]
    closed_issues = [issue for issue in issues if issue["state"] == "closed"]

    def issue_rows(group: list[dict[str, object]], state: str) -> str:
        return "\n".join(
            f"| #{issue['number']} | [{_escape_table_text(issue['title'])}]({issue['html_url']}) | {state} |"
            for issue in group
        ) or "| — | None | — |"

    highlights = release_highlights(changelog)
    highlight_list = "\n".join(f"- {item}" for item in highlights) or "- See the full changelog below."
    asset_rows = "\n".join(
        f"| {os_name} | {architecture} | [{_escape_table_text(filename)}]({url}) |"
        for os_name, architecture, filename, url in release_assets(repo, version)
    )
    gate_summary = f"{closed} of {len(issues)} release-validation gates closed"
    body = f"""![release {channel}](https://img.shields.io/badge/release-{channel.lower()}-{badge_color})
![validation issues](https://img.shields.io/badge/validation-{state_badge}-blue)

# KORA v{version}

{warning}

**Release channel:** {channel} · **Validation:** {gate_summary} · **Version:** `{version}`

[Source at this tag](https://github.com/{repo}/tree/v{version}) · [Documentation](https://kora.readthedocs.io/en/latest/) · [Changelog](https://github.com/{repo}/blob/v{version}/CHANGELOG.md)

## Highlights

{highlight_list}

## Download packages

Each archive contains the installed CMake development prefix. Extract it, then set
`CMAKE_PREFIX_PATH` to the extracted prefix before configuring your consumer project.

| Platform | Architecture | Package |
| --- | --- | --- |
{asset_rows}

All three package jobs must pass before this release is published. Archives do not include a Vulkan
driver, MoltenVK, a Python runtime, or a system C/C++ runtime.

## Install and use

Link an installed package from CMake:

```cmake
find_package(kora CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE kora::kora)
```

Or build from the matching source tag:

```sh
git clone --branch v{version} https://github.com/{repo}.git
cd kora
cmake --preset release -DKORA_VERSION_PATCH={version.rsplit('.', 1)[1]}
cmake --build --preset release --parallel
ctest --preset release --output-on-failure
```

## Release validation

The release line remains Beta until every release-validation issue in its milestone is closed. Stable
means repository release gates are complete; it is not a workload-specific guarantee.

### Remaining gates ({len(open_issues)})

| No. | Issue | Status |
| --- | --- | --- |
{issue_rows(open_issues, '🟡 Open')}

### Completed gates ({len(closed_issues)})

| No. | Issue | Status |
| --- | --- | --- |
{issue_rows(closed_issues, '✅ Closed')}

## Changelog

{changelog.strip()}

Release status is refreshed automatically when a tracked issue changes. Download links above follow
the package filenames emitted by the required release matrix.
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
    section = text[start : end if end >= 0 else len(text)]
    if not re.search(r"(?m)^\s*[-*+]\s+\S", section):
        raise ValueError(f"CHANGELOG.md section for {version} has no release entries")
    return section


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
