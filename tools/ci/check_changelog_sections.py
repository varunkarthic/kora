#!/usr/bin/env python3
"""Reject pull requests that modify changelog sections for already-published releases."""

from __future__ import annotations

import argparse
import re
import subprocess


RELEASE_HEADING = re.compile(r"^## \[(\d+\.\d+\.\d+)\](?:\s+-.*)?$")
APPROVED_CORRECTIONS = {
    "0.2.5": (
        "- Synchronize CPU executor task-drain counter updates with the worker wait mutex so graceful shutdown\n"
        "  cannot lose the final wakeup and strand idle workers."
    ),
}


def release_sections(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current_version: str | None = None
    current_lines: list[str] = []

    def save() -> None:
        if current_version is not None:
            sections[current_version] = "\n".join(current_lines).strip()

    for line in text.splitlines():
        if line.startswith("## ["):
            save()
            heading = RELEASE_HEADING.match(line)
            current_version = heading.group(1) if heading else None
            current_lines = [line] if current_version is not None else []
        elif current_version is not None:
            current_lines.append(line)
    save()
    return sections


def unreleased_entries(text: str) -> set[str]:
    """Return bullet entries beneath the Unreleased heading and its categories."""
    lines = text.splitlines()
    headings = [i for i, line in enumerate(lines) if line.strip() == "## [Unreleased]"]
    if len(headings) != 1:
        raise ValueError("CHANGELOG.md must contain exactly one Unreleased heading")
    start = headings[0] + 1
    entries: set[str] = set()
    category: str | None = None
    for line in lines[start:]:
        if line.startswith("## "):
            break
        if line.startswith("### "):
            category = line[4:].strip()
            if category not in {"Added", "Changed", "Deprecated", "Removed", "Fixed", "Security"}:
                category = None
        if re.match(r"^\s*-\s+\S", line):
            if category is not None:
                entries.add(f"{category}: {line.strip()}")
    return entries


def is_approved_correction(version: str, base_section: str, head_section: str) -> bool:
    correction = APPROVED_CORRECTIONS.get(version)
    if correction is None or base_section.count(correction) != 1:
        return False
    expected = base_section.replace(correction, "", 1)
    normalize_blank_lines = lambda section: re.sub(r"\n{3,}", "\n\n", section).strip()
    return normalize_blank_lines(expected) == normalize_blank_lines(head_section)


def validate(
    base_text: str,
    head_text: str,
    allowed_versions: set[str] | None = None,
    require_unreleased_entry: bool = False,
) -> None:
    if require_unreleased_entry:
        base_entries = unreleased_entries(base_text)
        head_entries = unreleased_entries(head_text)
        if not head_entries - base_entries:
            raise ValueError(
                "pull request must add a bullet under [Unreleased] in CHANGELOG.md; "
                "use an Added, Changed, Deprecated, Removed, Fixed, or Security heading"
            )
    base_sections = release_sections(base_text)
    head_sections = release_sections(head_text)
    changed = []
    for version in base_sections.keys() | head_sections.keys():
        base_section = base_sections.get(version)
        head_section = head_sections.get(version)
        if base_section == head_section:
            continue
        if (
            version in (allowed_versions or set())
            and base_section is not None
            and head_section is not None
            and is_approved_correction(version, base_section, head_section)
        ):
            continue
        changed.append(version)
    changed.sort()
    if not changed:
        return
    versions = ", ".join(changed)
    raise ValueError(
        f"pull request changes published changelog section(s): {versions}; "
        "keep released sections immutable and place new entries under [Unreleased]"
    )


def git_show(ref: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{ref}:CHANGELOG.md"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"unable to read CHANGELOG.md at {ref}: {result.stderr.strip()}")
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_ref", help="freshly fetched pull-request target ref")
    parser.add_argument("head_ref", help="freshly fetched pull-request head ref")
    parser.add_argument(
        "--allow-version",
        action="append",
        default=[],
        help="explicitly reviewed historical release section correction",
    )
    parser.add_argument(
        "--require-unreleased-entry",
        action="store_true",
        help="require a new categorized bullet beneath [Unreleased]",
    )
    args = parser.parse_args()
    try:
        validate(
            git_show(args.base_ref),
            git_show(args.head_ref),
            set(args.allow_version),
            require_unreleased_entry=args.require_unreleased_entry,
        )
    except (RuntimeError, ValueError) as error:
        print(error)
        return 1
    if args.allow_version:
        versions = ", ".join(sorted(args.allow_version))
        print(f"Published changelog sections validated; explicit correction(s) allowed for: {versions}.")
    else:
        print("Pull request leaves all published changelog sections unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
