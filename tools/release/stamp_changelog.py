#!/usr/bin/env python3
"""Move Unreleased changelog entries into a dated release section."""

from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path


def stamp(text: str, version: str, previous_tag: str | None, date: str) -> str:
    heading = "## [Unreleased]"
    if text.count(heading) != 1:
        raise ValueError("CHANGELOG.md must contain exactly one Unreleased heading")
    if f"## [{version}]" in text:
        raise ValueError(f"version {version} is already present in CHANGELOG.md")

    text = text.replace(heading, f"{heading}\n\n## [{version}] - {date}", 1)
    link = re.compile(r"^\[Unreleased\]:.*$", re.MULTILINE)
    if not link.search(text):
        raise ValueError("CHANGELOG.md is missing the Unreleased comparison link")

    tag = f"v{version}"
    release_url = (
        f"https://github.com/varunkarthic/kora/compare/{previous_tag}...{tag}"
        if previous_tag
        else f"https://github.com/varunkarthic/kora/releases/tag/{tag}"
    )
    links = (
        f"[Unreleased]: https://github.com/varunkarthic/kora/compare/{tag}...HEAD\n"
        f"[{version}]: {release_url}"
    )
    return link.sub(links, text, count=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("version")
    parser.add_argument("--previous-tag")
    parser.add_argument("--date", default=dt.date.today().isoformat())
    parser.add_argument("--file", type=Path, default=Path("CHANGELOG.md"))
    args = parser.parse_args()

    original = args.file.read_text(encoding="utf-8")
    args.file.write_text(stamp(original, args.version, args.previous_tag, args.date), encoding="utf-8")


if __name__ == "__main__":
    main()
