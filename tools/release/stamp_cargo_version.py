#!/usr/bin/env python3
"""Set the workspace package version in Cargo.toml (used by release packaging)."""

import re
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2 or not re.fullmatch(r"\d+\.\d+\.\d+", sys.argv[1]):
        raise SystemExit("usage: stamp_cargo_version.py MAJOR.MINOR.PATCH")
    path = Path("Cargo.toml")
    text = path.read_text(encoding="utf-8")
    stamped, count = re.subn(
        r'(\[workspace\.package\][^\[]*?\nversion = ")[^"]*(")', rf"\g<1>{sys.argv[1]}\g<2>", text, count=1
    )
    if count != 1:
        raise SystemExit("Cargo.toml has no [workspace.package] version")
    path.write_text(stamped, encoding="utf-8")
    print(f"Cargo workspace version set to {sys.argv[1]}")


if __name__ == "__main__":
    main()
