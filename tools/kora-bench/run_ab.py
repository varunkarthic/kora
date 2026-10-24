#!/usr/bin/env python3
"""Run two kora-bench builds in interleaved ABBA pairs on one device and keep every report.

Pair i runs A then B when i is even and B then A when it is odd, so slow drift in GPU clocks or host
load cannot favor either build. Reports are written gzip-compressed as <out>/a-NN.json.gz and
<out>/b-NN.json.gz for render_ab.py.
"""

from __future__ import annotations

import argparse
import gzip
import os
import subprocess
import sys
import time
from pathlib import Path


def run(binary: Path, arguments: list[str], destination: Path, environment: dict[str, str]) -> None:
    completed = subprocess.run([str(binary), *arguments], capture_output=True, env=environment, check=False)
    if completed.returncode != 0:
        sys.stderr.write(completed.stderr.decode("utf-8", "replace"))
        raise SystemExit(f"{binary} exited with {completed.returncode}")
    destination.write_bytes(gzip.compress(completed.stdout, mtime=0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True, help="baseline kora-bench")
    parser.add_argument("--b", type=Path, required=True, help="candidate kora-bench")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--device", help="KORA_DEVICE for both builds")
    parser.add_argument("bench_args", nargs="*", default=["--gpu", "--cluster"])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    if args.device is not None:
        environment["KORA_DEVICE"] = args.device
    for pair in range(args.pairs):
        order = (("a", args.a), ("b", args.b)) if pair % 2 == 0 else (("b", args.b), ("a", args.a))
        for side, binary in order:
            started = time.monotonic()
            run(binary, args.bench_args, args.out / f"{side}-{pair:02d}.json.gz", environment)
            print(f"pair {pair} {side}: {time.monotonic() - started:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
