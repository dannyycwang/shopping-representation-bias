from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import psutil


P2 = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    args = parser.parse_args()
    try:
        process = psutil.Process(args.pid)
        print(f"waiting for active primary runner pid={args.pid}", flush=True)
        process.wait()
    except psutil.NoSuchProcess:
        pass
    print("primary runner finished; starting resume-safe remaining matrix", flush=True)
    subprocess.run(
        [sys.executable, "-X", "utf8", "-u", str(P2 / "scripts/run_remaining.py")],
        cwd=P2.parent,
        check=True,
    )


if __name__ == "__main__":
    main()
