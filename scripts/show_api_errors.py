#!/usr/bin/env python3
"""
Print the last N lines of uvicorn.log and extract the most recent traceback block.
Usage:
  python scripts/show_api_errors.py --lines 200
"""
import argparse, os, sys, re
from collections import deque

def tail(path: str, n: int) -> list[str]:
    if not os.path.exists(path):
        print(f"Log file not found: {path}")
        return []
    dq = deque(maxlen=n)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            dq.append(line.rstrip("\n"))
    return list(dq)

def extract_last_traceback(lines: list[str]) -> list[str]:
    tb_start = None
    for i in range(len(lines)-1, -1, -1):
        if lines[i].startswith("Traceback (most recent call last):"):
            tb_start = i
            break
    if tb_start is None:
        # Sometimes uvicorn formats exceptions prefixed with "ERROR:" lines; try a looser search
        for i in range(len(lines)-1, -1, -1):
            if "Traceback" in lines[i]:
                tb_start = i
                break
    if tb_start is None:
        return []
    return lines[tb_start:]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="uvicorn.log")
    ap.add_argument("--lines", type=int, default=200)
    args = ap.parse_args()

    lines = tail(args.log, args.lines)
    if not lines:
        sys.exit(0)

    print(f"\n─── Last {len(lines)} lines of {args.log} ───")
    print("\n".join(lines))

    tb = extract_last_traceback(lines)
    if tb:
        print("\n─── Extracted traceback ───")
        print("\n".join(tb))
    else:
        print("\n(No traceback block detected in the last lines.)")

if __name__ == "__main__":
    main()
