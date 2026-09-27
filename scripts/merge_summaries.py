#!/usr/bin/env python
"""Merge several channel-experiment summary.json files into one.

Later runs win on (channel, snr_db, method) collisions, so the base run should
come first and refinements last. The merged file keeps the first run's metadata
but the union of conditions, which is what the plotting scripts consume.
"""
from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    merged = json.load(open(args.base, encoding="utf-8"))
    index = {(c["channel"], float(c["snr_db"]), c["method"]): c
             for c in merged["conditions"]}
    for path in args.extra:
        other = json.load(open(path, encoding="utf-8"))
        for cond in other["conditions"]:
            index[(cond["channel"], float(cond["snr_db"]), cond["method"])] = cond
        merged["channels"] = sorted(set(merged["channels"]) | set(other["channels"]))
        merged["snr_db"] = sorted(set(map(float, merged["snr_db"]))
                                  | set(map(float, other["snr_db"])))

    merged["conditions"] = sorted(index.values(),
                                  key=lambda c: (c["channel"], c["snr_db"], c["method"]))
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, ensure_ascii=False, indent=1)
    print("merged %d runs -> %s (%d conditions, snr=%s)"
          % (1 + len(args.extra), args.output, len(merged["conditions"]),
             merged["snr_db"]))


if __name__ == "__main__":
    main()
