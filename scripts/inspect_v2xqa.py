#!/usr/bin/env python
"""Inspect a locally downloaded V2X-QA tree and print its schema.

Run AFTER ``download_v2xqa.py`` (annotations-only is enough). Prints the
top-level directory layout, locates the first annotation file, and dumps one
sample's keys/values so the ``v2xqa`` adapter can be written against the REAL
format instead of guesses.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("V2XQA_LOCAL", "data/raw/V2X-QA"))


def _first_annotation(root: Path):
    for ext in ("*.json", "*.jsonl", "*.csv", "*.parquet"):
        hits = list(root.rglob(ext))
        if hits:
            return hits[0]
    return None


def main() -> None:
    if not ROOT.exists():
        sys.exit(f"no data at {ROOT}; run download_v2xqa.py first")
    print("== top-level entries ==")
    for p in sorted(ROOT.iterdir()):
        print("  ", p.name, ("<dir>" if p.is_dir() else f"{p.stat().st_size} B"))

    found = _first_annotation(ROOT)
    if found is None:
        print("\nno annotation file found; listing a few files for context:")
        for p in list(ROOT.rglob("*"))[:30]:
            print("  ", p.relative_to(ROOT))
        return

    print(f"\n== sample annotation file: {found.relative_to(ROOT)} ==")
    text = found.read_text(encoding="utf-8")
    if found.suffix == ".jsonl":
        first = json.loads(text.splitlines()[0])
    elif found.suffix == ".csv":
        print("(csv) first line:\n", text.splitlines()[0])
        return
    else:
        data = json.loads(text)
        first = data[0] if isinstance(data, list) else next(iter(data.values()))

    print("keys:", list(first.keys()))
    print(json.dumps(first, indent=2, ensure_ascii=False)[:2500])


if __name__ == "__main__":
    main()
