#!/usr/bin/env python
"""Fetch a deterministic ScienceQA subset via the official Dataset Viewer API.

This is a narrow fallback for networks where Hub file HEAD/large CDN downloads fail.
It does not mark the full dataset download as complete.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.io import write_jsonl  # noqa: E402


API = "https://datasets-server.huggingface.co/rows"
DATASET = "derek-thomas/ScienceQA"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation")
    parser.add_argument("--num-samples", type=int, default=150)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-rows", type=int, default=4241,
                        help="Pinned validation size; fail if API reports a different value")
    parser.add_argument("--output", default="data/processed/scienceqa_validation_pilot.jsonl")
    parser.add_argument("--image-dir", default="data/processed/scienceqa_validation_images")
    args = parser.parse_args()
    selected = sorted(random.Random(args.seed).sample(range(args.num_rows), args.num_samples))
    wanted, rows = set(selected), {}
    session = requests.Session()
    retries = Retry(total=10, connect=10, read=10, backoff_factor=1.0,
                    status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
    session.mount("https://", HTTPAdapter(max_retries=retries))
    for block in sorted({idx // 100 for idx in selected}):
        response = session.get(API, params={"dataset": DATASET, "config": "default", "split": args.split,
                                            "offset": block * 100, "length": 100}, timeout=120)
        response.raise_for_status()
        payload = response.json()
        reported = payload.get("num_rows_total")
        if reported is not None and reported != args.num_rows:
            raise RuntimeError(f"Dataset size changed: expected {args.num_rows}, API reports {reported}")
        for item in payload["rows"]:
            if item["row_idx"] in wanted:
                rows[item["row_idx"]] = dict(item["row"])
    missing = wanted - rows.keys()
    if missing:
        raise RuntimeError(f"API omitted selected rows: {sorted(missing)}")
    image_dir = Path(args.image_dir); image_dir.mkdir(parents=True, exist_ok=True)
    output = []
    for idx in selected:
        row = rows[idx]
        image = row.get("image")
        if isinstance(image, dict) and image.get("src"):
            suffix = Path(image["src"].split("?", 1)[0]).suffix or ".jpg"
            target = image_dir / f"{idx}{suffix}"
            valid = False
            if target.exists() and target.stat().st_size > 0:
                try:
                    with Image.open(target) as loaded:
                        loaded.verify()
                    valid = True
                except Exception:
                    valid = False
            if not valid:
                partial = target.with_suffix(target.suffix + ".part")
                with session.get(image["src"], timeout=120, stream=True) as response:
                    response.raise_for_status()
                    with partial.open("wb") as handle:
                        for chunk in response.iter_content(1024 * 1024):
                            handle.write(chunk)
                with Image.open(partial) as loaded:
                    loaded.verify()
                partial.replace(target)
            row["image"] = str(target)
        else:
            row["image"] = None
        row["row_idx"] = idx
        output.append(row)
    write_jsonl(args.output, output)
    manifest = [{"sample_id": str(idx), "dataset": "scienceqa", "split": args.split} for idx in selected]
    write_jsonl("data/manifests/scienceqa_val.jsonl", manifest)
    print(json.dumps({"samples": len(output), "images": sum(r["image"] is not None for r in output),
                      "seed": args.seed, "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
