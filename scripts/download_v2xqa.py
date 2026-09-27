#!/usr/bin/env python
"""Download the V2X-QA dataset for use as the main GSC-Token experiment dataset.

V2X-QA is a multi-view (ego / infrastructure / collaborative) multiple-choice VQA
benchmark for V2X autonomous driving. It matches the reframed scenario where K
vehicles emit semantic text reports and an RSU text coordinator fuses them.

Mirrors the project's existing ``datasets/download.py`` (HF mirror endpoint).

Usage (run WHERE outbound network to the HF mirror is available):
    HF_ENDPOINT=https://hf-mirror.com python scripts/download_v2xqa.py
    # annotations only first (small), to inspect the schema before pulling images:
    V2XQA_ANNOT_ONLY=1 python scripts/download_v2xqa.py

Override the repo / destination with env vars V2XQA_REPO, V2XQA_LOCAL.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

REPO_ID = os.environ.get("V2XQA_REPO", "junwei0001/V2X-QA")
LOCAL = os.environ.get("V2XQA_LOCAL", "data/raw/V2X-QA")
ANNOT_ONLY = os.environ.get("V2XQA_ANNOT_ONLY", "0") == "1"


def main() -> None:
    from huggingface_hub import snapshot_download

    local = Path(LOCAL)
    local.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO_ID} (repo_type=dataset) -> {local}")
    if ANNOT_ONLY:
        print("ANNOT_ONLY=1: pulling annotations/README only (no images).")
    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(local),
        resume_download=True,
        allow_patterns=(
            ["*.json", "*.jsonl", "*.csv", "*.parquet", "README*", "*.md",
             "*.txt", "*.yaml", "*.yml"] if ANNOT_ONLY else None
        ),
    )
    print("DONE ->", path)


if __name__ == "__main__":
    main()
