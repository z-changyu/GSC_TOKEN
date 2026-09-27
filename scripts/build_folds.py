#!/usr/bin/env python
"""Split a frozen manifest into stratified folds.

The paper stratify-samples the image-containing ScienceQA instances and uses
five-fold cross-validation. Here the folds are built inside every stratum so
that each fold keeps the stratum proportions of the whole set, and the dealing
is a plain round-robin over a seeded shuffle, which keeps the assignment
reproducible and free of leakage between folds.

Outputs a fold index plus one manifest per (fold, role) so that the existing
``--manifest`` entry point can consume a fold without further plumbing.
"""
from __future__ import annotations

import argparse, json, random, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io import read_jsonl, write_jsonl


def stratum_key(row: dict, keys: list[str]) -> str:
    return "|".join(str(row.get(key, "")) for key in keys)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/scienceqa_val_local.jsonl")
    parser.add_argument("--processed", default="data/processed/scienceqa_validation_local.jsonl")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--stratify-by", default="subject",
                        help="Comma-separated metadata fields, e.g. subject,grade")
    parser.add_argument("--index", default="data/manifests/scienceqa_val_local_folds.json")
    parser.add_argument("--fold-dir", default="data/manifests/folds")
    parser.add_argument("--prefix", default="scienceqa_local")
    args = parser.parse_args()

    manifest = read_jsonl(args.manifest)
    processed = {str(row["row_idx"]): row for row in read_jsonl(args.processed)}
    keys = [k.strip() for k in args.stratify_by.split(",") if k.strip()]
    num_folds = args.num_folds
    if num_folds < 2:
        raise SystemExit("need at least two folds")

    buckets: dict[str, list[str]] = defaultdict(list)
    for entry in manifest:
        sample_id = str(entry["sample_id"])
        row = processed.get(sample_id)
        if row is None:
            raise SystemExit(f"sample {sample_id} from the manifest is missing from the processed set")
        buckets[stratum_key(row, keys)].append(sample_id)

    # Round-robin dealing inside each stratum keeps the fold sizes balanced to
    # within one sample and preserves the stratum proportions in every fold.
    folds: list[list[str]] = [[] for _ in range(num_folds)]
    stratum_sizes = {}
    for stratum, members in sorted(buckets.items()):
        rng = random.Random(f"{args.seed}:{stratum}")
        shuffled = sorted(members)
        rng.shuffle(shuffled)
        for position, sample_id in enumerate(shuffled):
            folds[position % num_folds].append(sample_id)
        stratum_sizes[stratum] = len(members)

    fold_dir = Path(args.fold_dir)
    index = {
        "dataset": "scienceqa",
        "split": "validation",
        "manifest": args.manifest,
        "processed": args.processed,
        "num_folds": num_folds,
        "seed": args.seed,
        "stratify_by": keys,
        "stratum_sizes": stratum_sizes,
        "folds": [],
    }
    for fold, test_ids in enumerate(folds):
        test_ids = sorted(test_ids)
        test_set = set(test_ids)
        train_ids = sorted(str(entry["sample_id"]) for entry in manifest
                           if str(entry["sample_id"]) not in test_set)
        test_path = fold_dir / f"{args.prefix}_fold{fold}_test.jsonl"
        train_path = fold_dir / f"{args.prefix}_fold{fold}_train.jsonl"
        write_jsonl(test_path, [{"sample_id": sid, "dataset": "scienceqa",
                                 "split": "validation"} for sid in test_ids])
        write_jsonl(train_path, [{"sample_id": sid, "dataset": "scienceqa",
                                  "split": "validation"} for sid in train_ids])
        index["folds"].append({"fold": fold, "test_size": len(test_ids),
                               "train_size": len(train_ids),
                               "test_manifest": str(test_path),
                               "train_manifest": str(train_path)})

    Path(args.index).parent.mkdir(parents=True, exist_ok=True)
    Path(args.index).write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n",
                                encoding="utf-8")
    print(json.dumps({"num_folds": num_folds, "seed": args.seed, "stratify_by": keys,
                      "strata": len(buckets),
                      "fold_sizes": [len(f) for f in folds],
                      "index": args.index}, indent=2))


if __name__ == "__main__":
    main()
