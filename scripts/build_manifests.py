#!/usr/bin/env python
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_dataset_adapter
from src.utils.io import write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["scienceqa", "gqa", "aokvqa", "v2xqa"])
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--data-dir")
    args = parser.parse_args()
    adapter = load_dataset_adapter(args.dataset, args.split, data_dir=args.data_dir)
    indices = list(range(len(adapter)))
    if args.num_samples is not None:
        indices = sorted(random.Random(args.seed).sample(indices, min(args.num_samples, len(indices))))
    rows = [{"sample_id": adapter[i]["sample_id"], "dataset": args.dataset, "split": args.split}
            for i in indices]
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} fixed IDs to {args.output} (seed={args.seed})")


if __name__ == "__main__": main()
