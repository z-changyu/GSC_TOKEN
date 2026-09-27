#!/usr/bin/env python
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data import load_dataset_adapter


def describe(values):
    a = np.asarray(values, dtype=float)
    if not a.size: return {k: None for k in ["mean", "std", "median", "p25", "p75", "p95", "max"]}
    return {"mean": float(a.mean()), "std": float(a.std()), "median": float(np.median(a)),
            "p25": float(np.percentile(a, 25)), "p75": float(np.percentile(a, 75)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", choices=["scienceqa", "gqa", "aokvqa"])
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--data-dir", help="Local dataset directory or pilot JSONL (single dataset mode)")
    parser.add_argument("--tokenizer")
    parser.add_argument("--output-prefix", default="outputs/dataset_statistics")
    args = parser.parse_args()
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    records, details = [], {}
    for name in args.dataset or ["scienceqa", "gqa", "aokvqa"]:
        adapter = load_dataset_adapter(name, args.split, data_dir=args.data_dir)
        n = min(len(adapter), args.max_samples or len(adapter))
        samples = [adapter[i] for i in range(n)]
        qlens = [len(s["question"].split()) for s in samples]
        choices = [len(s["choices"] or []) for s in samples]
        missing = [sum(s[k] in (None, "", []) for k in ("question", "ground_truth")) for s in samples]
        token_lengths = [len(tokenizer(s["question"])["input_ids"]) for s in samples] if tokenizer else []
        row = {"dataset": name, "split": args.split, "number_of_samples": n,
               "image_availability": sum(s["image"] is not None for s in samples) / n if n else 0,
               "missing_data_ratio": sum(missing) / (2 * n) if n else 0,
               "mean_question_words": np.mean(qlens) if qlens else None,
               "mean_answer_choices": np.mean(choices) if choices else None,
               "answer_type_distribution": json.dumps(Counter(s["answer_type"] for s in samples))}
        records.append(row)
        details[name] = {"question_words": describe(qlens), "answer_choices": describe(choices),
                         "prompt_tokens": describe(token_lengths) if tokenizer else "pending transmitter tokenizer"}
    prefix = Path(args.output_prefix); prefix.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(prefix.with_suffix(".csv"), index=False)
    prefix.with_suffix(".json").write_text(json.dumps({"summary": records, "distributions": details}, indent=2), encoding="utf-8")
    print(f"Wrote {prefix.with_suffix('.csv')} and {prefix.with_suffix('.json')}")


if __name__ == "__main__": main()
