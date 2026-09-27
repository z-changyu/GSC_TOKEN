#!/usr/bin/env python
"""Receiver-protocol screening from an existing channel_results.csv.

The main experiment stores, per run, the exact text the receiver saw
(``selected_text``, post-erasure). We can therefore re-answer the same runs
under different answer prompts without touching selection, scheduling or the
channel -- a cheap way to screen protocol variants.

Variants
--------
v0  current protocol          : ANSWER_REQUEST + description
v1  task intent at receiver   : question + choices given, description appended
v2  v1 + letter-only output   : forces a parseable answer
v3  v0 + letter-only output

Justification: in the architecture the task intent (question + choices) is
issued by the RSU/cloud and known to both ends (feedback loop F), so v1/v2 do
not consume uplink budget. Every method is re-answered under every variant, so
the comparison stays homogeneous.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LETTER_ONLY = ("Reply with only the option letter (A, B, C or D), "
               "with no other text.")

RESTORE_REQUEST = (
    "The semantic description below was partially destroyed by a noisy radio "
    "link. Using the question and the candidate choices, rewrite the most "
    "plausible complete description of the scene. Keep the same wording "
    "wherever it survives; fill gaps with the most likely continuation. Output "
    "the restored description only.")


def build_prompt(variant: str, row, received: str, choices, blocks) -> str:
    from src.experiments.prompts import ANSWER_REQUEST

    if variant == "v0":
        return f"{ANSWER_REQUEST}\n\n{received}"
    if variant == "v3":
        return f"{ANSWER_REQUEST}\n\n{received}\n\n{LETTER_ONLY}"
    question = f"Question: {row['question']}\n\nCandidate choices:\n{blocks}"
    if variant == "v1":
        return (f"{question}\n\nA text-only coordinator sends the following "
                f"semantic description of the scene (some words may be missing):\n"
                f"{received}\n\n{ANSWER_REQUEST}")
    if variant == "v2":
        return (f"{question}\n\nA text-only coordinator sends the following "
                f"semantic description of the scene (some words may be missing):\n"
                f"{received}\n\n{ANSWER_REQUEST}\n\n{LETTER_ONLY}")
    if variant == "v7":
        # Validity control: question and choices only, no description at all.
        # It bounds how much of the accuracy comes from the receiver's prior
        # rather than from the transmitted content.
        return f"{question}\n\n{ANSWER_REQUEST}\n\n{LETTER_ONLY}"
    if variant == "v4_restore":
        # Pass 1 prompt: the receiver first reconstructs the damaged report.
        return f"{question}\n\nDamaged description:\n{received}\n\n{RESTORE_REQUEST}"
    if variant == "v4":
        # Pass 2 is built after restoration; placeholder handled by the caller.
        return (f"{question}\n\nA text-only coordinator sends the following "
                f"semantic description of the scene:\n{received}\n\n"
                f"{ANSWER_REQUEST}\n\n{LETTER_ONLY}")
    raise SystemExit(f"unknown variant {variant}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="outputs/channel_8b_final14/channel_results.csv")
    parser.add_argument("--processed", default="data/processed/v2xqa_test.jsonl")
    parser.add_argument("--output-dir", default="outputs/protocol_screen")
    parser.add_argument("--methods", nargs="*", default=["proposed", "adaptive_sc", "full"])
    parser.add_argument("--channels", nargs="*", default=["rayleigh"])
    parser.add_argument("--snr-db", type=float, nargs="*", default=[0.0, 15.0])
    parser.add_argument("--seeds", type=int, nargs="*", default=[0])
    parser.add_argument("--variants", nargs="*", default=["v0", "v1", "v2", "v3"])
    parser.add_argument("--receiver", default="models/Qwen3-4B")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.experiments.prompts import build_choice_block, canonical_prediction
    from src.metrics.qa import exact_match
    from src.utils.io import atomic_json, read_jsonl
    from src.utils.logging import configure_logging

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(out_dir / "screen.log")

    processed = {str(r["row_idx"]): r for r in read_jsonl(args.processed)}
    df = pd.read_csv(args.csv)
    df = df[(df.method.isin(args.methods)) & (df.channel.isin(args.channels))
            & (df.snr_db.isin(args.snr_db)) & (df.channel_seed.isin(args.seeds))]
    if args.limit:
        keep = sorted(df.sample_id.unique())[:args.limit]
        df = df[df.sample_id.isin(keep)]
    logger.info("screening %d runs: methods=%s cells=%s variants=%s",
                len(df), args.methods, [(c, s) for c in args.channels for s in args.snr_db],
                args.variants)

    tokenizer = AutoTokenizer.from_pretrained(args.receiver, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.receiver, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="eager").to("cuda:0").eval()

    results = defaultdict(list)
    start = time.perf_counter()
    for n, (_, row) in enumerate(df.iterrows()):
        sample = processed[str(row.sample_id)]
        choices = sample["choices"]
        blocks = build_choice_block(choices)
        received = "" if pd.isna(row.selected_text) else str(row.selected_text)
        ground_truth = str(choices[sample["answer"]])
        for variant in args.variants:
            if variant in ("v4", "v5", "v6"):
                # Damage-triggered restoration: the payload header carries the
                # original token count, so the receiver knows what fraction
                # arrived. Above the threshold the report is intact enough that
                # rewriting it would only corrupt good content (measured at
                # 15 dB), so the receiver answers directly instead.
                frac = float(row.received_tokens) / max(1.0, float(row.original_tokens))
                if variant in ("v5", "v6") and frac >= 0.6:
                    content = build_prompt("v2" if variant == "v6" else "v3",
                                           sample, received, choices, blocks)
                else:
                    restore = tokenizer.apply_chat_template(
                        [{"role": "user", "content": build_prompt("v4_restore", sample,
                                                                  received, choices, blocks)}],
                        tokenize=False, add_generation_prompt=True, enable_thinking=False)
                    inputs = tokenizer(restore, return_tensors="pt").to("cuda:0")
                    with torch.no_grad():
                        out = model.generate(**inputs, max_new_tokens=220, do_sample=False)
                    restored = tokenizer.decode(out[0, inputs.input_ids.shape[1]:],
                                                skip_special_tokens=True).strip()
                    content = build_prompt("v4", sample, restored, choices, blocks)
            else:
                content = build_prompt(variant, sample, received, choices, blocks)
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=24, do_sample=False)
            raw = tokenizer.decode(out[0, inputs.input_ids.shape[1]:],
                                   skip_special_tokens=True).strip()
            prediction, parse_mode = canonical_prediction(raw, choices)
            results[(row.method, row.channel, float(row.snr_db), variant)].append({
                "sample_id": row.sample_id, "seed": int(row.channel_seed),
                "correct": bool(exact_match(prediction, ground_truth)),
                "parse_mode": parse_mode, "raw_output": raw[:200]})
        if (n + 1) % 50 == 0:
            logger.info("  %d/%d runs [%.1f min]", n + 1, len(df),
                        (time.perf_counter() - start) / 60)
    model = None
    torch.cuda.empty_cache()

    # baseline accuracy of the same runs, straight from the CSV
    csv_base = df.groupby(["method", "channel", "snr_db"]).correct.mean()

    summary = []
    for (method, channel, snr_db, variant), rows in sorted(results.items()):
        acc = sum(r["correct"] for r in rows) / len(rows)
        raw_rate = sum(r["parse_mode"] == "raw" for r in rows) / len(rows)
        summary.append({"method": method, "channel": channel, "snr_db": float(snr_db),
                        "variant": variant, "n": len(rows), "accuracy": acc,
                        "raw_parse_rate": raw_rate,
                        "baseline_accuracy": float(csv_base[(method, channel, snr_db)])})
    atomic_json(out_dir / "summary.json", {"summary": summary})

    logger.info("%-16s %-10s %5s  %s", "method", "cell", "base", " ".join(
        f"{v:>7s}" for v in args.variants))
    for method in args.methods:
        for channel in args.channels:
            for snr_db in args.snr_db:
                base = csv_base.get((method, channel, snr_db))
                cells = []
                for variant in args.variants:
                    rows = results.get((method, channel, float(snr_db), variant))
                    cells.append(f"{(sum(r['correct'] for r in rows) / len(rows)) * 100:6.1f}%"
                                 if rows else "   n/a")
                logger.info("%-16s %-10s %5.1f%%  %s", method,
                            f"{channel}/{snr_db:g}", base * 100, " ".join(cells))


if __name__ == "__main__":
    main()
