#!/usr/bin/env python
"""Budget-aware semantic report generation (channel-aware transmitter).

Measured problem: at 0 dB only ~30 of the ~110 report tokens can be delivered,
i.e. 27% coverage. The reason is not which tokens are chosen but that the report
was written without any budget knowledge: it spends ~45% of its tokens
restating the question and listing the four choices -- redundant information,
because in the architecture the task intent is issued by the RSU/cloud and is
already known to the receiving coordinator (feedback loop F).

This script generates a second, budget-aware report per sample:
  * question and choices are NOT restated (the receiver already has them);
  * only discriminative visual evidence, one short clause per choice;
  * hard cap of ~40 tokens, so a 0 dB reliable budget can carry most of it.

The compact report is then scored with the same pragmatic scorer, scheduled
under the same reliable budget (proposed / proposed_span / adaptive_sc), put
through the identical per-token erasure physics, and answered under the
task-intent + letter-only protocol. Numbers are directly comparable with
scripts/run_span_screen.py (full-report baseline).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_span_screen import span_knapsack, spans_from  # noqa: E402

COMPACT_REQUEST = (
    "Question: {question}\n\nCandidate choices:\n{choices}\n\n"
    "The receiving coordinator ALREADY KNOWS the question and every candidate "
    "choice above, so do NOT restate them and do NOT list them again.\n"
    "Write exactly four lines, one per candidate choice, in this exact form:\n"
    "(A) <the single most decisive visible detail, or \"no visible evidence\">\n"
    "(B) <same>\n(C) <same>\n(D) <same>\n"
    "Keep each line under 12 tokens. Do not repeat the choice wording. Quote "
    "only detail actually present in the image. Do not say which choice is "
    "correct. Output the four lines only, with nothing before or after them.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", default="data/processed/v2xqa_test.jsonl")
    parser.add_argument("--manifest", default="data/manifests/v2xqa_test_filtered_8b.jsonl")
    parser.add_argument("--transmitter", default="models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--receiver", default="models/Qwen3-4B")
    parser.add_argument("--output-dir", default="outputs/budget_aware")
    parser.add_argument("--channels", nargs="*", default=["configs/channels/rayleigh.yaml"])
    parser.add_argument("--snr-db", type=float, nargs="*", default=[0.0, 15.0])
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--weights", type=float, nargs=3, default=[0.25, 0.30, 0.45])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--reuse-scores", action="store_true",
                        help="reuse an existing compact score cache")
    args = parser.parse_args()

    import torch
    import yaml
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
    from transformers import AutoProcessor, AutoTokenizer

    from src.channel import build_channel, decoding_probability
    from src.channel.link import ProtectionCodebook
    from src.channel.reliable_capacity import reliable_spectral_efficiency
    from src.experiments.prompts import (ANSWER_REQUEST, build_choice_block,
                                         canonical_prediction)
    from src.experiments.scoring import PragmaticScorer, SampleScores, combine_scores, load_image
    from src.metrics.communication import token_retention_ratio
    from src.metrics.qa import exact_match
    from src.scheduling import select_level
    from src.utils.io import atomic_json, read_jsonl
    from src.utils.logging import configure_logging

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(out_dir / "budget_aware.log")
    score_path = out_dir / "compact_scores.jsonl"

    LETTER_ONLY = ("Reply with only the option letter (A, B, C or D), "
                   "with no other text.")
    codebook = ProtectionCodebook()
    levels = torch.tensor(codebook.levels, dtype=torch.float32)
    bandwidth, slot_duration, eps_out, beta = 1e6, 1e-3, 0.05, 0.25

    processed = {str(r["row_idx"]): r for r in read_jsonl(args.processed)}
    manifest = [str(e["sample_id"]) for e in read_jsonl(args.manifest)]
    manifest = [s for s in manifest if processed.get(s, {}).get("image")][:args.limit]
    logger.info("budget-aware reports: %d samples", len(manifest))

    # ------------------------------------------------------------------ score
    cache = {}
    if args.reuse_scores and score_path.exists():
        cache = {str(r["sample_id"]): SampleScores.from_dict(r)
                 for r in read_jsonl(score_path)}
        logger.info("reused %d compact scores", len(cache))
    if len(cache) < len(manifest):
        config = {"transmitter": {"max_prompt_tokens": 96,
                                  "max_new_tokens": args.max_new_tokens},
                  "pragmatic": {"beta": beta, "layer_reduction": "mean",
                                "coupling_enabled": False,
                                "mask_token": "<|endoftext|>"}}
        processor = AutoProcessor.from_pretrained(
            args.transmitter, local_files_only=True, min_pixels=25088, max_pixels=200704)
        transmitter = AutoModelForImageTextToText.from_pretrained(
            args.transmitter, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="eager").to("cuda:0").eval()
        scorer = PragmaticScorer(transmitter, processor, config)
        written = []
        for sid in manifest:
            row = processed[sid]
            image = load_image(row["image"])
            request = COMPACT_REQUEST.format(question=row["question"],
                                             choices=build_choice_block(row["choices"]))
            sample = dict(row)
            sample["_request"] = request
            scores = scorer.score(sample, image)
            written.append(scores.to_dict())
            logger.info("  [gen] %s compact=%d tokens", sid, scores.num_tokens)
        with open(score_path, "w", encoding="utf-8") as handle:
            for row in written:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        cache = {str(r["sample_id"]): SampleScores.from_dict(r) for r in written}
        scorer = transmitter = processor = None
        torch.cuda.empty_cache()

    # ------------------------------------------------------- select + erasure
    methods = ["full", "proposed", "proposed_span", "adaptive_sc"]
    compact_tokenizer = AutoProcessor.from_pretrained(
        args.transmitter, local_files_only=True).tokenizer
    records = []
    for channel_path in args.channels:
        spec = yaml.safe_load(Path(channel_path).read_text())
        for snr_db in args.snr_db:
            for seed in args.seeds:
                channel = build_channel({**spec, "snr_db": float(snr_db), "seed": int(seed)})
                logger.info("cell %s/%s seed=%d", channel.name, snr_db, seed)
                for position, sid in enumerate(manifest):
                    scores = cache[sid]
                    utility = combine_scores(scores.attention, scores.semantic,
                                             scores.gradient, args.weights).float()
                    n_tokens = scores.num_tokens
                    state = channel.sample((1,))
                    c_eps = reliable_spectral_efficiency(state, eps_out)
                    budget = slot_duration * bandwidth * float(c_eps[0])
                    realized = math.log2(1.0 + float(state.snr_linear)
                                         * float(state.h.abs().pow(2)[0]))
                    q = decoding_probability(state, codebook).reshape(-1)
                    spans = spans_from(scores.token_strings)
                    for mi, method in enumerate(methods):
                        rng = random.Random((seed * 1000003 + position) * 131 + mi)
                        if method == "proposed_span":
                            level, _ = span_knapsack(spans, utility, q, levels, budget)
                        else:
                            level = select_level(method, utility,
                                                 torch.zeros(n_tokens, n_tokens), beta, q,
                                                 levels, budget, 8, rng)
                        selected = [i for i in range(n_tokens) if level[i] > 0]
                        kept, lost = [], 0
                        for token in selected:
                            rate = float(codebook.code_rates[int(level[token]) - 1])
                            dropped = rng.random() < float(state.packet_loss_rate)
                            if realized >= rate and not dropped:
                                kept.append(token)
                            else:
                                lost += 1
                        received = compact_tokenizer.decode(
                            [scores.token_ids[i] for i in kept], skip_special_tokens=True)
                        symbols = sum(float(levels[int(level[i]) - 1]) for i in selected)
                        records.append({
                            "sample_id": sid, "channel": channel.name, "snr_db": float(snr_db),
                            "seed": int(seed), "method": method, "num_tokens": n_tokens,
                            "selected_tokens": len(selected), "received_tokens": len(kept),
                            "TRR": token_retention_ratio(len(selected), n_tokens),
                            "transmitted_symbols": symbols, "budget": budget,
                            "token_erasure_rate": lost / len(selected) if selected else 0.0,
                            "received_text": received})

    # ---------------------------------------------------------------- answer
    tokenizer = AutoTokenizer.from_pretrained(args.receiver, local_files_only=True)
    from src.models.registry import load_generation_receiver
    model = load_generation_receiver(args.receiver)
    start = time.perf_counter()
    for n, rec in enumerate(records):
        row = processed[rec["sample_id"]]
        blocks = build_choice_block(row["choices"])
        prompt = (f"Question: {row['question']}\n\nCandidate choices:\n{blocks}\n\n"
                  f"A text-only coordinator sends the following semantic description "
                  f"of the scene (some words may be missing):\n{rec['received_text']}\n\n"
                  f"{ANSWER_REQUEST}\n\n{LETTER_ONLY}")
        chat = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                             tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        inputs = tokenizer(chat, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        raw = tokenizer.decode(out[0, inputs.input_ids.shape[1]:],
                               skip_special_tokens=True).strip()
        prediction, parse_mode = canonical_prediction(raw, row["choices"])
        rec["prediction"] = prediction
        rec["parse_mode"] = parse_mode
        rec["correct"] = bool(exact_match(prediction, str(row["choices"][row["answer"]])))
        if (n + 1) % 200 == 0:
            logger.info("  answered %d/%d [%.1f min]", n + 1, len(records),
                        (time.perf_counter() - start) / 60)
    model = None
    torch.cuda.empty_cache()

    agg = defaultdict(list)
    for rec in records:
        agg[(rec["method"], rec["channel"], rec["snr_db"])].append(rec)
    summary = []
    for key, rows in sorted(agg.items()):
        summary.append({
            "method": key[0], "channel": key[1], "snr_db": key[2], "n": len(rows),
            "accuracy": sum(r["correct"] for r in rows) / len(rows),
            "mean_symbols": sum(r["transmitted_symbols"] for r in rows) / len(rows),
            "mean_trr": sum(r["TRR"] for r in rows) / len(rows),
            "mean_received_tokens": sum(r["received_tokens"] for r in rows) / len(rows),
            "raw_parse_rate": sum(r["parse_mode"] == "raw" for r in rows) / len(rows)})
    atomic_json(out_dir / "summary.json", {"samples": len(manifest), "records": records,
                                           "summary": summary})
    logger.info("%-15s %-14s %6s %8s %9s", "method", "cell", "acc", "symbols", "rcvdTok")
    for s in summary:
        logger.info("%-15s %-14s %5.1f%% %8.0f %9.1f", s["method"],
                    f"{s['channel']}/{s['snr_db']:g}", s["accuracy"] * 100,
                    s["mean_symbols"], s["mean_received_tokens"])


if __name__ == "__main__":
    main()
