#!/usr/bin/env python
"""Span-level selection screen: contiguous clauses instead of scattered tokens.

Motivation (measured): at 0 dB the reliable budget only carries ~24 tokens of
the report. Token-level optimal selection spreads them across the whole report,
so the receiver sees fragments such as ``" the theA..B. It clar.. It, the,,,,"``
-- individually high-utility tokens, jointly unreadable. A receiver cannot
extract evidence from that, and accuracy sits near the four-way guessing rate.

Span selection groups the report into clauses (split on sentence punctuation,
capped length) and protects whole spans, so every delivered span stays
readable. The knapsack is solved greedily by value density over (span, level)
pairs, and the per-token erasure physics is unchanged, so the comparison with
the token-level scheduler and Adaptive-SC is like-for-like under the same
reliable budget.

The receiver uses the task-intent protocol (question + choices are known to
both ends as the RSU-issued query) with a letter-only answer format.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BREAK_CHARS = (".", "?", "!", ";", ":")


def spans_from(tokens, max_len: int = 12):
    """Clause-level spans: break after sentence punctuation or at ``max_len``."""
    spans, cur = [], []
    for i, tok in enumerate(tokens):
        cur.append(i)
        text = tok.replace("Ġ", " ").replace("Ċ", "\n").replace("▁", " ")
        if (any(text.rstrip().endswith(b) for b in BREAK_CHARS)
                or "\n" in text or len(cur) >= max_len):
            spans.append(cur)
            cur = []
    if cur:
        spans.append(cur)
    return spans


def span_knapsack(spans, utility, q, levels, budget: float):
    """Greedy value-density knapsack over whole spans (level 0 = not sent)."""
    import torch

    level = torch.zeros(utility.numel(), dtype=torch.long)
    items = []
    for span in spans:
        total = float(utility[span].sum())
        length = len(span)
        best = None
        for li, cost_per_token in enumerate(levels):
            cost = length * float(cost_per_token)
            if cost > budget:
                continue
            value = total * float(q[li])
            density = value / cost
            if best is None or density > best[0]:
                best = (density, li, cost)
        if best is not None:
            items.append((best[0], span, best[1], best[2]))
    items.sort(key=lambda t: -t[0])
    used = 0.0
    for _density, span, li, cost in items:
        if used + cost <= budget:
            level[span] = li + 1
            used += cost
    return level, used


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-cache", default="outputs/cache/v2xqa_scores_8b.jsonl")
    parser.add_argument("--processed", default="data/processed/v2xqa_test.jsonl")
    parser.add_argument("--manifest", default="data/manifests/v2xqa_test_filtered_8b.jsonl")
    parser.add_argument("--transmitter", default="models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--receiver", default="models/Qwen3-4B")
    parser.add_argument("--output-dir", default="outputs/span_screen")
    parser.add_argument("--channels", nargs="*", default=["configs/channels/rayleigh.yaml"])
    parser.add_argument("--snr-db", type=float, nargs="*", default=[0.0, 15.0])
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--weights", type=float, nargs=3, default=[0.25, 0.30, 0.45])
    parser.add_argument("--span-max-len", type=int, default=12)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--restore-threshold", type=float, default=0.6,
                        help="run the restoration pass when received/original falls below this")
    args = parser.parse_args()

    import torch
    import yaml
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    from src.channel import build_channel, decoding_probability
    from src.channel.link import ProtectionCodebook
    from src.channel.reliable_capacity import reliable_spectral_efficiency
    from src.experiments.prompts import (ANSWER_REQUEST, build_choice_block,
                                         canonical_prediction)
    from src.experiments.scoring import SampleScores, combine_scores
    from src.metrics.communication import token_retention_ratio
    from src.metrics.qa import exact_match
    from src.scheduling import select_level
    from src.utils.io import atomic_json, read_jsonl
    from src.utils.logging import configure_logging

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(out_dir / "span_screen.log")

    LETTER_ONLY = ("Reply with only the option letter (A, B, C or D), "
                   "with no other text.")
    codebook = ProtectionCodebook()
    levels = torch.tensor(codebook.levels, dtype=torch.float32)
    bandwidth, slot_duration, eps_out = 1e6, 1e-3, 0.05
    beta = 0.25

    cache = {str(r["sample_id"]): SampleScores.from_dict(r)
             for r in read_jsonl(args.score_cache)}
    processed = {str(r["row_idx"]): r for r in read_jsonl(args.processed)}
    manifest = [str(e["sample_id"]) for e in read_jsonl(args.manifest)]
    manifest = [s for s in manifest if s in cache and processed.get(s, {}).get("image")]
    if args.limit:
        manifest = manifest[:args.limit]
    logger.info("span screen: %d samples, channels=%s snr=%s seeds=%s",
                len(manifest), args.channels, args.snr_db, args.seeds)

    transmitter_tokenizer = AutoProcessor.from_pretrained(
        args.transmitter, local_files_only=True).tokenizer

    methods = ["proposed", "proposed_span", "adaptive_sc"]
    records = []
    for channel_path in args.channels:
        spec = yaml.safe_load(Path(channel_path).read_text())
        for snr_db in args.snr_db:
            for seed in args.seeds:
                channel = build_channel({**spec, "snr_db": float(snr_db), "seed": int(seed)})
                logger.info("cell %s/%s seed=%d", channel.name, snr_db, seed)
                for position, sid in enumerate(manifest):
                    scores = cache[sid]
                    row = processed[sid]
                    utility = combine_scores(scores.attention, scores.semantic,
                                             scores.gradient, args.weights).float()
                    n_tokens = scores.num_tokens
                    coupling = torch.zeros(n_tokens, n_tokens)
                    state = channel.sample((1,))
                    c_eps = reliable_spectral_efficiency(state, eps_out)
                    budget = slot_duration * bandwidth * float(c_eps[0])
                    realized = math.log2(1.0 + float(state.snr_linear)
                                         * float(state.h.abs().pow(2)[0]))
                    q = decoding_probability(state, codebook).reshape(-1)
                    spans = spans_from(scores.token_strings, args.span_max_len)
                    for mi, method in enumerate(methods):
                        rng = random.Random((seed * 1000003 + position) * 131 + mi)
                        if method == "proposed_span":
                            level, _used = span_knapsack(spans, utility, q, levels, budget)
                        else:
                            level = select_level(method, utility, coupling, beta, q,
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
                        received = transmitter_tokenizer.decode(
                            [scores.token_ids[i] for i in kept], skip_special_tokens=True)
                        symbols = sum(float(levels[int(level[i]) - 1]) for i in selected)
                        records.append({
                            "sample_id": sid, "channel": channel.name, "snr_db": float(snr_db),
                            "seed": int(seed), "method": method,
                            "num_tokens": n_tokens, "selected_tokens": len(selected),
                            "received_tokens": len(kept), "TRR": token_retention_ratio(len(selected), n_tokens),
                            "transmitted_symbols": symbols, "budget": budget,
                            "token_erasure_rate": lost / len(selected) if selected else 0.0,
                            "received_text": received})
                logger.info("  done: %s/%s/%d", channel.name, snr_db, seed)

    # ---------------------------------------------------------------- answer
    tokenizer = AutoTokenizer.from_pretrained(args.receiver, local_files_only=True)
    from src.models.registry import load_generation_receiver
    model = load_generation_receiver(args.receiver)
    start = time.perf_counter()
    for n, rec in enumerate(records):
        row = processed[rec["sample_id"]]
        blocks = build_choice_block(row["choices"])
        question = f"Question: {row['question']}\n\nCandidate choices:\n{blocks}"
        received = rec["received_text"]
        frac = rec["received_tokens"] / max(1, rec["num_tokens"])
        if frac < args.restore_threshold:
            restore_prompt = (
                f"{question}\n\nDamaged description:\n{received}\n\n"
                "The semantic description below was partially destroyed by a noisy "
                "radio link. Using the question and the candidate choices, rewrite "
                "the most plausible complete description of the scene. Keep the "
                "same wording wherever it survives; fill gaps with the most likely "
                "continuation. Output the restored description only.")
            chat = tokenizer.apply_chat_template([{"role": "user", "content": restore_prompt}],
                                                 tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=False)
            inputs = tokenizer(chat, return_tensors="pt").to("cuda:0")
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
            received = tokenizer.decode(out[0, inputs.input_ids.shape[1]:],
                                        skip_special_tokens=True).strip()
            rec["restored"] = True
        prompt = (f"{question}\n\nA text-only coordinator sends the following semantic "
                  f"description of the scene:\n{received}\n\n"
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
    logger.info("%-15s %-14s %6s %8s %8s", "method", "cell", "acc", "symbols", "rcvdTok")
    for s in summary:
        logger.info("%-15s %-14s %5.1f%% %8.0f %8.1f", s["method"],
                    f"{s['channel']}/{s['snr_db']:g}", s["accuracy"] * 100,
                    s["mean_symbols"], s["mean_received_tokens"])


if __name__ == "__main__":
    main()
