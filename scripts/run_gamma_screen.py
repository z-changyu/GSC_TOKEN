#!/usr/bin/env python
"""Objective adaptation for a restoration-capable receiver (gamma screen).

Measured problem (docs/v6主结果与效率对比.md §6): with the v6 receiver the
scheduler's linear objective sum u_i q_l over-values per-token reliability.
The receiver repairs erased fragments, so coverage breadth matters more than
the survival probability of any single token; Adaptive-SC (more tokens, mixed
protection) consequently overtakes proposed in the 0 dB cells even though both
deliver the same number of tokens.

Fix under test: replace q_l by q_l ** gamma in the selection objective.
gamma = 1 recovers the current scheduler, gamma < 1 flattens the level ranking
(cheaper levels become relatively more attractive -> broader coverage), and
gamma = 0 is the pure coverage-first extreme.

Validity: sample ids are split by parity into a dev half and a test half.
gamma is selected on dev only; the test half is reported for the chosen gamma
so the number is not tuned on the reported set.

Answering uses the v6 protocol (task intent + damage-triggered restoration +
letter-only output) so numbers are comparable with the v6 main table.
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-cache", default="outputs/cache/v2xqa_scores_8b.jsonl")
    parser.add_argument("--processed", default="data/processed/v2xqa_test.jsonl")
    parser.add_argument("--manifest", default="data/manifests/v2xqa_test_filtered_8b.jsonl")
    parser.add_argument("--receiver", default="models/Qwen3-4B")
    parser.add_argument("--output-dir", default="outputs/gamma_screen")
    parser.add_argument("--channels", nargs="*",
                        default=["configs/channels/rayleigh.yaml",
                                 "configs/channels/rician.yaml",
                                 "configs/channels/frequency_selective.yaml"])
    parser.add_argument("--snr-db", type=float, nargs="*", default=[0.0, 15.0])
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--gammas", type=float, nargs="*", default=[0.0, 0.25, 0.5, 1.0])
    parser.add_argument("--weights", type=float, nargs=3, default=[0.25, 0.30, 0.45])
    parser.add_argument("--restore-threshold", type=float, default=0.6)
    parser.add_argument("--limit", type=int)
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
    logger = configure_logging(out_dir / "gamma_screen.log")

    LETTER_ONLY = "Reply with only the option letter (A, B, C or D), with no other text."
    codebook = ProtectionCodebook()
    levels = torch.tensor(codebook.levels, dtype=torch.float32)
    bandwidth, slot_duration, eps_out, beta = 1e6, 1e-3, 0.05, 0.25

    cache = {str(r["sample_id"]): SampleScores.from_dict(r)
             for r in read_jsonl(args.score_cache)}
    processed = {str(r["row_idx"]): r for r in read_jsonl(args.processed)}
    manifest = [str(e["sample_id"]) for e in read_jsonl(args.manifest)]
    manifest = [s for s in manifest if s in cache and processed.get(s, {}).get("image")]
    if args.limit:
        manifest = manifest[:args.limit]
    gammas = [float(g) for g in args.gammas]
    methods = [f"g{g:g}" for g in gammas] + ["adaptive_sc"]
    logger.info("gamma screen: %d samples, gammas=%s, cells=%d",
                len(manifest), gammas, len(args.channels) * len(args.snr_db) * len(args.seeds))

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
                    for mi, method in enumerate(methods):
                        rng = random.Random((seed * 1000003 + position) * 131 + mi)
                        if method == "adaptive_sc":
                            level = select_level(method, utility,
                                                 torch.zeros(n_tokens, n_tokens), beta, q,
                                                 levels, budget, 8, rng)
                        else:
                            gamma = float(method[1:])
                            q_obj = torch.ones_like(q) if gamma == 0.0 else q ** gamma
                            level = select_level("proposed", utility,
                                                 torch.zeros(n_tokens, n_tokens), beta, q_obj,
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
                        from transformers import AutoProcessor
                        records.append({
                            "sample_id": sid, "channel": channel.name, "snr_db": float(snr_db),
                            "seed": int(seed), "method": method, "num_tokens": n_tokens,
                            "selected_tokens": len(selected), "received_tokens": len(kept),
                            "TRR": token_retention_ratio(len(selected), n_tokens),
                            "transmitted_symbols": sum(float(levels[int(level[i]) - 1])
                                                       for i in selected),
                            "budget": budget,
                            "token_erasure_rate": lost / len(selected) if selected else 0.0,
                            "token_ids": [scores.token_ids[i] for i in kept]})

    tokenizer = AutoTokenizer.from_pretrained(args.receiver, local_files_only=True)
    from src.models.registry import load_generation_receiver
    model = load_generation_receiver(args.receiver)
    transmitter_tokenizer = AutoProcessor.from_pretrained(
        "models/Qwen3-VL-8B-Instruct", local_files_only=True).tokenizer

    def ask(chat):
        inputs = tokenizer(chat, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=220, do_sample=False)
        return tokenizer.decode(out[0, inputs.input_ids.shape[1]:],
                                skip_special_tokens=True).strip()

    start = time.perf_counter()
    for n, rec in enumerate(records):
        row = processed[rec["sample_id"]]
        received = transmitter_tokenizer.decode(rec["token_ids"], skip_special_tokens=True)
        question = (f"Question: {row['question']}\n\nCandidate choices:\n"
                    f"{build_choice_block(row['choices'])}")
        if rec["received_tokens"] / max(1, rec["num_tokens"]) < args.restore_threshold:
            restore_chat = tokenizer.apply_chat_template(
                [{"role": "user", "content":
                  f"{question}\n\nDamaged description:\n{received}\n\n"
                  "The semantic description below was partially destroyed by a noisy "
                  "radio link. Using the question and the candidate choices, rewrite the "
                  "most plausible complete description of the scene. Keep the same wording "
                  "wherever it survives; fill gaps with the most likely continuation. "
                  "Output the restored description only."}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            received = ask(restore_chat)
        answer_chat = tokenizer.apply_chat_template(
            [{"role": "user", "content":
              f"{question}\n\nA text-only coordinator sends the following semantic "
              f"description of the scene:\n{received}\n\n{ANSWER_REQUEST}\n\n{LETTER_ONLY}"}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        raw = ask(answer_chat)
        prediction, parse_mode = canonical_prediction(raw, row["choices"])
        rec["prediction"] = prediction
        rec["parse_mode"] = parse_mode
        rec["correct"] = bool(exact_match(prediction, str(row["choices"][row["answer"]])))
        if (n + 1) % 400 == 0:
            logger.info("  answered %d/%d [%.1f min]", n + 1, len(records),
                        (time.perf_counter() - start) / 60)
    model = None
    torch.cuda.empty_cache()

    def fold(sid):
        try:
            return "dev" if int(sid) % 2 == 0 else "test"
        except ValueError:
            return "test"

    def acc(rows):
        return sum(r["correct"] for r in rows) / len(rows) if rows else float("nan")

    summary = []
    for method in methods:
        for channel in [Path(c).stem for c in args.channels]:
            for snr_db in args.snr_db:
                rows = [r for r in records if r["method"] == method
                        and r["channel"] == channel and r["snr_db"] == snr_db]
                if not rows:
                    continue
                summary.append({
                    "method": method, "channel": channel, "snr_db": float(snr_db),
                    "n": len(rows), "accuracy": acc(rows),
                    "accuracy_dev": acc([r for r in rows if fold(r["sample_id"]) == "dev"]),
                    "accuracy_test": acc([r for r in rows if fold(r["sample_id"]) == "test"]),
                    "mean_symbols": sum(r["transmitted_symbols"] for r in rows) / len(rows),
                    "mean_trr": sum(r["TRR"] for r in rows) / len(rows),
                    "raw_parse_rate": sum(r["parse_mode"] == "raw" for r in rows) / len(rows)})
    atomic_json(out_dir / "summary.json", {"records": records, "summary": summary})

    logger.info("%-12s %-24s %7s %7s %7s %8s", "method", "cell", "acc", "dev", "test", "symbols")
    for s in summary:
        logger.info("%-12s %-24s %6.1f%% %6.1f%% %6.1f%% %8.0f", s["method"],
                    f"{s['channel']}/{s['snr_db']:g}", s["accuracy"] * 100,
                    s["accuracy_dev"] * 100, s["accuracy_test"] * 100, s["mean_symbols"])
    logger.info("--- 6-cell average per method ---")
    for method in methods:
        rows = [s for s in summary if s["method"] == method]
        logger.info("%-12s all=%.1f%% dev=%.1f%% test=%.1f%%",
                    method, sum(s["accuracy"] for s in rows) / len(rows) * 100,
                    sum(s["accuracy_dev"] for s in rows) / len(rows) * 100,
                    sum(s["accuracy_test"] for s in rows) / len(rows) * 100)


if __name__ == "__main__":
    main()
