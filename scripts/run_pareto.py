#!/usr/bin/env python
"""Cost-accuracy Pareto sweep for the proposed scheduler (iso-cost evidence).

Motivation
----------
The main comparison holds the *reliable budget* constant but lets every method
choose how much of it to spend, so the methods end up at different symbol
counts (proposed 1360 vs Adaptive-SC 1921 on the 6-cell average). Reviewers
reasonably ask whether the accuracy tie is an artefact of proposed stopping
early, and what happens at equal cost.

Two complementary cost axes are therefore built for proposed:

``budget{B}``
    The scheduler is given an artificial symbol budget ``B``. Below the natural
    spending point this binds and traces the left part of the curve; above it
    the optimum refuses to spend more, which is itself the evidence that extra
    symbols have no value (at 15 dB the decoding probabilities of the three
    protection levels are nearly equal, so extra spend buys no reliability).

``fill{T}``
    Equality-constrained operating points. Starting from the unconstrained
    solution, protection levels are upgraded greedily (by value gain per extra
    symbol) until the target spend ``T`` is reached. Upgrading only raises the
    decoding probability, so the delivered token set can only grow: the
    operation is monotone-safe. ``T`` is taken from the measured spend of the
    baselines (Adaptive-SC, the lexical family, the capacity oracle) so the
    comparison is exactly iso-cost.

``uniform{rL,rM,rH}``
    All tokens at one protection level: the two ends of the feasible cost axis.

All points respect the reliable budget of Eq. (6) -- the sweep is clamped to it
-- so neither ``full`` (uncompressed, ignores the budget) nor
``plugin_capacity`` (genie capacity) semantics leak in.

Answering reuses the v6 receiver protocol through
``src/experiments/receiver_protocol.py`` with the same generation budget as the
main experiment, so accuracies are directly comparable with
``outputs/channel_8b_final14_v6final``.

Usage
-----
    python scripts/run_pareto.py --limit 3 --seeds 0      # smoke
    python scripts/run_pareto.py                          # full 15 dB sweep
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LEXICAL_METHODS = ("llmlingua2", "llmlingua", "longllmlingua", "selective_context",
                   "efpc", "tokcom", "answer_loo")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-cache", default="outputs/cache/v2xqa_scores_8b.jsonl")
    parser.add_argument("--processed", default="data/processed/v2xqa_test.jsonl")
    parser.add_argument("--manifest", default="data/manifests/v2xqa_test_filtered_8b.jsonl")
    parser.add_argument("--baseline-csv",
                        default="outputs/channel_8b_final14_v6final/channel_results.csv")
    parser.add_argument("--receiver", default="models/Qwen3-4B")
    parser.add_argument("--output-dir", default="outputs/pareto")
    parser.add_argument("--channels", nargs="*",
                        default=["configs/channels/rayleigh.yaml",
                                 "configs/channels/rician.yaml",
                                 "configs/channels/frequency_selective.yaml"])
    parser.add_argument("--snr-db", type=float, nargs="*", default=[15.0])
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--budgets", type=float, nargs="*",
                        default=[800.0, 1200.0, 1600.0, 2000.0, 2600.0, 3200.0, 4000.0])
    parser.add_argument("--weights", type=float, nargs=3, default=[0.25, 0.30, 0.45])
    parser.add_argument("--t-max", type=int, default=8)
    parser.add_argument("--answer-max-new-tokens", type=int, default=24,
                        help="must match the main experiment (config receiver.max_new_tokens)")
    parser.add_argument("--restore-threshold", type=float, default=0.6)
    parser.add_argument("--receiver-protocol", choices=["v0", "v6", "v6_norestore"],
                        default="v6",
                        help="v6_norestore mirrors the fair main line: the v6 answer "
                             "prompt with damage-triggered restoration disabled")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--coupling-variant", default="raw", choices=["raw", "row", "A", "B"],
                        help="coupling definition; must match the main table being "
                             "compared against")
    parser.add_argument("--token-embeddings",
                        default="outputs/cache/v2xqa_token_embeddings_8b.pt")
    parser.add_argument("--lp-utility", default="marginal", choices=["marginal", "context"],
                        help="'context' solves the LP on the Eq. (19) set-aware utility "
                             "so the coupling steers selection, not just refinement")
    args = parser.parse_args()

    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.channel import build_channel, decoding_probability
    from src.channel.link import ProtectionCodebook
    from src.channel.reliable_capacity import reliable_spectral_efficiency
    from src.experiments.prompts import ANSWER_REQUEST
    from src.experiments.receiver_protocol import answer_with_protocol
    from src.experiments.scoring import SampleScores, combine_scores
    from src.metrics.communication import token_retention_ratio
    from src.metrics.qa import exact_match
    from src.pragmatic.coupling import build_coupling_variant
    from src.pragmatic.utility import context_aware_utility
    from src.scheduling import select_level
    from src.utils.io import atomic_json, read_jsonl
    from src.utils.logging import configure_logging

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(out_dir / "pareto.log")

    codebook = ProtectionCodebook()
    levels = torch.tensor(codebook.levels, dtype=torch.float32)
    bandwidth, slot_duration, eps_out, beta = 1e6, 1e-3, 0.05, float(args.beta)

    cache = {str(r["sample_id"]): SampleScores.from_dict(r)
             for r in read_jsonl(args.score_cache)}
    processed = {str(r["row_idx"]): r for r in read_jsonl(args.processed)}
    manifest = [str(e["sample_id"]) for e in read_jsonl(args.manifest)]
    manifest = [s for s in manifest if s in cache and processed.get(s, {}).get("image")]
    if args.limit:
        manifest = manifest[:args.limit]

    # -------------------------------------------------- baseline iso-cost targets
    frame = pd.read_csv(args.baseline_csv)
    frame = frame[frame.snr_db.isin(args.snr_db)] if len(args.snr_db) else frame
    targets: dict[str, float] = {}
    for name, members in (("adaptive", ["adaptive_sc"]),
                          ("oracle", ["plugin_capacity"]),
                          ("lexical", list(LEXICAL_METHODS))):
        spend = [float(frame[frame.method == m].transmitted_symbols.mean())
                 for m in members if (frame.method == m).any()]
        if spend:
            targets[name] = sum(spend) / len(spend)
    logger.info("iso-cost targets (mean symbols): %s",
                {k: round(v, 1) for k, v in targets.items()})
    logger.info("pareto sweep: %d samples, channels=%d, snr=%s, seeds=%s, %d budget points",
                len(manifest), len(args.channels), args.snr_db, args.seeds,
                len(args.budgets))

    token_embeddings = {}
    if args.coupling_variant != "raw":
        token_embeddings = {str(k): v for k, v in
                            torch.load(args.token_embeddings, weights_only=False).items()}
    logger.info("coupling variant=%s lp_utility=%s beta=%s", args.coupling_variant,
                args.lp_utility, beta)

    # -------------------------------------------------------------- strategies
    def proposed(utility, q, budget, rng, coupling, selection_utility=None):
        return select_level("proposed", utility, coupling, beta, q, levels,
                            budget, args.t_max, rng, selection_utility=selection_utility)

    def fill_to_target(level, utility, q, target, budget):
        """Upgrade protection levels until ``target`` symbols are spent.

        Each upgrade trades extra symbols for a weakly larger decoding
        probability, so the delivered set can only grow; the order is the value
        gain per extra symbol, which keeps the fill as efficient as possible.
        """
        level = level.clone()
        spent = sum(float(levels[int(level[i]) - 1]) for i in range(level.numel())
                    if level[i] > 0)
        while spent < target - 1e-9:
            best = None
            for i in range(level.numel()):
                if level[i] == 0 or level[i] == 1:      # unselected / already strongest
                    continue
                old_idx, new_idx = int(level[i]) - 1, int(level[i]) - 2
                delta_cost = float(levels[new_idx] - levels[old_idx])
                if spent + delta_cost > budget:
                    continue
                delta_value = float(utility[i]) * float(q[new_idx] - q[old_idx])
                density = delta_value / delta_cost if delta_cost > 0 else 0.0
                if best is None or density > best[0]:
                    best = (density, i, delta_cost)
            if best is None:
                break                                    # nothing upgradable within budget
            _density, index, delta_cost = best
            level[index] -= 1
            spent += delta_cost
        return level

    records: list[dict] = []
    skipped: set[str] = set()          # uniform levels that would break the contract
    answered: dict[tuple, dict] = {}   # identical selections are answered once
    answer_cache: dict[tuple, dict] = {}
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
                    coupling = (torch.tensor(scores.coupling, dtype=torch.float32)
                                if scores.coupling is not None
                                else torch.zeros(n_tokens, n_tokens))
                    if args.coupling_variant != "raw":
                        coupling = build_coupling_variant(args.coupling_variant, coupling,
                                                          token_embeddings[sid])
                    selection_utility = None
                    if args.lp_utility == "context":
                        selection_utility = context_aware_utility(utility, coupling, beta)
                    state = channel.sample((1,))
                    c_eps = reliable_spectral_efficiency(state, eps_out)
                    reliable = slot_duration * bandwidth * float(c_eps[0])
                    realized = math.log2(1.0 + float(state.snr_linear)
                                         * float(state.h.abs().pow(2)[0]))
                    q = decoding_probability(state, codebook).reshape(-1)

                    # same rng stream for every strategy -> deterministic and paired
                    rng_seed = int(seed) * 1000003 + position
                    solutions: dict[str, torch.Tensor] = {}
                    for budget in args.budgets:
                        capped = min(float(budget), reliable)
                        solutions[f"budget{int(budget)}"] = proposed(
                            utility, q, capped, random.Random(rng_seed), coupling,
                            selection_utility=selection_utility)
                    base = solutions[f"budget{int(max(args.budgets))}"]
                    for name, target in targets.items():
                        solutions[f"fill_{name}"] = fill_to_target(
                            base, utility, q, min(target, reliable), reliable)
                    for name, level_index in (("uniform_rL", 3), ("uniform_rM", 2),
                                              ("uniform_rH", 1)):
                        cost = float(levels[level_index - 1]) * n_tokens
                        if cost <= reliable:
                            solutions[name] = torch.full((n_tokens,), level_index,
                                                         dtype=torch.long)
                        else:
                            # Sending the whole report at this level would break the
                            # 5% outage contract; such a point is not admissible for
                            # a compliant method, so it is skipped rather than faked.
                            skipped.add(name)
                    solutions["anchor_adaptive"] = select_level(
                        "adaptive_sc", utility, coupling, beta, q, levels, reliable,
                        args.t_max, random.Random(rng_seed))

                    for name, level in solutions.items():
                        selected = [i for i in range(n_tokens) if level[i] > 0]
                        symbols = sum(float(levels[int(level[i]) - 1]) for i in selected)
                        # Certified rounding can overshoot by at most one strongest
                        # level. Measured on the main run: 0.19% of proposed runs,
                        # max 16 symbols (0.7%). Anything larger is a genuine
                        # contract violation and must fail loudly.
                        overshoot = symbols - reliable
                        if overshoot > float(levels[0]):
                            raise AssertionError(
                                f"{name} breaks the reliable budget by {overshoot:.1f} "
                                f"symbols ({symbols:.1f} > {reliable:.1f})")
                        rng = random.Random(rng_seed)
                        kept, lost = [], 0
                        for token in selected:
                            rate = float(codebook.code_rates[int(level[token]) - 1])
                            dropped = rng.random() < float(state.packet_loss_rate)
                            if realized >= rate and not dropped:
                                kept.append(token)
                            else:
                                lost += 1
                        counts = [0, 0, 0]
                        for token in selected:
                            counts[int(level[token]) - 1] += 1
                        records.append({
                            "sample_id": sid, "channel": channel.name,
                            "snr_db": float(snr_db), "seed": int(seed), "strategy": name,
                            "num_tokens": n_tokens, "selected_tokens": len(selected),
                            "received_tokens": len(kept),
                            "transmitted_symbols": symbols, "reliable_budget": reliable,
                            "budget_utilisation": symbols / reliable if reliable else 0.0,
                            "overshoot": overshoot,
                            "TRR": token_retention_ratio(len(selected), n_tokens),
                            "token_erasure_rate": lost / len(selected) if selected else 0.0,
                            "kept_rH": counts[0], "kept_rM": counts[1], "kept_rL": counts[2],
                            "realized_efficiency": realized,
                            "sig": hashlib.md5(bytes(int(v) for v in level.tolist())
                                               ).hexdigest()[:12],
                            "token_ids": [scores.token_ids[i] for i in kept]})
                logger.info("  selected: %s/%s/%d", channel.name, snr_db, seed)

    # ---------------------------------------------------------------- answering
    tokenizer = AutoTokenizer.from_pretrained(args.receiver, local_files_only=True)
    # Text-only receivers load as causal LMs; vision-language receivers (e.g.
    # Qwen3-VL-8B driven in text-only mode) need the image-text-to-text class,
    # mirroring the fallback in scripts/run_channel_experiment.py.
    load_kwargs = dict(local_files_only=True, attn_implementation="eager",
                       dtype=torch.bfloat16)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.receiver, **load_kwargs)
    except (ValueError, KeyError, RuntimeError, TypeError):
        gc.collect()
        torch.cuda.empty_cache()
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            args.receiver, **load_kwargs)
    model = model.to("cuda:0").eval()
    from transformers import AutoProcessor
    transmitter_tokenizer = AutoProcessor.from_pretrained(
        "models/Qwen3-VL-8B-Instruct", local_files_only=True).tokenizer

    start = time.perf_counter()
    reused = 0
    for n, rec in enumerate(records):
        row = processed[rec["sample_id"]]
        token_ids = rec.pop("token_ids")
        key = (rec["channel"], rec["snr_db"], rec["seed"], rec["sample_id"], rec["sig"])
        cached = answered.get(key)
        if cached is not None:
            # Same selection as an earlier strategy (the budget plateau): reuse
            # the receiver's verdict instead of generating it again.
            rec.update(cached)
            reused += 1
            continue
        received = transmitter_tokenizer.decode(token_ids, skip_special_tokens=True)
        answer = answer_with_protocol(
            model=model, tokenizer=tokenizer, row=row, received=received,
            survived_tokens=rec["received_tokens"], total_tokens=rec["num_tokens"],
            answer_request=ANSWER_REQUEST,
            max_new_tokens=args.answer_max_new_tokens,
            protocol=("v0" if args.receiver_protocol == "v0" else "v6"),
            restore_threshold=(0.0 if args.receiver_protocol == "v6_norestore"
                               else args.restore_threshold))
        verdict = {"restored": int(answer["restored"]), "parse_mode": answer["parse_mode"],
                   "prediction": answer["prediction"],
                   "correct": bool(exact_match(answer["prediction"],
                                               str(row["choices"][row["answer"]])))}
        answered[key] = verdict
        rec.update(verdict)
        if (n + 1) % 500 == 0:
            logger.info("  answered %d/%d [%.1f min] (reused %d)", n + 1, len(records),
                        (time.perf_counter() - start) / 60, reused)
    logger.info("answering done: %d generations, %d reused from identical selections",
                len(records) - reused, reused)
    if skipped:
        logger.info("skipped (would exceed the reliable budget): %s",
                    ", ".join(sorted(skipped)))
    # release in the caller scope: deleting a local name inside a helper cannot
    # drop the reference held by this frame (an earlier bug leaked 4 GiB).
    model = tokenizer = transmitter_tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()

    # ------------------------------------------------------------- aggregation
    def aggregate(rows: list[dict]) -> dict:
        return {
            "n": len(rows),
            "accuracy": sum(r["correct"] for r in rows) / len(rows),
            "mean_symbols": sum(r["transmitted_symbols"] for r in rows) / len(rows),
            "mean_budget_utilisation": sum(r["budget_utilisation"] for r in rows) / len(rows),
            "mean_trr": sum(r["TRR"] for r in rows) / len(rows),
            "mean_received_tokens": sum(r["received_tokens"] for r in rows) / len(rows),
            "mean_kept_rH": sum(r["kept_rH"] for r in rows) / len(rows),
            "mean_kept_rM": sum(r["kept_rM"] for r in rows) / len(rows),
            "mean_kept_rL": sum(r["kept_rL"] for r in rows) / len(rows),
            "restore_rate": sum(r["restored"] for r in rows) / len(rows),
            "raw_parse_rate": sum(r["parse_mode"] == "raw" for r in rows) / len(rows),
            "mean_overshoot": sum(r["overshoot"] for r in rows) / len(rows),
            "overshoot_rate": sum(1 for r in rows if r["overshoot"] > 1e-6) / len(rows)}

    by_strategy = defaultdict(list)
    by_cell = defaultdict(list)
    for rec in records:
        by_strategy[rec["strategy"]].append(rec)
        by_cell[(rec["strategy"], rec["channel"])].append(rec)

    summary = {
        "samples": len(manifest), "snr_db": [float(v) for v in args.snr_db],
        "seeds": [int(v) for v in args.seeds], "budgets": [float(v) for v in args.budgets],
        "iso_cost_targets": targets, "baseline_csv": args.baseline_csv,
        # Provenance of the answering receiver, so the sweep can be traced back
        # to the same edge model and protocol as the main run it is compared to.
        "receiver": args.receiver, "receiver_protocol": args.receiver_protocol,
        "restore_threshold": (0.0 if args.receiver_protocol == "v6_norestore"
                              else args.restore_threshold),
        "score_cache": args.score_cache, "channels": list(args.channels),
        "strategies": {name: aggregate(rows) for name, rows in sorted(by_strategy.items())},
        "per_channel": {f"{name}@{channel}": aggregate(rows)
                        for (name, channel), rows in sorted(by_cell.items())}}
    atomic_json(out_dir / "summary.json", summary)
    pd.DataFrame([{k: v for k, v in rec.items()} for rec in records]).to_csv(
        out_dir / "results.csv", index=False)

    logger.info("%-16s %7s %9s %9s %8s %8s", "strategy", "acc", "symbols", "budgetUse",
                "rcvdTok", "restore")
    for name, agg in sorted(summary["strategies"].items(),
                            key=lambda kv: (kv[1]["mean_symbols"], kv[0])):
        logger.info("%-16s %6.1f%% %9.0f %8.2f %8.1f %8.2f", name,
                    agg["accuracy"] * 100, agg["mean_symbols"],
                    agg["mean_budget_utilisation"], agg["mean_received_tokens"],
                    agg["restore_rate"])

    # iso-cost paired tests against the in-run Adaptive-SC anchor
    def paired(a_name: str, b_name: str) -> dict:
        a = {(r["sample_id"], r["channel"], r["seed"]): r["correct"]
             for r in by_strategy.get(a_name, [])}
        b = {(r["sample_id"], r["channel"], r["seed"]): r["correct"]
             for r in by_strategy.get(b_name, [])}
        wins = sum(1 for k in a if k in b and a[k] and not b[k])
        losses = sum(1 for k in a if k in b and b[k] and not a[k])
        return {"wins": wins, "losses": losses}

    iso = {f"fill_{name}": paired(f"fill_{name}", "anchor_adaptive")
           for name in targets}
    summary["iso_cost_paired_vs_anchor"] = iso
    atomic_json(out_dir / "summary.json", summary)
    for name, res in iso.items():
        logger.info("paired %s vs anchor_adaptive: %d : %d",
                    name, res["wins"], res["losses"])
    logger.info("DONE -> %s", out_dir / "summary.json")


if __name__ == "__main__":
    main()
