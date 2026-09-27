#!/usr/bin/env python
"""Experiment 4: scheduling latency versus CSI aging (Section VI-F extension).

The scheduler plans from the channel estimate available at decision time, but the
payload is decoded against the channel that exists at transmission time. Inserting
an age ``tau`` between the two decorrelates them (Clarke/Jakes, ``rho = J0(2 pi f_d
tau)``), so the protection levels chosen from the stale estimate no longer match
the realised channel. This driver measures the consequences and the mitigation.

Arms
----
``nominal``   plan from the aged estimate with Eq. (5)/(6) -- the paper design.
``robust``    plan from the same aged estimate but with the conservative
              Eq. (7)/(8) budget: writing ``h_hat = rho h + n`` shows that
              ``h_hat / rho`` carries an equivalent error variance
              ``nu_eff = ((1 - rho^2) + sigma_e^2) / rho^2``, which is what the
              distributionally robust capacity consumes.
``fresh``     plan from the *transmission-time* channel at the same age; the
              no-aging reference showing what the same schedule would achieve if
              the CSI were current.

Reported per (channel, snr, age, arm): planned violation (symbols above the budget
the scheduler itself used), realised reliability violation (erasure rate above
epsilon_out), decoding success, answer accuracy, transmitted symbols and accuracy
per kSymbol. Timing of the scheduling call is recorded as well, so the applicable
age follows from the measured compute time.

Usage
-----
    python scripts/run_csi_aging.py --limit 6 --ages 0 0.5 1        # smoke
    python scripts/run_csi_aging.py                                 # full sweep
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import random
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.channel import build_channel  # noqa: E402
from src.channel.link import ProtectionCodebook, decoding_probability  # noqa: E402
from src.channel.reliable_capacity import (  # noqa: E402
    reliable_spectral_efficiency, robust_reliable_spectral_efficiency)
from src.experiments.prompts import ANSWER_REQUEST  # noqa: E402
from src.experiments.receiver_protocol import answer_with_protocol  # noqa: E402
from src.experiments.scoring import SampleScores, combine_scores  # noqa: E402
from src.models.registry import load_generation_receiver  # noqa: E402
from src.scheduling import select_level  # noqa: E402
from src.utils.io import atomic_json, read_jsonl  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("csi_aging")
ARMS = ("fresh", "nominal", "robust")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default="configs/experiments/v2x_edge_edge_main_4b8b.yaml")
    parser.add_argument("--ages", type=float, nargs="*",
                        default=[0.0, 0.0002, 0.0005, 0.001, 0.002, 0.005],
                        help="age between CSI acquisition and transmission [s]")
    parser.add_argument("--doppler-hz", type=float, default=330.0,
                        help="330 Hz is 60 km/h at 5.9 GHz; 165 Hz is 30 km/h")
    parser.add_argument("--channels", nargs="*", default=None)
    parser.add_argument("--snr-db", type=float, nargs="*", default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--methods", nargs="*", default=["proposed", "adaptive_sc"])
    parser.add_argument("--weights", type=float, nargs=3, default=[0.25, 0.30, 0.45])
    parser.add_argument("--t-max", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", default="outputs/csi_aging")
    return parser.parse_args()


def summarise(values: list[float]) -> dict:
    xs = [v for v in values if v == v]
    if not xs:
        return {"n": 0, "mean": float("nan"), "p50": float("nan"), "p95": float("nan")}
    xs_sorted = sorted(xs)
    rank = min(len(xs_sorted) - 1, int(math.ceil(0.95 * len(xs_sorted))) - 1)
    return {"n": len(xs), "mean": statistics.fmean(xs),
            "p50": statistics.median(xs_sorted), "p95": xs_sorted[max(0, rank)]}


def main() -> None:
    args = parse_args()
    torch.set_grad_enabled(False)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    config["cuda_visible_devices"] = config.get("experiment", {}).get(
        "cuda_visible_devices", "0")

    cache = {str(r["sample_id"]): SampleScores.from_dict(r)
             for r in read_jsonl(config["score_cache"])}
    processed = {str(r["sample_id"]): r
                 for r in read_jsonl(config["dataset"]["processed"])}
    manifest = [str(r["sample_id"])
                for r in read_jsonl(config["dataset"]["manifest"])]
    manifest = [s for s in manifest if s in cache and processed.get(s, {}).get("image")]
    if args.limit:
        manifest = manifest[:args.limit]

    channel_paths = args.channels or config["channels"]
    snr_values = args.snr_db or [float(v) for v in config["snr_db"]]
    seeds = args.seeds or [int(v) for v in config["experiment"]["channel_seeds"]][:2]
    epsilon_out = float(config["radio"]["epsilon_out"])
    bandwidth = float(config["radio"]["bandwidth"])
    slot = float(config["radio"]["slot_duration"])
    levels = torch.tensor(config["radio"]["protection"]["levels"], dtype=torch.float32)
    codebook = ProtectionCodebook(
        levels=tuple(float(v) for v in config["radio"]["protection"]["levels"]),
        information_bits_per_token=float(
            config["radio"]["protection"]["information_bits_per_token"]))
    beta = float(config["pragmatic"]["beta"])
    weights = args.weights

    LOGGER.info("CSI aging sweep: %d samples, ages=%s, f_d=%.0f Hz",
                len(manifest), args.ages, args.doppler_hz)
    LOGGER.info("loading receiver %s", config["model"]["receiver_path"])
    from transformers import AutoTokenizer
    receiver_tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["receiver_path"], local_files_only=True)
    receiver = load_generation_receiver(config["model"]["receiver_path"])

    rows: list[dict] = []
    for channel_path in channel_paths:
        spec = yaml.safe_load(Path(channel_path).read_text())
        for snr_db in snr_values:
            for seed in seeds:
                for age in args.ages:
                    age_value = float(age)
                    params = {**spec, "snr_db": float(snr_db), "seed": int(seed),
                              "csi_age_s": age_value, "doppler_hz": args.doppler_hz}
                    fresh_params = {**params, "csi_age_s": 0.0}
                    for position, sid in enumerate(manifest):
                        scores = cache[sid]
                        row = processed[sid]
                        utility = torch.tensor(
                            [float(u) for u in combine_scores(
                                scores.attention, scores.semantic, scores.gradient,
                                weights)], dtype=torch.float32)
                        coupling = (torch.tensor(scores.coupling, dtype=torch.float32)
                                    if scores.coupling is not None else
                                    torch.zeros(scores.num_tokens, scores.num_tokens))
                        rng = random.Random(int(seed) * 1000003 + position)
                        # Paired design: the very same realised gain is used by all
                        # arms, so the comparison isolates the age effect.
                        state = build_channel(params).sample((1,))
                        state_fresh = build_channel(fresh_params).sample((1,))
                        rho = float(state.metadata.get("csi_correlation", 1.0))
                        sigma_e2 = float(state.csi_error_variance)
                        # Equivalent estimation-error variance carried by h_hat/rho.
                        if rho > 0.0:
                            nu_eff = ((1.0 - rho * rho) + sigma_e2) / (rho * rho)
                            state_robust = replace(state, h_hat=state.h_hat / rho,
                                                   csi_error_variance=nu_eff)
                        else:
                            state_robust = state
                        realized = math.log2(
                            1.0 + float(state.snr_linear)
                            * float(state.h.abs().pow(2)[0]))

                        plans = {}
                        for arm in ARMS:
                            if arm == "fresh":
                                plan_state = state_fresh
                                budget = float(slot * bandwidth * float(
                                    reliable_spectral_efficiency(state_fresh, epsilon_out)[0]))
                            elif arm == "nominal":
                                plan_state = state
                                budget = float(slot * bandwidth * float(
                                    reliable_spectral_efficiency(state, epsilon_out)[0]))
                            else:
                                plan_state = state_robust
                                budget = float(slot * bandwidth * float(
                                    robust_reliable_spectral_efficiency(
                                        state_robust, epsilon_out)[0]))
                            q = decoding_probability(plan_state, codebook).reshape(-1)
                            plans[arm] = (plan_state, budget, q)

                        method = args.methods[0]
                        for arm in ARMS:
                            plan_state, budget, q = plans[arm]
                            start = time.perf_counter()
                            level = select_level(
                                method, utility, coupling, beta, q, levels, budget,
                                args.t_max, rng)
                            schedule_seconds = time.perf_counter() - start
                            selected = [i for i in range(scores.num_tokens)
                                        if float(level[i]) > 0]
                            symbols = sum(float(codebook.levels[int(level[i]) - 1])
                                          for i in selected)
                            kept, lost, lost_channel, lost_packet = [], 0, 0, 0
                            for token in selected:
                                rate = float(codebook.code_rates[int(level[token]) - 1])
                                decoded = realized >= rate
                                dropped = rng.random() < float(state.packet_loss_rate)
                                if not decoded:
                                    lost_channel += 1
                                if decoded and dropped:
                                    lost_packet += 1
                                if decoded and not dropped:
                                    kept.append(token)
                                else:
                                    lost += 1
                            erasure = (lost / len(selected)) if selected else 0.0
                            # The outage target epsilon_out prices *channel* failures.
                            # The uniform packet-loss floor (5%) is a separate,
                            # irreducible impairment, so the reliability contract is
                            # checked against the channel-caused outage only.
                            outage_rate = (lost_channel / len(selected)) if selected else 0.0
                            received = ''.join(
                                receiver_tokenizer.decode([scores.token_ids[i]],
                                                          skip_special_tokens=True)
                                for i in kept)
                            answer = answer_with_protocol(
                                model=receiver, tokenizer=receiver_tokenizer, row=row,
                                received=received, survived_tokens=len(kept),
                                total_tokens=scores.num_tokens,
                                answer_request=ANSWER_REQUEST,
                                max_new_tokens=int(config["receiver"]["max_new_tokens"]),
                                protocol="v6", restore_threshold=0.0)
                            ground_truth = str(row["choices"][row["answer"]])
                            rows.append({
                                "channel": state.channel, "snr_db": float(snr_db),
                                "channel_seed": int(seed), "sample_id": sid,
                                "arm": arm, "method": method, "age_s": age_value,
                                "csi_correlation": rho,
                                "doppler_hz": args.doppler_hz,
                                "budget": budget, "symbols": symbols,
                                "selected_tokens": len(selected),
                                "received_tokens": len(kept),
                                "erasure_rate": erasure,
                                "outage_rate": outage_rate,
                                "packet_loss_rate": (lost_packet / len(selected))
                                if selected else 0.0,
                                "plan_violation": float(symbols > budget),
                                "reliability_violation": float(outage_rate > epsilon_out),
                                "correct": float(answer["prediction"] == ground_truth),
                                "schedule_seconds": schedule_seconds})
                    LOGGER.info("done %s/%s/seed=%d/age=%.4gs", state.channel, snr_db, seed, age_value)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "csi_aging_results.csv", index=False)

    groups = frame.groupby(["channel", "snr_db", "age_s", "arm"], sort=True)
    summary = {}
    for key, g in groups:
        channel, snr_db, age, arm = key
        summary[f"{channel}@{snr_db:g}dB|age={age * 1e3:g}ms|{arm}"] = {
            "channel": channel, "snr_db": float(snr_db), "age_s": float(age), "arm": arm,
            "csi_correlation": float(g.csi_correlation.iloc[0]),
            "n": int(len(g)),
            "plan_violation_rate": float(g.plan_violation.mean()),
            "reliability_violation_rate": float(g.reliability_violation.mean()),
            "erasure_rate": summarise(g.erasure_rate.tolist()),
            "outage_rate": summarise(g.outage_rate.tolist()),
            "packet_loss_rate": summarise(g.packet_loss_rate.tolist()),
            "decoding_success": float(1.0 - g.erasure_rate.mean()),
            "accuracy": summarise(g.correct.tolist()),
            "symbols": summarise(g.symbols.tolist()),
            "schedule_seconds": summarise(g.schedule_seconds.tolist()),
            "accuracy_per_ksymbol": (float(g.correct.mean()) * 1000.0
                                     / max(1e-9, float(g.symbols.mean())))}
    atomic_json(out_dir / "csi_aging_summary.json", summary)

    LOGGER.info("%-26s %8s %8s %9s %9s %9s", "cell", "rho", "planViol",
                "realViol", "decode", "acc")
    for name, cell in sorted(summary.items()):
        LOGGER.info("%-26s %8.3f %8.4f %9.4f %9.3f %9.3f", name,
                    cell["csi_correlation"], cell["plan_violation_rate"],
                    cell["reliability_violation_rate"], cell["decoding_success"],
                    cell["accuracy"]["mean"])
    LOGGER.info("wrote %s", out_dir)
    del receiver
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
