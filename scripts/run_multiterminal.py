#!/usr/bin/env python
"""Section VI-B-5: multi-terminal scheduling, K in {4, 10, 20}.

Each slot holds K ScienceQA samples, one per terminal. Terminals report
``g_hat_k = w_k T_s c_eps(h_hat_k)`` (Eq. 47) and the coordinator solves the
proportional-fair KKT allocation of Eq. (46) by bisection on ``nu``
(``allocate_bandwidth``). Each terminal then runs the single-terminal
reliability-aware scheduler of Algorithm 1 on its own budget ``C_tau_k =
T_s W_k B c_eps(h_hat_k)`` and the receiver answers from the surviving tokens
alone.

The accuracy/fairness Pareto front of Eq. (62) is traced by sweeping the
proportional-fair weight ``theta``: ``theta -> 0`` collapses to bang-bang
(everything to the best terminal, high mean accuracy, low Jain), while
``theta -> inf`` is the equal split. Mean per-terminal accuracy and the Jain
index (on the per-slot vector of terminal accuracies) are recorded for every
operating point.
"""
from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.channel import build_channel, decoding_probability  # noqa: E402
from src.channel.link import ProtectionCodebook  # noqa: E402
from src.channel.reliable_capacity import reliable_spectral_efficiency  # noqa: E402
from src.experiments.prompts import ANSWER_REQUEST, canonical_prediction  # noqa: E402
from src.experiments.scoring import SampleScores, combine_scores  # noqa: E402
from src.metrics.qa import exact_match  # noqa: E402
from src.metrics.statistics import mean_std_ci95  # noqa: E402
from src.scheduling.bandwidth import allocate_bandwidth, jain_index  # noqa: E402
from src.scheduling.methods import select_level  # noqa: E402
from src.utils.io import atomic_json, read_jsonl  # noqa: E402
from src.utils.logging import configure_logging, create_run_directory  # noqa: E402
from src.utils.power import PowerSampler  # noqa: E402
from src.utils.seed import set_global_seed  # noqa: E402


def apply_realized_channel(state, selected, codebook, rng):
    """Return the token indices that survive erasure on the realized channel.

    ``selected`` is a list of ``(idx, level)`` pairs (level 0 = not transmitted).
    """
    realized = math.log2(1.0 + float(state.snr_linear) * float(state.h.abs().pow(2)[0]))
    kept = []
    for idx, level in selected:
        if level <= 0:
            continue
        rate = float(codebook.code_rates[level - 1])
        decoded = realized >= rate
        dropped = rng.random() < float(state.packet_loss_rate)
        if decoded and not dropped:
            kept.append(idx)
    return kept


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/channel_scienceqa.yaml")
    parser.add_argument("--num-samples", type=int, default=60)
    parser.add_argument("--ks", type=int, nargs="*", default=(4, 10, 20))
    parser.add_argument("--snr-db", type=float, nargs="*", default=(0.0, 15.0))
    parser.add_argument("--thetas", type=float, nargs="*",
                        default=(0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 1e9))
    parser.add_argument("--weights", type=float, nargs=3, default=(0.25, 0.0, 0.75))
    parser.add_argument("--output-dir", default="outputs/multiterminal")
    args = parser.parse_args()

    config = (json.loads(Path(args.config).read_text()) if args.config.endswith(".json")
              else __import__("yaml").safe_load(Path(args.config).read_text()))
    set_global_seed(int(config["experiment"]["model_seed"]))
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    run = create_run_directory("outputs/runs", "multiterminal_experiment", vars(args))
    logger = configure_logging(run / "run.log")

    weights = tuple(float(w) for w in args.weights)
    codebook = ProtectionCodebook(
        levels=tuple(float(v) for v in config["radio"]["protection"]["levels"]),
        information_bits_per_token=float(config["radio"]["protection"]["information_bits_per_token"]))
    bandwidth = float(config["radio"]["bandwidth"])
    slot_duration = float(config["radio"]["slot_duration"])
    epsilon_out = float(config["radio"]["epsilon_out"])
    beta = float(config["pragmatic"]["beta"])
    coupling_enabled = bool(config["pragmatic"].get("coupling_enabled", False))

    cache = {row["sample_id"]: SampleScores.from_dict(row)
             for row in read_jsonl(config["score_cache"])}
    processed = {str(row["row_idx"]): row for row in read_jsonl(config["dataset"]["processed"])}
    manifest = [str(entry["sample_id"]) for entry in read_jsonl(config["dataset"]["manifest"])]
    manifest = [sid for sid in manifest if sid in cache][: args.num_samples]
    logger.info("multi-terminal: %d terminals, K=%s, SNR=%s, thetas=%s",
                len(manifest), args.ks, args.snr_db, args.thetas)

    receiver_tokenizer = AutoTokenizer.from_pretrained(config["model"]["receiver_path"],
                                                       local_files_only=True)
    from src.models.registry import load_generation_receiver
    receiver = load_generation_receiver(config["model"]["receiver_path"])
    tx_tokenizer = AutoProcessor.from_pretrained(config["model"]["transmitter_path"],
                                                 local_files_only=True).tokenizer
    power = PowerSampler()

    records = []
    for snr in args.snr_db:
        for k in args.ks:
            slots_base = [manifest[i:i + k] for i in range(0, len(manifest), k)]
            slots_base = [s for s in slots_base if len(s) == k]   # keep only full slots
            # Every channel seed re-evaluates the same slot partition, so the
            # pooled statistic rests on len(slots_base) * len(seed_list) units
            # instead of one realisation (K=20 would otherwise have only 3 slots).
            seed_list = [int(v) for v in config["experiment"]["channel_seeds"]] or [0]
            slots = [s for s in slots_base for _ in seed_list]
            for theta in args.thetas:
                per_slot_correct, per_slot_jain_w = [], []
                for slot_idx, slot in enumerate(slots):
                    reports, terminals = [], []
                    for t_idx, sid in enumerate(slot):
                        scores = cache[sid]
                        n = scores.num_tokens
                        u_prime = torch.tensor(combine_scores(
                            scores.attention, scores.semantic, scores.gradient, weights),
                            dtype=torch.float32)
                        coupling = (torch.tensor(scores.coupling, dtype=torch.float32)
                                    if (coupling_enabled and scores.coupling) else
                                    torch.zeros(n, n))
                        base_idx = slot_idx % len(slots_base)
                        channel_seed = seed_list[slot_idx // len(slots_base)]
                        seed = channel_seed * 100003 + base_idx * 97 + t_idx
                        channel_spec = __import__("yaml").safe_load(open(config["channels"][0]))
                        channel_spec.update({"snr_db": float(snr), "seed": seed})
                        channel = build_channel(channel_spec)
                        state = channel.sample((1,))
                        c_eps = float(reliable_spectral_efficiency(state, epsilon_out)[0])
                        q = decoding_probability(state, codebook).reshape(-1)
                        report = float(u_prime.sum()) * slot_duration * c_eps
                        reports.append(report)
                        terminals.append((scores, u_prime, coupling, state, c_eps, q, sid))
                    bandwidths = allocate_bandwidth(torch.tensor(reports), 1.0,
                                                    float(theta)).bandwidth
                    slot_correct = []
                    for t_idx, (scores, u_prime, coupling, state, c_eps, q, sid) in enumerate(
                            terminals):
                        budget = slot_duration * float(bandwidths[t_idx]) * bandwidth * c_eps
                        level = select_level(
                            "proposed", u_prime, coupling, beta, q,
                            torch.tensor(codebook.levels, dtype=torch.float32),
                            budget, int(config["scheduling"]["t_max"]), __import__("random").Random(slot_idx * 7 + t_idx))
                        selected = [(int(i), int(level[i])) for i in range(scores.num_tokens)
                                    if level[i] > 0]
                        rng = __import__("random").Random(slot_idx * 13 + t_idx)
                        kept_ids = apply_realized_channel(state, selected, codebook, rng)
                        received = tx_tokenizer.decode(
                            [scores.token_ids[i] for i in kept_ids], skip_special_tokens=True)
                        message = receiver_tokenizer.apply_chat_template(
                            [{"role": "user", "content": f"{ANSWER_REQUEST}\n\n{received}"}],
                            tokenize=False, add_generation_prompt=True, enable_thinking=False)
                        inputs = receiver_tokenizer(message, return_tensors="pt").to("cuda:0")
                        with torch.no_grad():
                            output = receiver.generate(**inputs, max_new_tokens=config["receiver"]["max_new_tokens"],
                                                      do_sample=False)
                        raw = receiver_tokenizer.decode(
                            output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
                        prediction, _ = canonical_prediction(raw, processed[sid]["choices"])
                        correct = exact_match(prediction, str(processed[sid]["choices"][processed[sid]["answer"]]))
                        slot_correct.append(float(correct))
                        records.append({"snr_db": float(snr), "K": k, "theta": float(theta),
                                        "slot": slot_idx, "base_slot": base_idx,
                                        "channel_seed": channel_seed, "terminal": sid,
                                        "bandwidth_share": float(bandwidths[t_idx]),
                                        "budget": budget, "correct": correct,
                                        "selected_tokens": len(selected),
                                        "received_tokens": len(kept_ids)})
                    per_slot_correct.append(np.mean(slot_correct))
                    per_slot_jain_w.append(jain_index(torch.tensor([float(b) for b in bandwidths])))
                mean_acc = mean_std_ci95(per_slot_correct)
                mean_jain_w = float(np.mean(per_slot_jain_w))
                logger.info("snr=%.0f K=%d theta=%.3g mean_acc=%.4f jain_w=%.4f nslots=%d",
                            snr, k, theta, mean_acc["mean"], mean_jain_w, len(slots))

    frame = pd.DataFrame(records)
    frame.to_csv(output_dir / "multiterminal_results.csv", index=False)
    # Aggregate the front per (snr, K, theta): mean accuracy and Jain on outcomes.
    front_rows = []
    for (snr, k, theta), group in frame.groupby(["snr_db", "K", "theta"], sort=True):
        # Jain index measures fairness *across the K terminals within each slot*;
        # we average the per-slot Jain over slots. Computed for both the achieved
        # accuracies (fairness of outcomes) and the bandwidth shares (allocation).
        per_slot_acc, per_slot_w = [], []
        for _, g in group.groupby("slot"):
            per_slot_acc.append(jain_index(torch.tensor(g["correct"].values.astype(float))))
            per_slot_w.append(jain_index(torch.tensor(g["bandwidth_share"].values.astype(float))))
        front_rows.append({"snr_db": float(snr), "K": int(k), "theta": float(theta),
                           "mean_accuracy": float(group["correct"].mean()),
                           "accuracy_ci95": float(mean_std_ci95(
                               list(group["correct"].astype(float))).get("ci95", 0.0)),
                           "jain_accuracy": float(np.mean(per_slot_acc)),
                           "jain_bandwidth": float(np.mean(per_slot_w)),
                           "mean_selected_tokens": float(group["selected_tokens"].mean()),
                           "mean_received_tokens": float(group["received_tokens"].mean())})
    summary = {"samples": len(manifest), "K": list(args.ks), "snr_db": list(args.snr_db),
               "thetas": list(args.thetas), "weights": list(weights),
               "epsilon_out": epsilon_out, "energy_backend": power.backend,
               "front": front_rows}
    atomic_json(output_dir / "summary.json", summary)
    atomic_json(run / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "front"}, indent=2))
    print(pd.DataFrame(front_rows)[["snr_db", "K", "theta", "mean_accuracy",
                                    "jain_accuracy"]].to_string(index=False))


if __name__ == "__main__":
    main()
