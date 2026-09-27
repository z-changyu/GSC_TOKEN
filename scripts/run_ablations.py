#!/usr/bin/env python
"""Section VI-B ablation study (the paper's nine ablations).

Each ablation keeps the end-to-end channel pipeline identical and flips exactly
one modelling choice, so the accuracy at a matched communication budget isolates
the contribution of that choice:

  proposed      - full method: calibrated weights, coupling, LP+rounding+refine,
                  differentiated protection (all |R| levels live).
  no_coupling   - drop the direction-preserving coupling term (Eq. 16): set M=0.
  no_gradient   - drop G_i (Eq. 12-14): lambda_G = 0  -> proxy is purely A+S.
  no_semantic   - drop S_i (Eq. 11): lambda_S = 0.
  no_attention  - drop A_i (Eq. 7): lambda_A = 0.
  static        - legacy fixed weights (0.25, 0.30, 0.45) instead of calibrated.
  single_level  - disable differentiated protection: every kept token uses rho_H.
  lp_only       - drop the coupling-aware local refinement (Eq. 56-59): t_max = 0.

The channel is fixed to Rayleigh at 15 dB so the comparison is not confounded by
the channel; the realised outage is reported to confirm the reliable-capacity
guard still holds under every ablation.
"""
from __future__ import annotations

import argparse, json, math, random, sys, time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.channel import build_channel, decoding_probability  # noqa: E402
from src.channel.link import ProtectionCodebook  # noqa: E402
from src.channel.reliable_capacity import reliable_spectral_efficiency  # noqa: E402
from src.experiments.prompts import ANSWER_REQUEST, canonical_prediction  # noqa: E402
from src.experiments.scoring import SampleScores, combine_scores  # noqa: E402
from src.metrics.communication import token_retention_ratio  # noqa: E402
from src.metrics.qa import exact_match  # noqa: E402
from src.metrics.statistics import mean_std_ci95  # noqa: E402
from src.scheduling.methods import select_level  # noqa: E402
from src.utils.io import atomic_json, read_jsonl  # noqa: E402
from src.utils.logging import configure_logging, create_run_directory  # noqa: E402
from src.utils.power import PowerSampler  # noqa: E402
from src.utils.seed import set_global_seed  # noqa: E402


def apply_realized(state, selected, codebook, rng):
    realized = math.log2(1.0 + float(state.snr_linear) * float(state.h.abs().pow(2)[0]))
    kept = []
    for idx, level in selected:
        if level <= 0:
            continue
        decoded = realized >= float(codebook.code_rates[level - 1])
        dropped = rng.random() < float(state.packet_loss_rate)
        if decoded and not dropped:
            kept.append(idx)
    return kept


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/channel_scienceqa.yaml")
    parser.add_argument("--output-dir", default="outputs/ablations")
    parser.add_argument("--snr-db", type=float, nargs="*", default=(0.0, 5.0, 10.0, 15.0))
    parser.add_argument("--bandwidth", type=float, default=None,
                        help="Override B so the budget binds across the SNR sweep")
    parser.add_argument("--channel", type=str, default=None,
                        help="Channel YAML (rician.yaml / frequency_selective.yaml). "
                             "Defaults to config channels[0] (rayleigh).")
    parser.add_argument("--num-samples", type=int, default=60)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_global_seed(int(config["experiment"]["model_seed"]))
    # Per-channel output so Rayleigh results stay intact when sweeping other channels.
    channel_path = args.channel or config["channels"][0]
    channel_name = yaml.safe_load(open(channel_path))["name"]
    output_dir = Path(args.output_dir) / channel_name
    output_dir.mkdir(parents=True, exist_ok=True)
    run = create_run_directory("outputs/runs", "ablation_study", config)
    logger = configure_logging(run / "run.log")

    codebook = ProtectionCodebook(
        levels=tuple(float(v) for v in config["radio"]["protection"]["levels"]),
        information_bits_per_token=float(config["radio"]["protection"]["information_bits_per_token"]))
    bandwidth = float(args.bandwidth or config["radio"]["bandwidth"])
    slot_duration = float(config["radio"]["slot_duration"])
    epsilon_out = float(config["radio"]["epsilon_out"])
    beta = float(config["pragmatic"]["beta"])

    cache = {row["sample_id"]: SampleScores.from_dict(row)
             for row in read_jsonl(config["score_cache"])}
    processed = {str(row["row_idx"]): row for row in read_jsonl(config["dataset"]["processed"])}
    manifest = [str(entry["sample_id"]) for entry in read_jsonl(config["dataset"]["manifest"])]
    manifest = [sid for sid in manifest if sid in cache][:args.num_samples]
    tx_tokenizer = AutoProcessor.from_pretrained(config["model"]["transmitter_path"],
                                                 local_files_only=True).tokenizer
    receiver_tokenizer = AutoTokenizer.from_pretrained(config["model"]["receiver_path"],
                                                       local_files_only=True)
    from src.models.registry import load_generation_receiver
    receiver = load_generation_receiver(config["model"]["receiver_path"])
    power = PowerSampler()

    tx_tokenizer = AutoProcessor.from_pretrained(config["model"]["transmitter_path"],
                                                 local_files_only=True).tokenizer

    configs = {
        "proposed":     dict(weights=(0.25, 0.0, 0.75), coupling=True,  t_max=8, levels=len(codebook.levels)),
        "no_coupling":  dict(weights=(0.25, 0.0, 0.75), coupling=False, t_max=8, levels=len(codebook.levels)),
        "no_gradient":  dict(weights=(0.25, 0.75, 0.0), coupling=True,  t_max=8, levels=len(codebook.levels)),
        "no_semantic":  dict(weights=(0.25, 0.0, 0.75), coupling=True,  t_max=8, levels=len(codebook.levels)),
        "no_attention": dict(weights=(0.0, 0.25, 0.75), coupling=True,  t_max=8, levels=len(codebook.levels)),
        "static":       dict(weights=(0.25, 0.30, 0.45), coupling=True, t_max=8, levels=len(codebook.levels)),
        "single_level": dict(weights=(0.25, 0.0, 0.75), coupling=True,  t_max=8, levels=1),
        "lp_only":      dict(weights=(0.25, 0.0, 0.75), coupling=True,  t_max=0, levels=len(codebook.levels)),
    }

    records = []
    for snr in args.snr_db:
        spec = yaml.safe_load(open(channel_path))
        spec.update({"snr_db": float(snr), "seed": int(config["experiment"]["channel_seeds"][0])})
        logger.info("=== SNR %.0f dB, B=%.0f Hz ===", snr, bandwidth)
        for name, cfg in configs.items():
            rates = torch.tensor(codebook.levels[:cfg["levels"]], dtype=torch.float32)
            correct, symbols, trr, outages, kept_counts = [], [], [], [], []
            for position, sample_id in enumerate(manifest):
                scores = cache[sample_id]
                n = scores.num_tokens
                util = torch.tensor(combine_scores(scores.attention, scores.semantic, scores.gradient,
                                                   cfg["weights"]), dtype=torch.float32)
                coupling = (torch.tensor(scores.coupling, dtype=torch.float32) if (cfg["coupling"] and scores.coupling)
                            else torch.zeros(n, n))
                state = build_channel(spec).sample((1,))
                c_eps = float(reliable_spectral_efficiency(state, epsilon_out)[0])
                budget = slot_duration * bandwidth * c_eps
                q = decoding_probability(state, codebook).reshape(-1)[:cfg["levels"]]
                rng = random.Random(position + 1)
                level = select_level("proposed", util, coupling, beta, q, rates, budget, cfg["t_max"], rng)
                selected = [(int(i), int(level[i])) for i in range(n) if level[i] > 0]
                kept = apply_realized(state, selected, codebook, rng)
                received = tx_tokenizer.decode([scores.token_ids[i] for i in kept], skip_special_tokens=True)
                message = receiver_tokenizer.apply_chat_template(
                    [{"role": "user", "content": f"{ANSWER_REQUEST}\n\n{received}"}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
                inputs = receiver_tokenizer(message, return_tensors="pt").to("cuda:0")
                with torch.no_grad():
                    out = receiver.generate(**inputs, max_new_tokens=config["receiver"]["max_new_tokens"],
                                            do_sample=False)
                raw = receiver_tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
                pred, _ = canonical_prediction(raw, processed[sample_id]["choices"])
                correct.append(exact_match(pred, str(processed[sample_id]["choices"][processed[sample_id]["answer"]])))
                symbols.append(float(rates[(level - 1).clamp(min=0)][level > 0].sum()) if any(level > 0) else 0.0)
                trr.append(token_retention_ratio(int(sum(1 for i in range(n) if level[i] > 0)), n))
                realized_eff = math.log2(1.0 + float(state.snr_linear) * float(state.h.abs().pow(2)[0]))
                outages.append(int(slot_duration * bandwidth * realized_eff < budget))
                kept_counts.append(len(kept))
                del out, inputs
            acc = mean_std_ci95([float(c) for c in correct])
            records.append({
                "snr_db": float(snr), "ablation": name, "accuracy": acc["mean"],
                "accuracy_ci95": acc["ci95"], "mean_symbols": float(pd.Series(symbols).mean()),
                "mean_TRR": float(pd.Series(trr).mean()),
                "mean_kept_tokens": float(pd.Series(kept_counts).mean()),
                "capacity_outage_rate": float(pd.Series(outages).mean()), "n": len(correct)})
            logger.info("ablation %-12s acc=%.4f TRR=%.3f symbols=%.1f outage=%.3f",
                        name, acc["mean"], records[-1]["mean_TRR"], records[-1]["mean_symbols"],
                        records[-1]["capacity_outage_rate"])

    frame = pd.DataFrame(records)
    frame.to_csv(output_dir / "ablation_results.csv", index=False)
    summary = {"channel": channel_name, "snr_db": list(args.snr_db), "bandwidth": bandwidth,
               "epsilon_out": epsilon_out, "energy_backend": power.backend,
               "ablations": records}
    atomic_json(output_dir / "summary.json", summary)
    atomic_json(run / "summary.json", summary)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
