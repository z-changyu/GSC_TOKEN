#!/usr/bin/env python
"""Calibrate the reception model before trusting any scheduling result.

The scheduler plans against ``q_{i,ell}`` (marginal) and ``Q_{ell,m}`` (pairwise)
probabilities. If those disagree with how often tokens actually survive, every
downstream comparison is meaningless, so this script measures the discrepancy
directly by Monte-Carlo over channel realisations:

    predicted  = E_{h_hat} [ model probability ]      (what the scheduler believes)
    empirical  = fraction of realisations that survive (what the channel does)

Two things are checked:

1. **marginals** ``q_ell`` -- the conditional decoding probability of Eq. (5);
2. **pairs** ``Q_{ell,m}`` -- the joint probability the revised coupling term uses,
   against the empirical joint survival and against the old ``q_ell * q_m`` product.
   A shared instantaneous channel must make pairs *more* reliable than the product.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", nargs="+",
                        default=["configs/channels/rayleigh.yaml",
                                 "configs/channels/rician.yaml",
                                 "configs/channels/frequency_selective.yaml"])
    parser.add_argument("--snr-db", type=float, nargs="+", default=[0.0, 5.0, 15.0])
    parser.add_argument("--realisations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--levels", type=float, nargs="+", default=[48.0, 32.0, 16.0])
    parser.add_argument("--bits-per-token", type=float, default=16.0)
    parser.add_argument("--output", default="outputs/verify_reception_model.json")
    args = parser.parse_args()

    import yaml

    from src.channel import build_channel, decoding_probability
    from src.channel.link import ProtectionCodebook, joint_decoding_probability

    def load_channel_spec(path: str) -> dict:
        spec = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        spec.setdefault("name", Path(path).stem)
        return spec

    codebook = ProtectionCodebook(levels=tuple(args.levels),
                                  information_bits_per_token=args.bits_per_token)
    rates = torch.tensor(codebook.code_rates)
    generator = torch.Generator().manual_seed(args.seed)
    n_levels = codebook.num_levels

    report = []
    for channel_path in args.channels:
        base_spec = load_channel_spec(channel_path)
        for snr_db in args.snr_db:
            channel = build_channel({**base_spec, "snr_db": float(snr_db),
                                     "seed": int(args.seed)})
            state = channel.sample((args.realisations,))
            loss = float(torch.as_tensor(state.packet_loss_rate).reshape(-1)[0])

            # ---- what the scheduler believes
            predicted_q = decoding_probability(state, codebook).reshape(-1, n_levels)
            predicted_q_mean = predicted_q.mean(dim=0)
            predicted_joint = joint_decoding_probability(state, codebook).reshape(
                -1, n_levels, n_levels).mean(dim=0)

            # ---- what the channel actually does
            realized = torch.log2(1.0 + state.snr_linear * state.h.abs().pow(2)).reshape(-1)
            decoded = realized.reshape(-1, 1) >= rates.reshape(1, -1)      # [K, |R|]
            drop_a = torch.rand(decoded.shape, generator=generator) >= loss
            drop_b = torch.rand(decoded.shape, generator=generator) >= loss
            kept_a = decoded & drop_a
            kept_b = decoded & drop_b
            empirical_q = kept_a.float().mean(dim=0)
            empirical_joint = (kept_a[:, :, None] & kept_b[:, None, :]).float().mean(dim=0)
            product = torch.outer(empirical_q, empirical_q)

            entry = {
                "channel": channel.name, "snr_db": float(snr_db),
                "packet_loss_rate": loss,
                "predicted_q": [round(float(v), 5) for v in predicted_q_mean],
                "empirical_q": [round(float(v), 5) for v in empirical_q],
                "marginal_abs_error": [round(abs(float(a) - float(b)), 5)
                                       for a, b in zip(predicted_q_mean, empirical_q)],
                "joint_abs_error": round(
                    float((predicted_joint - empirical_joint).abs().max()), 5),
                "joint_minus_product": [
                    [round(float(empirical_joint[i, j] - product[i, j]), 5)
                     for j in range(n_levels)] for i in range(n_levels)],
            }
            report.append(entry)
            worst = max(entry["marginal_abs_error"])
            gap = min(min(row) for row in entry["joint_minus_product"])
            print(f"{channel.name:20s} {snr_db:5.1f} dB  "
                  f"max|q_pred - q_emp| = {worst:.4f}   "
                  f"joint-Q max err = {entry['joint_abs_error']:.4f}   "
                  f"min(Q_emp - q_i q_j) = {gap:+.4f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"realisations": args.realisations,
                               "levels": list(args.levels), "report": report},
                              indent=2), encoding="utf-8")
    worst_marginal = max(max(e["marginal_abs_error"]) for e in report)
    worst_joint = max(e["joint_abs_error"] for e in report)
    worst_gap = min(min(min(r) for r in e["joint_minus_product"]) for e in report)
    print(f"\nworst marginal error {worst_marginal:.4f} | worst joint error {worst_joint:.4f} | "
          f"min(Q_emp - product) {worst_gap:+.4f}")
    print("VERDICT: " + ("reception model is calibrated"
                         if worst_marginal < 0.02 and worst_joint < 0.02 else
                         "MISMATCH -- fix the link model before running the scheduler"))
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
