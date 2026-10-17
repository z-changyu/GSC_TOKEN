#!/usr/bin/env python
"""Schedule impact of the coupling definition and of where the coupling is applied.

Three questions, answered by replaying the scheduler on cached scores with identical
randomness across arms:

1. does the coupling change the selected schedule at all (coupling on vs off)?
2. does the coupling *definition* matter (raw / row / A / B)?
3. does it matter *where* the coupling is applied -- only in the local refinement
   (``--lp-utility marginal``) or also in the LP/rounding surrogate via the Eq. (19)
   context-aware utility (``--lp-utility context``)?

The third is the decisive one: the base LP and the certified rounding are additive by
construction, so with a marginal LP the coupling can only flip the at most ``t_max``
refinement moves.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/v2x_edge_edge_8b_rx8b_smoke.yaml")
    parser.add_argument("--score-cache", default="outputs/cache/v2xqa_scores_8b_pred_cpl.jsonl")
    parser.add_argument("--embeddings", default="outputs/cache/v2xqa_token_embeddings_8b.pt")
    parser.add_argument("--variants", nargs="+", default=["row", "A", "B"])
    parser.add_argument("--lp-utility", nargs="+", default=["marginal", "context"],
                        choices=["marginal", "context"])
    parser.add_argument("--target-share", type=float, default=0.25)
    parser.add_argument("--kappa-cap", type=float, default=0.9)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="*", default=[0])
    parser.add_argument("--output", default="outputs/variant_schedule_effect.json")
    args = parser.parse_args()

    import yaml
    from src.channel import build_channel, decoding_probability
    from src.channel.link import ProtectionCodebook, joint_decoding_probability
    from src.channel.reliable_capacity import reliable_spectral_efficiency
    from src.experiments.scoring import combine_scores
    from src.pragmatic.coupling import (build_coupling_variant, incident_coupling)
    from src.pragmatic.utility import context_aware_utility, coupling_regime
    from src.scheduling.methods import select_level

    def load_channel_spec(path: str) -> dict:
        spec = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        spec.setdefault("name", Path(path).stem)
        return spec

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    pragmatic = config["pragmatic"]
    weights = tuple(pragmatic.get("weights") or [
        pragmatic["lambda_attention"], pragmatic["lambda_semantic"], pragmatic["lambda_gradient"]])
    codebook = ProtectionCodebook(
        levels=tuple(float(v) for v in config["radio"]["protection"]["levels"]),
        information_bits_per_token=float(
            config["radio"]["protection"]["information_bits_per_token"]))
    rates = torch.tensor(codebook.levels, dtype=torch.float32)
    bandwidth = float(config["radio"]["bandwidth"])
    slot = float(config["radio"]["slot_duration"])
    eps_out = float(config["radio"]["epsilon_out"])
    t_max = int(config["scheduling"]["t_max"])

    rows = [json.loads(l) for l in open(args.score_cache, encoding="utf-8")][:args.num_samples]
    embeddings = torch.load(args.embeddings, weights_only=False)
    prepared = []
    for r in rows:
        sid = str(r["sample_id"])
        prepared.append((sid,
                         combine_scores(r["attention"], r["semantic"], r["gradient"],
                                        weights).float(),
                         torch.tensor(r["coupling"], dtype=torch.float32),
                         embeddings[sid]))

    conditions = []
    for channel_path in config["channels"]:
        spec = load_channel_spec(channel_path)
        for snr in [float(v) for v in config["snr_db"]]:
            for seed in args.seeds:
                conditions.append((build_channel({**spec, "snr_db": snr, "seed": int(seed)}),
                                   snr, seed))

    def arm_levels(utility, coupling, beta, q, budget, both, selection_utility, seed, pos):
        rng = random.Random(int(seed) * 1000003 + pos)
        return select_level("proposed", utility=utility, coupling=coupling, beta=beta,
                            q=q, rates=rates, budget=budget, t_max=t_max, rng=rng,
                            joint_q=both, selection_utility=selection_utility)

    report = {}
    for variant in args.variants:
        for lp in args.lp_utility:
            differ = total = 0
            changed = []
            for channel, snr, seed in conditions:
                state = channel.sample((1,))
                budget = slot * bandwidth * float(
                    reliable_spectral_efficiency(state, eps_out)[0])
                q = decoding_probability(state, codebook).reshape(-1)
                both = joint_decoding_probability(state, codebook).reshape(
                    codebook.num_levels, codebook.num_levels)
                for pos, (sid, utility, raw_coupling, emb) in enumerate(prepared):
                    variant_coupling = build_coupling_variant(variant, raw_coupling, emb)
                    incident = incident_coupling(variant_coupling)
                    mass = float((incident @ utility).median())
                    d_m = coupling_regime(variant_coupling, 1.0)["d_M"] or 0.0
                    beta = (args.target_share / abs(mass)) if abs(mass) > 1e-12 else 0.0
                    if d_m > 0:
                        beta = min(beta, 2.0 * args.kappa_cap / d_m)
                    selection = None
                    if lp == "context":
                        selection = context_aware_utility(utility, variant_coupling, beta)
                    on = arm_levels(utility, variant_coupling, beta, q, budget, both,
                                    selection, seed, pos)
                    off = arm_levels(utility, torch.zeros_like(variant_coupling), 0.0, q,
                                     budget, None, None, seed, pos)
                    total += 1
                    delta = int((on != off).sum())
                    changed.append(delta)
                    differ += int(delta > 0)
            entry = {"variant": variant, "lp_utility": lp, "compared": total,
                     "fraction": differ / max(1, total),
                     "mean_tokens_changed": sum(changed) / max(1, len(changed)),
                     "max_tokens_changed": max(changed) if changed else 0}
            report[f"{variant}/{lp}"] = entry
            print(f"[{variant:4s}/{lp:9s}] 耦合开vs关：调度不同 {entry['fraction']*100:5.1f}%  "
                  f"平均 {entry['mean_tokens_changed']:.2f} token"
                  f"（最大 {entry['max_tokens_changed']}）  n={total}")
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] -> {args.output}")


if __name__ == "__main__":
    main()
