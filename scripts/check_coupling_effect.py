#!/usr/bin/env python
"""Does the coupling term change any scheduling decision?

Run before paying for a full channel experiment: the calibration showed beta=0.25
gives a ~3.7% coupling share, and the paper's weak-coupling bound caps beta at
2/d_M ~= 0.68 (~10%). This script replays the scheduler on the cached scores under

    {joint, independent} x {beta} x {channel, snr}

and reports how often the *schedule itself* differs -- paired by a common random
number so the two arms see identical search randomness. If the schedules coincide, a
downstream accuracy comparison can only return a null result, regardless of sample
size.
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
    parser.add_argument("--score-cache", default=None)
    parser.add_argument("--beta", type=float, nargs="+", default=[0.25, 1.0, 1.5])
    parser.add_argument("--coupling-normalization", default="row", choices=["row", "none"],
                        help="'row' = Sinkhorn unit-row scaling so d_M = 1 and the "
                             "weak-coupling bound allows beta up to 2; 'none' = raw M")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--channels", nargs="*", default=None)
    parser.add_argument("--snr-db", type=float, nargs="*", default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=[0])
    parser.add_argument("--output", default="outputs/coupling_effect.json")
    args = parser.parse_args()

    import yaml
    from src.channel import build_channel, decoding_probability
    from src.channel.link import ProtectionCodebook, joint_decoding_probability
    from src.channel.reliable_capacity import reliable_spectral_efficiency
    from src.experiments.scoring import combine_scores
    from src.pragmatic.coupling import normalize_coupling
    from src.scheduling.methods import select_level

    def load_channel_spec(path: str) -> dict:
        spec = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        spec.setdefault("name", Path(path).stem)
        return spec

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    weights = tuple(config["pragmatic"].get("weights")
                    or [config["pragmatic"]["lambda_attention"],
                        config["pragmatic"]["lambda_semantic"],
                        config["pragmatic"]["lambda_gradient"]])
    codebook = ProtectionCodebook(
        levels=tuple(float(v) for v in config["radio"]["protection"]["levels"]),
        information_bits_per_token=float(
            config["radio"]["protection"]["information_bits_per_token"]))
    rates = torch.tensor(codebook.levels, dtype=torch.float32)
    bandwidth = float(config["radio"]["bandwidth"])
    slot = float(config["radio"]["slot_duration"])
    eps_out = float(config["radio"]["epsilon_out"])
    t_max = int(config["scheduling"]["t_max"])

    cache_path = args.score_cache or config["score_cache"]
    rows = [json.loads(line) for line in open(cache_path, encoding="utf-8")]
    rows = [r for r in rows if r.get("coupling")][:args.num_samples]
    print(f"{len(rows)} samples with coupling from {cache_path}")

    channels = args.channels or config["channels"]
    snrs = args.snr_db if args.snr_db is not None else [float(v) for v in config["snr_db"]]
    seeds = args.seeds

    report = []
    for beta in args.beta:
        differ = total = 0
        coupling_moved = 0
        token_diff = []
        coupling_diff = []
        for channel_path in channels:
            spec = load_channel_spec(channel_path)
            for snr in snrs:
                for seed in seeds:
                    channel = build_channel({**spec, "snr_db": float(snr), "seed": int(seed)})
                    for position, row in enumerate(rows):
                        utility = combine_scores(row["attention"], row["semantic"],
                                                 row["gradient"], weights).float()
                        coupling = normalize_coupling(
                            torch.tensor(row["coupling"], dtype=torch.float32),
                            args.coupling_normalization)
                        state = channel.sample((1,))
                        budget = slot * bandwidth * float(
                            reliable_spectral_efficiency(state, eps_out)[0])
                        q = decoding_probability(state, codebook).reshape(-1)
                        both = joint_decoding_probability(state, codebook).reshape(
                            codebook.num_levels, codebook.num_levels)
                        # Three arms, identical search randomness: no coupling at all,
                        # coupling under the independent-pair model, coupling under the
                        # joint (shared-channel) model. The first contrast answers
                        # "does the set-dependence matter?", the second "does the
                        # corrected pair probability matter?".
                        arms = {
                            "off": (torch.zeros_like(coupling), None),
                            "indep": (coupling, None),
                            "joint": (coupling, both),
                        }
                        levels = {}
                        for tag, (cpl, jq) in arms.items():
                            rng = random.Random(int(seed) * 1000003 + position)
                            levels[tag] = select_level(
                                "proposed", utility=utility, coupling=cpl, beta=beta,
                                q=q, rates=rates, budget=budget, t_max=t_max,
                                rng=rng, joint_q=jq)
                        total += 1
                        d_joint = int((levels["joint"] != levels["indep"]).sum())
                        d_coupling = int((levels["indep"] != levels["off"]).sum())
                        token_diff.append(d_joint)
                        coupling_diff.append(d_coupling)
                        differ += int(d_joint > 0)
                        coupling_moved += int(d_coupling > 0)
        entry = {"beta": beta, "coupling_normalization": args.coupling_normalization,
                 "compared": total,
                 "joint_vs_independent_fraction": differ / max(1, total),
                 "coupling_on_vs_off_fraction": coupling_moved / max(1, total),
                 "mean_tokens_changed_joint": (sum(token_diff) / len(token_diff)) if token_diff else 0.0,
                 "mean_tokens_changed_coupling": (sum(coupling_diff) / len(coupling_diff)) if coupling_diff else 0.0,
                 "max_tokens_changed_coupling": max(coupling_diff) if coupling_diff else 0}
        report.append(entry)
        print(f"beta={beta:5.2f}  耦合开vs关: 调度不同 {entry['coupling_on_vs_off_fraction']*100:5.1f}% "
              f"(平均 {entry['mean_tokens_changed_coupling']:.2f} token, 最大 {entry['max_tokens_changed_coupling']})"
              f"   |   joint vs independent: {entry['joint_vs_independent_fraction']*100:5.1f}%")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"score_cache": cache_path, "report": report}, indent=2),
                   encoding="utf-8")
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
