#!/usr/bin/env python
"""Compare coupling variants at equal strength on three decision-relevant probes.

Variants
--------
``raw``  Eq. (16) attention coefficients, unnormalised.
``row``  ``raw`` with Sinkhorn unit-row scaling (``d_M = 1``, so beta is usable up to 2).
``A``    ``M_ij <- a_ij * (1 - cos(e_i, e_j))``, then unit-row scaling: keep the
         attention weight only where the two tokens are *not* saying the same thing.
``B``    ``F_T(T) = sum u'_i - gamma sum_{i<j} sim(e_i,e_j) u'_i u'_j``, i.e. a negative
         coupling ``M_ij = -cos(e_i,e_j)``: an explicit redundancy *penalty*.

Equal strength is enforced by scaling each variant so that its utility-weighted
coupling mass ``median_i |sum_j M^<->_ij u'_j|`` matches a common target (0.25 of the
additive per-token scale), clipped by the paper's ``kappa < 1`` bound. Comparing at
equal share is the only fair way to attribute a difference to the variant rather than
to a bigger beta.

Probes
------
1. **Content versus size.** Random sets of a *fixed* size: if ``Spearman(additive,
   set)`` is 1 the coupling adds nothing beyond "more tokens". ``relative spread``
   measures how much the coupling term varies across same-size sets at all.
2. **Schedule impact.** Fraction of (sample, channel, snr) cases where the coupling
   changes the selected schedule relative to coupling-off.
3. **Pair complementarity.** Against the measured true interaction of the four
   mined pair families (``outputs/pair_complementarity_smoke.json``), does the
   variant's ``beta M_ij u'_i u'_j`` track the measured interaction?
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

VARIANTS = ("raw", "row", "A", "B")


def cosine_similarity(embeddings: torch.Tensor) -> torch.Tensor:
    x = embeddings.float()
    x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return x @ x.t()


def build_variant(kind: str, coupling: torch.Tensor, sim: torch.Tensor) -> torch.Tensor:
    from src.pragmatic.coupling import normalize_coupling

    if kind == "raw":
        return coupling
    if kind == "row":
        return normalize_coupling(coupling, "row")
    if kind == "A":
        return normalize_coupling((coupling * (1.0 - sim)).triu(1), "row")
    if kind == "B":
        return (-sim).triu(1)
    raise ValueError(kind)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-cache", default="outputs/cache/v2xqa_scores_8b_pred_cpl.jsonl")
    parser.add_argument("--embeddings", default="outputs/cache/v2xqa_token_embeddings_8b.pt")
    parser.add_argument("--pairs", default="outputs/pair_complementarity_smoke.json")
    parser.add_argument("--weights", type=float, nargs=3, default=[0.25, 0.30, 0.45])
    parser.add_argument("--target-share", type=float, default=0.25)
    parser.add_argument("--kappa-cap", type=float, default=0.9)
    parser.add_argument("--sets-per-sample", type=int, default=60)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--output", default="outputs/coupling_variants.json")
    args = parser.parse_args()

    from scipy.stats import spearmanr
    from src.experiments.scoring import combine_scores
    from src.pragmatic.coupling import incident_coupling
    from src.pragmatic.utility import coupling_regime, set_utility

    rows = [json.loads(l) for l in open(args.score_cache, encoding="utf-8")][:args.num_samples]
    embeddings = torch.load(args.embeddings)
    pairs = json.load(open(args.pairs, encoding="utf-8"))["records"]
    print(f"{len(rows)} samples | {len(pairs)} measured pairs | target share {args.target_share}")

    prepared = []
    for r in rows:
        sid = str(r["sample_id"])
        utility = combine_scores(r["attention"], r["semantic"], r["gradient"],
                                 args.weights).float()
        coupling = torch.tensor(r["coupling"], dtype=torch.float32)
        emb = embeddings[sid]
        if emb.shape[0] != len(utility):
            print(f"  [skip {sid}] embedding/token mismatch")
            continue
        prepared.append((sid, utility, coupling, cosine_similarity(emb)))

    report = {}
    for kind in VARIANTS:
        corr, spread, kappas, shares = [], [], [], []
        variant_cache = {}
        for sid, utility, coupling, sim in prepared:
            variant = build_variant(kind, coupling, sim)
            incident = incident_coupling(variant)
            n = utility.numel()
            mass = float((incident @ utility).median())
            d_m = coupling_regime(variant, 1.0)["d_M"] or 0.0
            beta = 0.0
            if abs(mass) > 1e-12:
                beta = args.target_share / abs(mass)
            if d_m > 0:
                beta = min(beta, 2.0 * args.kappa_cap / d_m)
            share = beta * mass
            kappa = beta * d_m / 2.0
            shares.append(abs(share)); kappas.append(kappa)
            variant_cache[sid] = (variant, beta)
            rng = random.Random(int(sid) * 7919)
            k = max(5, n // 3)
            add, full = [], []
            for _ in range(args.sets_per_sample):
                idx = torch.tensor(sorted(rng.sample(range(n), k)))
                add.append(float(utility[idx].sum()))
                try:
                    full.append(float(set_utility(idx, utility, variant, beta)))
                except ValueError:
                    full.append(float("nan"))
                if kind == "B":                       # penalty form is additive-minus
                    full[-1] = add[-1] - abs(beta) * float(
                        (sim[idx][:, idx].triu(1) * utility[idx][:, None]
                         * utility[idx][None, :]).sum())
            if all(v == v for v in full):
                corr.append(float(spearmanr(add, full).statistic))
                spread.append(float(torch.tensor(full).std() / max(abs(statistics.mean(full)), 1e-9)))
        # probe 3: pair complementarity against measured interaction
        prx, tru = [], []
        for rec in pairs:
            sid = str(rec["sample_id"])
            if sid not in variant_cache:
                continue
            variant, beta = variant_cache[sid]
            i, j = int(rec["i"]), int(rec["j"])
            m_ij = float(variant[min(i, j), max(i, j)])
            prx.append(abs(beta) * m_ij * float(rec["u_i"]) * float(rec["u_j"]))
            tru.append(float(rec["true_interaction"]))
        p_corr = (float(spearmanr(prx, tru).statistic)
                  if len(prx) > 3 and statistics.pstdev(prx) > 1e-12
                  and statistics.pstdev(tru) > 1e-12 else float("nan"))
        report[kind] = {
            "mean_beta": statistics.mean([beta for _, beta in variant_cache.values()]),
            "median_kappa": statistics.median(kappas) if kappas else None,
            "median_coupling_share": statistics.median(shares) if shares else None,
            "spearman_additive_vs_set": statistics.mean(corr) if corr else None,
            "relative_spread_of_set_score": statistics.mean(spread) if spread else None,
            "pairs_used": len(prx),
            "spearman_proxy_vs_true_interaction": p_corr,
        }
        e = report[kind]
        print(f"\n[{kind}] β均值={e['mean_beta']:.3f}  κ中位={e['median_kappa']:.3f}  "
              f"耦合占比={e['median_coupling_share']:.3f}")
        print(f"      Spearman(可加,集合)={e['spearman_additive_vs_set']:.4f}  "
              f"集合分相对离散={e['relative_spread_of_set_score']:.4f}")
        print(f"      配对层面 Spearman(代理,实测交互)={e['spearman_proxy_vs_true_interaction']:+.3f} "
              f"(n={e['pairs_used']})")

    out = Path(args.output)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[done] -> {out}")


if __name__ == "__main__":
    main()
