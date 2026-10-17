#!/usr/bin/env python
"""Calibrate the coupling strength beta against the paper's own weak-coupling bound.

Measured fact that motivates this script: with beta = 0.25 and the A/S/G utilities as
produced by the pipeline, the coupling term contributes ~3% of the additive term, so
the set-dependent objective never changes a scheduling decision -- the "joint
selection" of the paper is inert, and an ablation of it can only return a null result.

Two scales are computed per sample, over the cached pragmatic scores:

* **theory cap.**  Eq. (63): ``kappa = beta * d_M / 2 < 1``, i.e. ``beta < 2 / d_M``
  with ``d_M = max_i sum_j M^<->_ij``. Raising beta past this voids the certified
  rounding guarantee, so it is a hard ceiling, not a preference.
* **decision scale.** The insertion gain of Eq. (19)/(56) is
  ``u'_i q_{i,ell} (1 + beta * sum_j M^<->_ij u'_j q_{j,ell_j})``; the bracketed factor
  is how much the coupling moves the *ranking* the scheduler uses. Its typical size
  ``r = beta * s`` should be O(0.25-0.5) for the coupling to matter at all.

The script reports the beta that achieves a requested coupling share, and whether that
beta is reachable under the theory cap. If it is not, the finding is structural and
belongs in the paper: the weak-coupling condition itself forbids a decision-relevant
coupling for this coupling matrix.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-cache",
                        default="outputs/cache/v2xqa_scores_8b_pred_cpl.jsonl")
    parser.add_argument("--weights", type=float, nargs=3, default=[0.25, 0.30, 0.45])
    parser.add_argument("--beta", type=float, default=0.25, help="currently used beta")
    parser.add_argument("--shares", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    parser.add_argument("--output", default="outputs/beta_calibration.json")
    args = parser.parse_args()

    from src.experiments.scoring import combine_scores
    from src.pragmatic.coupling import incident_coupling
    from src.pragmatic.utility import coupling_regime

    rows = [json.loads(line) for line in open(args.score_cache, encoding="utf-8")]
    rows = [r for r in rows if r.get("coupling")]
    if not rows:
        raise SystemExit(f"{args.score_cache} carries no coupling matrix; rebuild it "
                         f"with scripts/build_score_cache.py --coupling")
    print(f"{len(rows)} samples with a coupling matrix from {args.score_cache}")

    d_m, s_values, r_values, sizes, record = [], [], [], [], []
    for r in rows:
        utility = combine_scores(r["attention"], r["semantic"], r["gradient"],
                                 args.weights).float()
        coupling = torch.tensor(r["coupling"], dtype=torch.float32)
        incident = incident_coupling(coupling)
        regime = coupling_regime(coupling, args.beta)

        # Eq. (19) convention: the reference set is every token, and the strongest
        # protection level is the one a token would get if the budget were free.
        per_token = utility * utility.max().clamp_min(1e-6)      # u'_j * q_j (any level)
        # r_i / beta = sum_j M^<->_ij u'_j q_j  -> the coupling share at beta = 1
        share_at_one = (incident @ (utility * 1.0)).abs()
        d_m.append(regime["d_M"])
        s_values.append(float(share_at_one.median()))
        r_values.append(float((args.beta * share_at_one).median()))
        sizes.append(int(utility.numel()))
        record.append({"sample_id": r["sample_id"], "tokens": int(utility.numel()),
                       "d_M": round(regime["d_M"], 4),
                       "kappa_at_current_beta": round(regime["kappa"], 4),
                       "coupling_share_at_current_beta": round(r_values[-1], 5),
                       "additive_scale": round(float((utility * 0.7).median()), 5)})

    beta_cap = 2.0 / max(d_m)                 # binding over samples
    beta_cap_per_sample = 2.0 / max(d_m)      # conservative: worst sample
    median_s = statistics.median(s_values)
    print(f"tokens/sample: median {statistics.median(sizes):.0f}")
    print(f"d_M: median {statistics.median(d_m):.4f}  max {max(d_m):.4f}")
    print(f"kappa at beta={args.beta}: median "
          f"{statistics.median([r['kappa_at_current_beta'] for r in record]):.4f}")
    print(f"coupling share r at beta={args.beta}: median "
          f"{statistics.median(r_values):.5f}  (additive marginal is ~1.0)")
    print()
    print(f"theory cap (Eq. 63, kappa < 1): beta < 2 / d_M(max) = {beta_cap:.3f}")
    print()
    print(f"{'target coupling share':>22s} {'beta required':>15s} {'reachable?':>12s}")
    plan = {}
    for share in args.shares:
        needed = share / median_s if median_s > 0 else float("inf")
        plan[f"share_{share}"] = {"beta_required": round(needed, 4),
                                  "reachable": bool(needed < beta_cap)}
        print(f"{share:22.2f} {needed:15.4f} "
              f"{'yes' if needed < beta_cap else 'NO (kappa>=1)':>12s}")

    recommended = min(beta_cap * 0.5, args.shares[-1] / median_s)
    print()
    print(f"recommended beta = min(0.5 * cap, beta(share={args.shares[-1]})) "
          f"= {recommended:.4f}   (kappa = {recommended * max(d_m) / 2:.3f})")
    summary = {"score_cache": args.score_cache, "weights": args.weights,
               "current_beta": args.beta,
               "d_M_median": statistics.median(d_m), "d_M_max": max(d_m),
               "beta_cap_kappa_lt_1": beta_cap,
               "coupling_share_at_current_beta": statistics.median(r_values),
               "share_per_unit_beta": median_s,
               "beta_plan": plan, "recommended_beta": round(recommended, 4),
               "per_sample": record}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
