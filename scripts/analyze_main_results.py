#!/usr/bin/env python
"""Aggregate and compare channel-experiment arms (fair vs legacy protocol).

Usage
-----
    python scripts/analyze_main_results.py \
        --arms fair=outputs/channel_main_2b4b_fair \
               legacy=outputs/channel_main_2b4b_v0

Each arm is an output directory containing ``channel_results.csv``. The script

1. reports, per method, the accuracy, its 95% CI, the transmitted symbol cost,
   symbols per correct answer (efficiency), capacity-outage compliance, TRR and
   (when recorded) the damage-triggered restoration rate;
2. runs **paired** McNemar tests between arms (same sample/channel/seed cell) so
   the effect of the receiver protocol is measured on identical payloads;
3. runs paired McNemar tests between methods inside an arm (proposed vs the
   strongest baselines), which is the honest way to compare two compressors that
   face the same reads.

Nothing here loads a model: it is pure CSV analysis.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

# Methods that must always be shown even if an arm is a subset run.
KEY_METHODS = ("full", "plugin_capacity", "proposed", "adaptive_sc", "tokcom_uep",
               "answer_loo", "answer_attn", "answer_loo_uep")
PAIRINGS = (("proposed", "adaptive_sc"), ("proposed", "plugin_capacity"),
            ("proposed", "full"), ("proposed", "tokcom_uep"),
            # Task-aware/oracle signals: same scheduler and budget as proposed, so a
            # win here means the *importance signal* (not the scheduler) is stronger.
            ("proposed", "answer_loo_uep"), ("proposed", "answer_attn"),
            ("proposed", "answer_loo"), ("answer_loo_uep", "adaptive_sc"),
            ("answer_loo_uep", "plugin_capacity"))
CELL_KEY = ["sample_id", "channel", "snr_db", "channel_seed", "method"]


def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial p-value at p=0.5 (McNemar's exact test)."""
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2.0 * tail)


def load_arm(path: Path) -> pd.DataFrame:
    csv = path / "channel_results.csv"
    if not csv.exists():
        raise SystemExit(f"missing {csv}")
    frame = pd.read_csv(csv)
    frame["arm"] = path.name
    return frame


def cluster_ci95(per_sample: pd.Series, draws: int = 2000, seed: int = 0) -> float:
    """95% CI of the accuracy, bootstrapping over *questions* (sample_id).

    The naive binomial CI over observations is far too narrow here: every question
    contributes many correlated observations (3 channels x 2 SNRs x 3 seeds), so
    the effective sample size is the number of questions, not the number of rows.
    """
    import random

    values = per_sample.to_numpy(dtype=float)
    if values.size < 2:
        return float("nan")
    rng = random.Random(seed)
    n = values.size
    means = []
    for _ in range(draws):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * draws)]
    hi = means[min(draws - 1, int(0.975 * draws))]
    return (hi - lo) / 2


def method_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in frame.groupby("method"):
        acc = float(group["correct"].mean())
        symbols = float(group["transmitted_symbols"].mean())
        per_sample = group.groupby("sample_id")["correct"].mean()
        row = {
            "method": method,
            "observations": len(group),
            "questions": int(per_sample.size),
            "accuracy": acc,
            "ci95": cluster_ci95(per_sample),
            "symbols": symbols,
            "acc_per_1k_symbols": acc / symbols * 1000 if symbols else float("nan"),
            "outage_rate": float(group["capacity_outage"].mean()),
            "plugin_outage": float(group["plugin_outage"].mean()),
            "mean_TRR": float(group["TRR"].mean()),
            "original_tokens": float(group["original_tokens"].mean()),
        }
        if "restored" in group.columns:
            row["restore_rate"] = float(group["restored"].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("accuracy", ascending=False)


def pair_and_test(a: pd.Series, b: pd.Series) -> dict:
    """Paired McNemar on two boolean series aligned by their shared index."""
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    only_a = int(((joined.a == 1) & (joined.b == 0)).sum())
    only_b = int(((joined.a == 0) & (joined.b == 1)).sum())
    return {"pairs": int(len(joined)), "a_wins": only_a, "b_wins": only_b,
            "a_acc": float(joined.a.mean()) if len(joined) else float("nan"),
            "b_acc": float(joined.b.mean()) if len(joined) else float("nan"),
            "p_exact": binom_two_sided(min(only_a, only_b), only_a + only_b)}


def paired_across_arms(frame_a: pd.DataFrame, frame_b: pd.DataFrame, method: str) -> dict:
    """Same method, identical sample/channel/seed cells, two receiver arms."""
    a = frame_a[frame_a["method"] == method].set_index(CELL_KEY)["correct"]
    b = frame_b[frame_b["method"] == method].set_index(CELL_KEY)["correct"]
    return pair_and_test(a, b)


def paired_within_arm(frame: pd.DataFrame, lhs: str, rhs: str) -> dict:
    """Two methods on the same cells inside one arm (method is not part of the key)."""
    key = [c for c in CELL_KEY if c != "method"]
    a = frame[frame["method"] == lhs].set_index(key)["correct"]
    b = frame[frame["method"] == rhs].set_index(key)["correct"]
    return pair_and_test(a, b)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", required=True,
                        help="name=output_dir pairs, first arm is the reference")
    parser.add_argument("--filter-manifest", default=None,
                        help="keep only samples listed in this manifest (e.g. a run that "
                             "covered more questions than the filtered main set)")
    parser.add_argument("--json", default=None, help="optional JSON dump path")
    args = parser.parse_args()

    keep: set[str] | None = None
    if args.filter_manifest:
        import json as _json

        rows = [ _json.loads(line) for line in
                 Path(args.filter_manifest).read_text(encoding="utf-8").splitlines() if line.strip() ]
        keep = {str(r["sample_id"]) for r in rows}
        print(f"[filter] keeping the {len(keep)} questions of {args.filter_manifest}")

    arms: dict[str, pd.DataFrame] = {}
    for spec in args.arms:
        name, _, path = spec.partition("=")
        frame = load_arm(Path(path))
        if keep is not None:
            frame = frame[frame["sample_id"].astype(str).isin(keep)]
        arms[name] = frame
        print(f"[load] {name}: {len(arms[name])} observations "
              f"({frame['sample_id'].nunique()} questions) from {path}/channel_results.csv")

    dump = {}
    for name, frame in arms.items():
        table = method_table(frame)
        dump[name] = table.to_dict(orient="records")
        proto = (", ".join(sorted(frame["receiver_protocol"].dropna().unique()))
                 if "receiver_protocol" in frame.columns else "n/a")
        print(f"\n=== arm {name} (receiver protocol: {proto}) ===")
        has_restore = "restore_rate" in table.columns
        head = (f"{'method':<18}{'acc':>8}{'±95%':>7}{'symbols':>9}"
                f"{'acc/1k':>8}{'outage':>8}{'TRR':>6}"
                + (f"{'restore':>9}" if has_restore else ""))
        print(head)
        for _, row in table.iterrows():
            line = (f"{row['method']:<18}{row['accuracy']*100:>7.1f}%"
                    f"{row['ci95']*100:>6.1f}{row['symbols']:>9.0f}"
                    f"{row['acc_per_1k_symbols']:>8.1f}{row['outage_rate']:>8.3f}"
                    f"{row['mean_TRR']:>6.2f}")
            if has_restore:
                line += f"{row['restore_rate']:>9.2f}"
            print(line)

    names = list(arms)
    if len(names) >= 2:
        ref, other = names[0], names[1]
        print(f"\n=== paired McNemar: {other} vs {ref} "
              f"(identical sample/channel/seed cells) ===")
        print(f"{'method':<18}{'pairs':>7}{ref+'_win':>12}{other+'_win':>13}"
              f"{ref+'_acc':>11}{other+'_acc':>12}{'p':>10}")
        rows = []
        for method in sorted(set(arms[ref]["method"]) & set(arms[other]["method"])):
            res = {"method": method, **paired_across_arms(arms[ref], arms[other], method)}
            rows.append(res)
            print(f"{method:<18}{res['pairs']:>7}{res['a_wins']:>12}{res['b_wins']:>13}"
                  f"{res['a_acc']*100:>10.1f}%{res['b_acc']*100:>11.1f}%"
                  f"{res['p_exact']:>10.2g}")
        dump["paired_arms"] = rows

    for name, frame in arms.items():
        print(f"\n=== paired McNemar inside arm {name} ===")
        rows = []
        for lhs, rhs in PAIRINGS:
            have = set(frame["method"])
            if lhs not in have or rhs not in have:
                continue
            res = paired_within_arm(frame, lhs, rhs)
            rows.append({"pair": f"{lhs} vs {rhs}", **res})
            print(f"{lhs} vs {rhs:<16} pairs={res['pairs']:>5} "
                  f"wins={res['a_wins']:>4}:{res['b_wins']:<4} "
                  f"acc={res['a_acc']*100:.1f}% vs {res['b_acc']*100:.1f}% "
                  f"p={res['p_exact']:.2g}")
        dump[f"paired_within_{name}"] = rows

    if args.json:
        Path(args.json).write_text(json.dumps(dump, indent=2), encoding="utf-8")
        print(f"\n[done] JSON written to {args.json}")


if __name__ == "__main__":
    main()
