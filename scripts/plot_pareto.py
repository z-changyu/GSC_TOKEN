#!/usr/bin/env python
"""Cost-accuracy Pareto figure for the proposed scheduler.

Panel (a): accuracy versus transmitted symbols. The proposed scheduler is drawn
as a curve (artificial-budget sweep), with the equality-constrained fill points
and the uniform-level endpoints marked separately; the thirteen baselines are
scatter points read from the v6 main run, plus the uncompressed reference.

Panel (b): budget responsiveness -- the budget handed to the scheduler versus
the symbols it actually spends, which shows the saturation that makes the
iso-cost question moot: beyond ~2000 symbols the optimum refuses to spend more.

Caveat plotted explicitly: a uniform-level point is only admissible when the
whole report fits in the reliable budget, so those markers cover the feasible
draws only; the budget sweep and the fill points cover every draw.

Usage
-----
    python scripts/plot_pareto.py --pareto outputs/pareto/summary.json \
        --baseline-csv outputs/channel_8b_final14_v6final/channel_results.csv \
        --out-dir outputs/pareto
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CH_ORDER = ["rayleigh", "rician", "frequency_selective"]
CH_LABEL = {"rayleigh": "Rayleigh", "rician": "Rician",
            "frequency_selective": "Freq.-selective"}
STYLE = [
    ("full", "Prompt-SC [5] (uncompressed)", "#8d8d8d", "s"),
    ("plugin_capacity", "Capacity oracle (internal)", "#b0bec5", "P"),
    ("adaptive_sc", "Adaptive-SC [8]", "#1e88e5", "D"),
    ("proposed", "Proposed", "#c62828", "o"),
    ("answer_loo_uep", "LOO + our scheduler", "#00acc1", "h"),
    ("answer_attn", "Answer-attention", "#ffb300", "8"),
    ("tokcom_uep", "TokCom-UEP (2025)", "#d81b60", "v"),
    ("llmlingua2", "LLMLingua-2 (2024)", "#7b1fa2", "^"),
    ("efpc", "EFPC-style (2025)", "#6d4c41", "<"),
    ("tokcom", "TokCom (2025)", "#5e35b1", ">"),
    ("longllmlingua", "LongLLMLingua", "#ef6c00", "X"),
    ("llmlingua", "LLMLingua [16]", "#9c27b0", "*"),
    ("selective_context", "Selective-Context", "#00897b", "+"),
    ("answer_loo", "Answer-LOO (gold std.)", "#455a64", "p"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pareto", default="outputs/pareto/summary.json")
    ap.add_argument("--baseline-csv",
                    default="outputs/channel_8b_final14_v6final/channel_results.csv")
    ap.add_argument("--out-dir", default="outputs/pareto")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--note", default="15 dB, 60 samples x 3 channels x 3 channel seeds")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.pareto, encoding="utf-8") as handle:
        pareto = json.load(handle)
    strategies = pareto["strategies"]
    budgets = [float(b) for b in pareto["budgets"]]
    snr_db = [float(v) for v in pareto["snr_db"]]
    targets = {k: float(v) for k, v in pareto["iso_cost_targets"].items()}

    frame = pd.read_csv(args.baseline_csv)
    frame = frame[frame.snr_db.isin(snr_db)] if snr_db else frame

    sweep = [(strategies[f"budget{int(b)}"]["mean_symbols"],
              strategies[f"budget{int(b)}"]["accuracy"] * 100.0)
             for b in budgets if f"budget{int(b)}" in strategies]
    sweep.sort()
    fills = [(strategies[name]["mean_symbols"], strategies[name]["accuracy"] * 100.0,
              name.replace("fill_", ""))
             for name in sorted(strategies) if name.startswith("fill_")]
    uniforms = [(strategies[name]["mean_symbols"], strategies[name]["accuracy"] * 100.0,
                 name.replace("uniform_", ""))
                for name in sorted(strategies) if name.startswith("uniform_")]

    # The uniform-level markers only cover draws where the whole report fits
    # the reliable budget, so they sit above the sweep. To keep the comparison
    # honest the plateau point is recomputed on exactly the same subset.
    same_subset = []
    results_path = os.path.join(os.path.dirname(os.path.abspath(args.pareto)),
                                "results.csv")
    if os.path.exists(results_path):
        res = pd.read_csv(results_path)
        key = ["sample_id", "channel", "seed"]
        feasible = res[res.strategy == "uniform_rL"][key]
        plateau = res[res.strategy == f"budget{int(max(budgets))}"].merge(feasible, on=key)
        if not plateau.empty:
            same_subset.append((plateau.transmitted_symbols.mean(),
                                plateau.correct.mean() * 100.0))

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6))

    ax = axes[0]
    for key, label, color, marker in STYLE:
        rows = frame[frame.method == key]
        if rows.empty:
            continue
        ax.scatter([rows.transmitted_symbols.mean()], [rows.correct.mean() * 100.0],
                   s=110, color=color, marker=marker, edgecolor="black",
                   linewidth=0.6, zorder=3, label=label)
    ax.plot([p[0] for p in sweep], [p[1] for p in sweep], color="#c62828",
            linewidth=2.2, marker="o", markersize=8, markeredgecolor="black",
            markeredgewidth=0.6, zorder=4, label="Proposed (budget sweep)")
    ax.plot([p[0] for p in fills], [p[1] for p in fills], color="#c62828",
            linewidth=1.4, linestyle="--", marker="*", markersize=14,
            markeredgecolor="black", markeredgewidth=0.5, zorder=4,
            label="Proposed (iso-cost fill)")
    ax.plot([p[0] for p in uniforms], [p[1] for p in uniforms], color="#c62828",
            linewidth=0.0, marker="^", markersize=10, markeredgecolor="black",
            markeredgewidth=0.6, zorder=4, label="Proposed (uniform level, feasible draws)")
    if same_subset:
        ax.plot([p[0] for p in same_subset], [p[1] for p in same_subset],
                color="#c62828", linewidth=0.0, marker="o", markersize=11,
                markerfacecolor="none", markeredgecolor="#c62828",
                markeredgewidth=2.0, zorder=4,
                label="Proposed plateau on the same feasible subset")
    for spend, acc, name in fills:
        ax.annotate("fill=%s" % name, (spend, acc), textcoords="offset points",
                    xytext=(6, -12), fontsize=7.5, color="#7f0000")
    if "adaptive" in targets:
        ax.axvline(targets["adaptive"], color="#1e88e5", linestyle=":", alpha=0.7)
        ax.annotate("Adaptive-SC spend", (targets["adaptive"], ax.get_ylim()[0] + 2),
                    rotation=90, fontsize=8, color="#1e88e5", va="bottom", ha="right")
    ax.set_xlabel("Transmitted symbols / sample")
    ax.set_ylabel("Answer accuracy (%)")
    ax.set_title("(a) Cost-accuracy Pareto front")
    ax.grid(linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, ncol=1, loc="lower right", frameon=True)

    ax = axes[1]
    given = [b for b in budgets if f"budget{int(b)}" in strategies]
    spent = [strategies[f"budget{int(b)}"]["mean_symbols"] for b in given]
    accs = [strategies[f"budget{int(b)}"]["accuracy"] * 100.0 for b in given]
    ax.plot(given, spent, color="#c62828", linewidth=2.2, marker="o",
            markersize=8, markeredgecolor="black", markeredgewidth=0.6,
            label="symbols actually spent")
    ax.plot(given, given, color="#9e9e9e", linestyle="--", linewidth=1.4,
            label="budget fully used (y = x)")
    ax.set_xlabel("Symbol budget handed to the scheduler")
    ax.set_ylabel("Symbols actually spent")
    for value, spend, acc in zip(given, spent, accs):
        ax.annotate("%.1f%%" % acc, (value, spend), textcoords="offset points",
                    xytext=(6, 6), fontsize=8)
    ax.set_title("(b) Extra budget buys nothing beyond ~2000 symbols")
    ax.grid(linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)

    fig.suptitle("Iso-cost evidence for the proposed scheduler -- %s" % args.note, y=1.0)
    fig.tight_layout()
    out = os.path.join(args.out_dir, "fig_pareto%s.png" % args.suffix)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved -> %s" % out)
    print("sweep points: %s" % ", ".join("(%.0f, %.1f%%)" % (p[0], p[1]) for p in sweep))
    print("fill points : %s" % ", ".join("(%s: %.0f sym, %.1f%%)" % (p[2], p[0], p[1])
                                         for p in fills))
    print("uniform pts : %s" % ", ".join("(%s: %.0f sym, %.1f%%)" % (p[2], p[0], p[1])
                                         for p in uniforms))


if __name__ == "__main__":
    main()
