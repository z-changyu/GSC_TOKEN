#!/usr/bin/env python
"""Paper figure: cost-accuracy front of the scheduler (main-line 2B run).

Panel (a): answer accuracy versus transmitted symbols. The thirteen baselines are
scatter points taken from the main run; the proposed scheduler is a curve traced
by handing it artificial symbol budgets, with the iso-cost fill points and the
uniform-protection endpoints marked separately. A uniform-level point is only
admissible when the whole report fits in the reliable budget, so those markers
cover the feasible draws only (documented in the caption).

Panel (b): composition of the transmitted tokens by protection level as a
function of the budget the scheduler is handed, with accuracy on the twin axis.
This answers "where would extra budget go" -- directly the iso-cost discussion in
the paper.

Monochrome, vector output (IEEE two-column figure*).

Usage
-----
    python scripts/plot_pareto_figure.py \
        --pareto outputs/pareto_main_2b4b/summary.json \
        --baseline-csv outputs/channel_main_2b4b_fair/channel_results.csv \
        --out paper/images/fig_pareto.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import json  # noqa: E402

# baselines drawn with their own marker; everything else shares the "other" style
KEY_STYLE = {
    "full": ("Prompt-SC (uncompressed)", "s", 7.0, "none"),
    "plugin_capacity": ("Capacity oracle (internal)", "D", 7.0, "none"),
    "adaptive_sc": ("Adaptive-SC [8]", "^", 8.0, "0.55"),
    "tokcom_uep": ("TokCom-UEP", "v", 8.0, "none"),
    "answer_loo_uep": ("LOO-UEP", "P", 8.0, "none"),
    "answer_attn": ("Answer-attention", "X", 8.0, "none"),
    "answer_loo": ("Answer-LOO (gold standard)", "p", 8.0, "none"),
}
LEVEL_COLORS = {"rH": "black", "rM": "0.42", "rL": "0.70"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pareto", default="outputs/pareto_main_2b4b/summary.json")
    parser.add_argument("--baseline-csv",
                        default="outputs/channel_main_2b4b_fair/channel_results.csv")
    parser.add_argument("--channels", nargs="*",
                        default=["rayleigh", "rician", "frequency_selective"])
    parser.add_argument("--out", default="paper/images/fig_pareto.pdf")
    parser.add_argument("--note", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.pareto) as handle:
        summary = json.load(handle)
    strategies = summary["strategies"]
    budgets = [float(b) for b in summary["budgets"]]

    frame = pd.read_csv(args.baseline_csv)
    if args.channels:
        frame = frame[frame.channel.isin(args.channels)]
    base = (frame.groupby("method")
            .agg(symbols=("transmitted_symbols", "mean"),
                 accuracy=("correct", "mean"), n=("correct", "size"))
            .reset_index())

    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.7,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "legend.frameon": False})
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.7))
    ax = axes[0]

    # ------------------------------------------------------- (a) baselines
    # ``proposed`` is drawn as the sweep curve, never as a scatter point
    others = base[~base.method.isin(KEY_STYLE) & (base.method != "proposed")]
    ax.scatter(others.symbols, others.accuracy * 100.0, s=22, marker="o",
               facecolors="none", edgecolors="0.55", linewidths=0.8, zorder=3)
    for method, (label, marker, size, face) in KEY_STYLE.items():
        row = base[base.method == method]
        if row.empty:
            continue
        ax.scatter(row.symbols, row.accuracy * 100.0, s=size ** 2 * 0.09,
                   marker=marker, facecolors=face, edgecolors="black",
                   linewidths=0.8, zorder=4)
    handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=4.5,
                      markerfacecolor="none", markeredgecolor="0.55",
                      label="other baselines (%d)" % len(others))]
    for method, (label, marker, size, face) in KEY_STYLE.items():
        handles.append(Line2D([0], [0], marker=marker, linestyle="none",
                              markersize=4.5, markerfacecolor=face or "none",
                              markeredgecolor="black", label=label))

    # ------------------------------------------------------- (a) sweep
    sweep = sorted(((strategies["budget%d" % int(b)]["mean_symbols"],
                     strategies["budget%d" % int(b)]["accuracy"] * 100.0)
                    for b in budgets), key=lambda p: p[0])
    ax.plot([p[0] for p in sweep], [p[1] for p in sweep], color="black",
            linewidth=1.5, marker="o", markersize=3.6, markeredgecolor="black",
            zorder=6, label="Proposed (artificial budget sweep)")
    fills = sorted(((strategies[name]["mean_symbols"],
                     strategies[name]["accuracy"] * 100.0)
                    for name in strategies if name.startswith("fill_")),
                   key=lambda p: p[0])
    if fills:
        ax.plot([p[0] for p in fills], [p[1] for p in fills], color="black",
                linewidth=1.0, linestyle="--", marker="*", markersize=7.5,
                zorder=6, label="Proposed (iso-cost fill points)")
    uniforms = sorted(((strategies[name]["mean_symbols"],
                        strategies[name]["accuracy"] * 100.0)
                       for name in strategies if name.startswith("uniform_")),
                      key=lambda p: p[0])
    if uniforms:
        ax.scatter([p[0] for p in uniforms], [p[1] for p in uniforms], s=42,
                   marker="^", facecolors="none", edgecolors="black", linewidths=0.9,
                   zorder=5, label="Proposed (single uniform level,\nfeasible draws only)")
    handles += ax.get_legend_handles_labels()[0][-3:]

    # per-method anchor for Adaptive-SC spend
    targets = summary.get("iso_cost_targets", {})
    if "adaptive" in targets:
        ax.axvline(targets["adaptive"], color="0.45", linestyle=":", linewidth=0.8,
                   zorder=1)

    ax.set_xlabel("Transmitted symbols per sample")
    ax.set_ylabel("Answer accuracy (%)")
    ax.set_title("(a) Cost-accuracy front", fontsize=8.5)
    ax.grid(linestyle="--", linewidth=0.4, alpha=0.4)
    ax.set_axisbelow(True)
    plateau = [p for p in sweep if p[0] >= max(q[0] for q in sweep) - 1e-6]
    if plateau and len(sweep) > 2:
        ax.annotate("extra budget\nspent on nothing",
                    xy=(plateau[-1][0], plateau[-1][1]),
                    xytext=(0.42, 0.30), textcoords="axes fraction",
                    fontsize=5.6, color="0.25", ha="left",
                    arrowprops=dict(arrowstyle="->", linewidth=0.6, color="0.35"))

    # ------------------------------------------------------- (b) composition
    ax = axes[1]
    x = np.arange(len(budgets))
    bottom = np.zeros(len(budgets))
    for level in ("rL", "rM", "rH"):
        counts = np.array([strategies["budget%d" % int(b)]["mean_kept_%s" % level]
                           for b in budgets])
        ax.bar(x, counts, 0.72, bottom=bottom, color=LEVEL_COLORS[level],
               edgecolor="black", linewidth=0.5,
               label={"rL": "$r_L$ (16 symb)", "rM": "$r_M$ (32)",
                      "rH": "$r_H$ (48)"}[level])
        bottom += counts
    ax.set_xticks(x)
    ax.set_xticklabels(["%d" % b for b in budgets], fontsize=6.0, rotation=45)
    ax.set_xlabel("Symbol budget handed to the scheduler")
    ax.set_ylabel("Transmitted tokens by level")
    ax.grid(linestyle="--", linewidth=0.4, alpha=0.4, axis="y")
    ax.set_axisbelow(True)

    ax2 = ax.twinx()
    acc = [strategies["budget%d" % int(b)]["accuracy"] * 100.0 for b in budgets]
    ax2.plot(x, acc, color="black", linewidth=1.3, marker="s", markersize=3.2,
             linestyle="-", label="accuracy")
    ax2.set_ylabel("Answer accuracy (%)")
    ax2.set_ylim(min(acc) - 4, 100)
    ax.set_title("(b) Where the budget goes", fontsize=8.5)
    handles += ([Patch(facecolor=LEVEL_COLORS[k], edgecolor="black", linewidth=0.5,
                       label={"rL": "$r_L$ (16 symb)", "rM": "$r_M$ (32)",
                              "rH": "$r_H$ (48)"}[k])
                 for k in ("rH", "rM", "rL")] +
                [Line2D([0], [0], color="black", marker="s", markersize=3.2,
                        label="accuracy (panel b)")])

    if args.note:
        axes[0].text(0.02, 0.02, args.note, transform=axes[0].transAxes,
                     fontsize=5.4, color="0.35", va="bottom")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.4)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.34),
               ncol=5, fontsize=5.6, handletextpad=0.4, columnspacing=1.0,
               labelspacing=0.30)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    print("wrote %s" % out)
    print("sweep (symbols, acc):")
    for symbols, accuracy in sweep:
        print("   %7.0f  %5.1f%%" % (symbols, accuracy))
    print("iso-cost targets: %s" % {k: round(v, 0) for k, v in targets.items()})


if __name__ == "__main__":
    main()
