#!/usr/bin/env python
"""Comparison plots for the V2X-QA channel experiment (proposed budget ablation).

Two runs are compared:

  * conservative budget -- Eq. (6) reliable capacity c_eps  -> "Proposed (cons)"
  * maximum budget      -- plug-in / genie capacity         -> "Proposed (max)"

By construction "Proposed (max)" coincides with the "Capacity oracle" baseline:
both drive the very same scheduler (Algorithm 1) with the very same budget, so
the two bars are expected to be identical.

Usage
-----
    python scripts/plot_channel_comparison.py \
        --cons outputs/channel_8b/summary.json \
        --max  outputs/channel_8b_maxbudget/summary.json \
        --out-dir outputs/figures
"""

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CH_ORDER = ["ideal", "rayleigh", "rician", "frequency_selective"]
CH_LABEL = {
    "ideal": "Ideal",
    "rayleigh": "Rayleigh",
    "rician": "Rician",
    "frequency_selective": "Freq.-sel.",
}
FADE = ["rayleigh", "rician", "frequency_selective"]

# (series key, method key in summary.json, source run, label, colour)
SERIES = [
    ("full", "full", "max", "Full (uncompressed)", "#8d8d8d"),
    ("plugin_capacity", "plugin_capacity", "max", "Capacity oracle", "#b0bec5"),
    ("proposed_max", "proposed", "max", "Proposed (max budget)", "#c62828"),
    ("proposed_cons", "proposed", "cons", "Proposed (conservative)", "#fb8c00"),
    ("fixed_H", "fixed_H", "max", "Fixed-H", "#1e88e5"),
    ("random", "random", "max", "Random", "#43a047"),
]


def load(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def index(summary):
    return {(c["channel"], float(c["snr_db"]), c["method"]): c
            for c in summary["conditions"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cons", default="outputs/channel_8b/summary.json",
                    help="summary.json of the conservative-budget run")
    ap.add_argument("--max", dest="run_max",
                    default="outputs/channel_8b_maxbudget/summary.json",
                    help="summary.json of the maximum-budget run")
    ap.add_argument("--out-dir", default="outputs/figures")
    ap.add_argument("--title", default="V2X-QA over time-varying channels "
                                       "(8B transmitter / 4B receiver)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    runs = {"cons": index(load(args.cons)), "max": index(load(args.run_max))}

    snrs = [0.0, 15.0]
    groups = [(ch, s) for ch in CH_ORDER for s in snrs]
    n = len(SERIES)
    width = 0.8 / n
    x = np.arange(len(groups))

    # ---------------------------------------------------------------- Fig. 1
    fig, ax = plt.subplots(figsize=(14, 5.6))
    for i, (_key, mkey, src, label, color) in enumerate(SERIES):
        acc, err = [], []
        for ch, s in groups:
            cond = runs[src].get((ch, s, mkey))
            acc.append(cond["accuracy"] * 100.0 if cond else np.nan)
            err.append(cond["accuracy_ci95"] * 100.0 if cond else 0.0)
        pos = x + (i - (n - 1) / 2.0) * width
        ax.bar(pos, acc, width, yerr=err, label=label, color=color,
               edgecolor="black", linewidth=0.5, capsize=2,
               error_kw={"elinewidth": 0.8, "ecolor": "#333333"})
    ax.set_xticks(x)
    ax.set_xticklabels(["%s\nSNR %d dB" % (CH_LABEL[ch], int(s))
                        for ch, s in groups])
    ax.set_ylabel("Answer accuracy (%)")
    ax.set_ylim(0, 108)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(ncol=6, fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, -0.10), frameon=False)
    ax.set_title("%s\naccuracy by method (60 samples x 3 channel seeds, "
                 "95%% CI)" % args.title)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig_accuracy.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------- Fig. 2
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    panels = [
        ("accuracy", 100.0, "Answer accuracy (%)",
         "(a) Accuracy (mean over fading channels)", True),
        ("mean_symbols", 1.0, "Transmitted symbols / sample",
         "(b) Transmission cost", False),
    ]
    xg = np.arange(len(snrs))
    for ax, (metric, scale, ylabel, title, _pct) in zip(axes, panels):
        for i, (_key, mkey, src, label, color) in enumerate(SERIES):
            vals = []
            for s in snrs:
                vs = [runs[src][(ch, s, mkey)][metric] for ch in FADE
                      if (ch, s, mkey) in runs[src]]
                vals.append(np.mean(vs) * scale if vs else np.nan)
            pos = xg + (i - (n - 1) / 2.0) * width
            ax.bar(pos, vals, width, label=label, color=color,
                   edgecolor="black", linewidth=0.5)
        ax.set_xticks(xg)
        ax.set_xticklabels(["SNR %d dB" % int(s) for s in snrs])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig_accuracy_cost.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------- CSV table
    csv_path = os.path.join(args.out_dir, "channel_comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["channel", "snr_db", "variant", "accuracy", "ci95",
                    "mean_symbols", "mean_TRR", "token_erasure_rate"])
        for ch, s in groups:
            for key, mkey, src, label, _c in SERIES:
                cond = runs[src].get((ch, s, mkey))
                if not cond:
                    continue
                w.writerow([ch, int(s), label,
                            round(cond["accuracy"], 4),
                            round(cond["accuracy_ci95"], 4),
                            round(cond["mean_symbols"], 1),
                            round(cond["mean_TRR"], 4),
                            round(cond["mean_token_erasure_rate"], 4)])

    # -------------------------------------------------------- stdout summary
    print("conditions: %s" % ", ".join("%s/SNR%d" % (CH_LABEL[c], int(s))
                                       for c, s in groups))
    header = "%-14s %-26s %8s %8s %8s" % ("condition", "variant", "acc%",
                                          "symbols", "TRR")
    print(header)
    print("-" * len(header))
    for ch, s in groups:
        for key, mkey, src, label, _c in SERIES:
            cond = runs[src].get((ch, s, mkey))
            if not cond:
                continue
            print("%-14s %-26s %8.1f %8.0f %8.3f" % (
                "%s/%d" % (CH_LABEL[ch], int(s)), label,
                cond["accuracy"] * 100.0, cond["mean_symbols"],
                cond["mean_TRR"]))
    print("\nsaved -> %s" % args.out_dir)


if __name__ == "__main__":
    main()
