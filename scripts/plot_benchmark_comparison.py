#!/usr/bin/env python
"""Comparison plots against published prompt-compression / semantic-communication baselines.

All methods except the two references share the *same* reliable budget of Eq. (6):

    full                Prompt-SC [5]: whole prompt, budget ignored (uncompressed ref.)
    plugin_capacity     internal baseline (1): Eq. (6) replaced by plug-in capacity.
                        It buys accuracy with a *larger* budget and therefore
                        violates the outage target about half of the time, so the
                        outage rate is reported next to it in every table.
    proposed            Algorithm 1: Lagrangian + certified rounding + refinement
    adaptive_sc         Adaptive-SC [8]: equal protection for every token, but both
                        the protection level and the token count adapt to the channel
    llmlingua           Jiang et al., ACL 2023  -- coarse-to-fine PPL pruning
    longllmlingua       question-aware contrastive PPL (LongLLMLingua, 2024)
    selective_context   Li et al., EMNLP 2023   -- self-information top-k

The ideal channel and the degenerate baselines (random / truncate / fixed_H) are
deliberately excluded: the comparison is run over the three fading channels only.

Usage
-----
    python scripts/plot_benchmark_comparison.py \
        --run outputs/channel_8b_published/summary.json \
        --out-dir outputs/figures
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Fading channels only -- the ideal channel is excluded on purpose.
CH_ORDER = ["rayleigh", "rician", "frequency_selective"]
CH_LABEL = {"rayleigh": "Rayleigh", "rician": "Rician",
            "frequency_selective": "Freq.-sel."}
FADE = CH_ORDER

# (series key, method key, source run, label, colour)
SERIES = [
    ("full", "full", "run", "Prompt-SC [5] (uncompressed)", "#8d8d8d"),
    ("plugin_capacity", "plugin_capacity", "run",
     "Capacity oracle (internal 1)", "#b0bec5"),
    ("proposed", "proposed", "run", "Proposed", "#c62828"),
    ("adaptive_sc", "adaptive_sc", "run", "Adaptive-SC [8]", "#1e88e5"),
    ("answer_loo", "answer_loo", "run", "Answer-LOO (gold std.)", "#455a64"),
    ("answer_loo_uep", "answer_loo_uep", "run", "LOO + our scheduler", "#00acc1"),
    ("answer_attn", "answer_attn", "run", "Answer-attention", "#ffb300"),
    ("tokcom_uep", "tokcom_uep", "run", "TokCom-UEP (2025)", "#d81b60"),
    ("llmlingua2", "llmlingua2", "run", "LLMLingua-2 (2024)", "#7b1fa2"),
    ("efpc", "efpc", "run", "EFPC-style (2025)", "#6d4c41"),
    ("tokcom", "tokcom", "run", "TokCom (2025)", "#5e35b1"),
    ("llmlingua", "llmlingua", "run", "LLMLingua [16]", "#9c27b0"),
    ("longllmlingua", "longllmlingua", "run", "LongLLMLingua", "#ef6c00"),
    ("selective_context", "selective_context", "run", "Selective-Context", "#00897b"),
]


def load(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def index(summary):
    return {(c["channel"], float(c["snr_db"]), c["method"]): c
            for c in summary["conditions"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/channel_8b_published/summary.json")
    ap.add_argument("--out-dir", default="outputs/figures")
    ap.add_argument("--title", default="V2X-QA over time-varying channels "
                                       "(8B transmitter / 4B receiver)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    runs = {"run": index(load(args.run))}
    series = [s for s in SERIES if s[2] in runs]

    snrs = [0.0, 15.0]
    groups = [(ch, s) for ch in CH_ORDER for s in snrs]
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(groups))

    # ---------------------------------------------------------------- Fig. 1
    fig, ax = plt.subplots(figsize=(14, 5.8))
    for i, (_k, mkey, src, label, color) in enumerate(series):
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
    ax.set_xticklabels(["%s\nSNR %d dB" % (CH_LABEL[ch], int(s)) for ch, s in groups])
    ax.set_ylabel("Answer accuracy (%)")
    ax.set_ylim(0, 108)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(ncol=4, fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, -0.11), frameon=False)
    ax.set_title("%s\nproposed versus published baselines "
                 "(60 samples x 3 channel seeds, identical Eq. (6) budget, 95%% CI)"
                 % args.title)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig_published_accuracy.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------- Fig. 2 accuracy / cost
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    xg = np.arange(len(snrs))
    for ax, (metric, scale, ylabel, title) in zip(axes, [
        ("accuracy", 100.0, "Answer accuracy (%)",
         "(a) Accuracy (mean over fading channels)"),
        ("mean_symbols", 1.0, "Transmitted symbols / sample",
         "(b) Transmission cost"),
    ]):
        for i, (_k, mkey, src, label, color) in enumerate(series):
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
    fig.savefig(os.path.join(args.out_dir, "fig_published_accuracy_cost.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------- Fig. 3 efficiency frontier
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
    for ax, snr in zip(axes, snrs):
        for _k, mkey, src, label, color in series:
            pts = [(runs[src][(ch, snr, mkey)]["mean_symbols"],
                    runs[src][(ch, snr, mkey)]["accuracy"] * 100.0)
                   for ch in FADE if (ch, snr, mkey) in runs[src]]
            if not pts:
                continue
            sx = float(np.mean([p[0] for p in pts]))
            sy = float(np.mean([p[1] for p in pts]))
            ax.scatter(sx, sy, s=120, color=color, edgecolor="black",
                       linewidth=0.6, zorder=3)
            ax.annotate(label, (sx, sy), textcoords="offset points",
                        xytext=(8, 5), fontsize=8.5, zorder=4)
        ax.set_xlabel("Transmitted symbols / sample")
        ax.set_title("SNR %d dB (mean over fading channels)" % int(snr))
        ax.grid(linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Answer accuracy (%)")
    fig.suptitle("Efficiency frontier: accuracy versus transmission cost", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig_published_frontier.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------- CSV table
    csv_path = os.path.join(args.out_dir, "published_comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["channel", "snr_db", "variant", "accuracy", "ci95",
                    "mean_symbols", "mean_TRR", "token_erasure_rate",
                    "capacity_outage_rate", "plugin_outage_rate"])
        for ch, s in groups:
            for _k, mkey, src, label, _c in series:
                cond = runs[src].get((ch, s, mkey))
                if not cond:
                    continue
                w.writerow([ch, int(s), label, round(cond["accuracy"], 4),
                            round(cond["accuracy_ci95"], 4),
                            round(cond["mean_symbols"], 1),
                            round(cond["mean_TRR"], 4),
                            round(cond["mean_token_erasure_rate"], 4),
                            round(cond["capacity_outage_rate"], 4),
                            round(cond["plugin_outage_rate"], 4)])

    # -------------------------------------------------------- stdout summary
    head = "%-14s %-32s %7s %9s %7s %9s" % ("condition", "variant", "acc%",
                                             "symbols", "TRR", "outage")
    print(head)
    print("-" * len(head))
    for ch, s in groups:
        for _k, mkey, src, label, _c in series:
            cond = runs[src].get((ch, s, mkey))
            if not cond:
                continue
            outage = (cond["plugin_outage_rate"] if mkey == "plugin_capacity"
                      else cond["capacity_outage_rate"])
            print("%-14s %-32s %7.1f %9.0f %7.3f %9.3f" % (
                "%s/%d" % (CH_LABEL[ch], int(s)), label, cond["accuracy"] * 100.0,
                cond["mean_symbols"], cond["mean_TRR"], outage))
    print("\nsaved -> %s" % args.out_dir)


if __name__ == "__main__":
    main()
