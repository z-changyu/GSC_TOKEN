#!/usr/bin/env python
"""Line-chart comparison of the channel experiment: accuracy versus SNR.

One panel per fading channel, one line per method, 95% CI as error bars plus a
shaded band. A second figure shows each method's (cost, accuracy) operating
point at the two SNRs.

Usage
-----
    python scripts/plot_channel_lines.py \
        --run outputs/channel_8b_published_v3/summary.json \
        --out-dir outputs/figures
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CH_ORDER = ["rayleigh", "rician", "frequency_selective"]
CH_LABEL = {"rayleigh": "Rayleigh", "rician": "Rician",
            "frequency_selective": "Freq.-selective"}

# method key -> (label, colour, marker); legend keeps this order
STYLE = [
    ("full", "Prompt-SC [5] (uncompressed)", "#8d8d8d", "s"),
    ("plugin_capacity", "Capacity oracle (internal)", "#b0bec5", "P"),
    ("proposed", "Proposed", "#c62828", "o"),
    ("adaptive_sc", "Adaptive-SC [8]", "#1e88e5", "D"),
    ("answer_loo", "Answer-LOO (gold std.)", "#455a64", "p"),
    ("answer_loo_uep", "LOO + our scheduler", "#00acc1", "h"),
    ("answer_attn", "Answer-attention", "#ffb300", "8"),
    ("tokcom_uep", "TokCom-UEP (2025)", "#d81b60", "v"),
    ("llmlingua2", "LLMLingua-2 (2024)", "#7b1fa2", "^"),
    ("efpc", "EFPC-style (2025)", "#6d4c41", "<"),
    ("tokcom", "TokCom (2025)", "#5e35b1", ">"),
    ("longllmlingua", "LongLLMLingua", "#ef6c00", "X"),
    ("llmlingua", "LLMLingua [16]", "#9c27b0", "*"),
    ("selective_context", "Selective-Context", "#00897b", "+"),
    ("fixed_H", "Fixed-H", "#4dd0e1", "d"),
    ("truncate", "Truncate", "#827717", "1"),
    ("random", "Random", "#43a047", "2"),
]


def load(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def index(summary):
    return {(c["channel"], float(c["snr_db"]), c["method"]): c
            for c in summary["conditions"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/channel_8b_published_v3/summary.json")
    ap.add_argument("--out-dir", default="outputs/figures")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    runs = index(load(args.run))
    snrs = sorted({float(c["snr_db"]) for c in json.load(open(args.run, encoding="utf-8"))["conditions"]})
    methods = [key for key, _label, _color, _marker in STYLE
               if any((ch, s, key) in runs for ch in CH_ORDER for s in snrs)]

    # ------------------------------------------------ Fig 1: accuracy vs SNR
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
    for ax, ch in zip(axes, CH_ORDER):
        for key, label, color, marker in STYLE:
            if key not in methods:
                continue
            acc, err = [], []
            for s in snrs:
                cond = runs.get((ch, s, key))
                acc.append(cond["accuracy"] * 100.0 if cond else float("nan"))
                err.append(cond["accuracy_ci95"] * 100.0 if cond else 0.0)
            ax.errorbar(snrs, acc, yerr=err, label=label, color=color, marker=marker,
                        markersize=7, linewidth=2.0, capsize=4,
                        markeredgecolor="black", markeredgewidth=0.6)
            ax.fill_between(snrs, [a - e for a, e in zip(acc, err)],
                            [a + e for a, e in zip(acc, err)], color=color, alpha=0.10)
        ax.set_title(CH_LABEL[ch])
        ax.set_xticks(snrs)
        ax.set_xticklabels(["%d dB" % int(s) for s in snrs])
        ax.set_xlabel("SNR")
        ax.grid(linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Answer accuracy (%)")
    axes[0].set_ylim(0, 100)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, fontsize=9, loc="upper center",
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle("Accuracy versus SNR%s" % ((" -- " + args.note) if args.note else ""),
                y=1.0)
    fig.tight_layout()
    out1 = os.path.join(args.out_dir, "fig_lines_accuracy%s.png" % args.suffix)
    fig.savefig(out1, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------- Fig 2: cost-accuracy operating points
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), sharey=True)
    for ax, snr in zip(axes, snrs):
        for key, label, color, marker in STYLE:
            if key not in methods:
                continue
            pts = [(runs[(ch, snr, key)]["mean_symbols"],
                    runs[(ch, snr, key)]["accuracy"] * 100.0)
                   for ch in CH_ORDER if (ch, snr, key) in runs]
            if not pts:
                continue
            sx = float(sum(p[0] for p in pts) / len(pts))
            sy = float(sum(p[1] for p in pts) / len(pts))
            ax.scatter([sx], [sy], s=110, color=color, marker=marker,
                       edgecolor="black", linewidth=0.6, zorder=3, label=label)
            ax.annotate(label, (sx, sy), textcoords="offset points",
                        xytext=(8, 5), fontsize=8, zorder=4)
        ax.set_xlabel("Transmitted symbols / sample (mean over fading channels)")
        ax.set_title("SNR %d dB" % int(snr))
        ax.grid(linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Answer accuracy (%)")
    fig.suptitle("Cost-accuracy operating points%s"
                 % ((" -- " + args.note) if args.note else ""), y=1.0)
    fig.tight_layout()
    out2 = os.path.join(args.out_dir, "fig_lines_frontier%s.png" % args.suffix)
    fig.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("saved -> %s , %s" % (out1, out2))
    print("methods plotted: %s" % ", ".join(methods))


if __name__ == "__main__":
    main()
