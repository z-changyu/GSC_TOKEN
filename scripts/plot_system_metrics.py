#!/usr/bin/env python
"""System-level views of an existing channel summary (no GPU needed).

fig_latency_accuracy: per-method (latency, accuracy) points at both SNRs.
fig_reliability_price: accuracy versus measured outage -- the price-of-
reliability line from Proposed (~0.05) to the Capacity oracle (~0.5).
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
]
CH_ORDER = ["rayleigh", "rician", "frequency_selective"]
SNRS = (0.0, 15.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/channel_8b_final14/summary.json")
    ap.add_argument("--out-dir", default="outputs/figures")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    summary = json.load(open(args.run, encoding="utf-8"))
    conds = {(c["channel"], float(c["snr_db"]), c["method"]): c
             for c in summary["conditions"]}
    methods = [m for m, _l, _c, _mk in STYLE
               if any((ch, s, m) in conds for ch in CH_ORDER for s in SNRS)]

    def metrics(ch, snr, m):
        c = conds[(ch, snr, m)]
        lat = c["mean_receive_seconds"] + c["mean_schedule_seconds"]
        outage = (c["plugin_outage_rate"] if m == "plugin_capacity"
                  else c["capacity_outage_rate"])
        return lat, c["accuracy"] * 100.0, outage

    # ------------------------------------------------- latency vs accuracy
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
    for ax, snr in zip(axes, SNRS):
        for key, label, color, marker in STYLE:
            if key not in methods:
                continue
            pts = [metrics(ch, snr, key) for ch in CH_ORDER]
            lat = sum(p[0] for p in pts) / len(pts)
            acc = sum(p[1] for p in pts) / len(pts)
            ax.scatter([lat], [acc], s=110, color=color, marker=marker,
                       edgecolor="black", linewidth=0.6, zorder=3, label=label)
            ax.annotate(label, (lat, acc), textcoords="offset points",
                        xytext=(8, 5), fontsize=8, zorder=4)
        ax.set_xscale("log")
        ax.set_xlabel("System latency (s, receiver + scheduling, log scale)")
        ax.set_title("SNR %d dB (mean over fading channels)" % int(snr))
        ax.grid(linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Answer accuracy (%)")
    fig.suptitle("Latency-accuracy operating points", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig_latency_accuracy.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------ price of reliability
    cells = [(ch, s) for ch in CH_ORDER for s in SNRS]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for key in ("proposed", "plugin_capacity"):
        label = {"proposed": "Proposed (honours outage contract)",
                 "plugin_capacity": "Capacity oracle (genie CSI)"}[key]
        color = {"proposed": "#c62828", "plugin_capacity": "#b0bec5"}[key]
        pts = [metrics(ch, s, key) for ch, s in cells]
        out = sum(p[2] for p in pts) / len(pts)
        acc = sum(p[1] for p in pts) / len(pts)
        ax.scatter([out], [acc], s=160, color=color, edgecolor="black",
                   linewidth=0.8, zorder=3, label=label)
        ax.annotate("%s\nout=%.3f acc=%.1f" % (label.split(" (")[0], out, acc),
                    (out, acc), textcoords="offset points", xytext=(10, -14),
                    fontsize=9, zorder=4)
    ax.plot([0.058, 0.512], [58.6, 62.6], linestyle="--", color="#999999",
            zorder=1, label="price of reliability")
    ax.set_xlabel("Measured outage rate (log scale)")
    ax.set_xscale("log")
    ax.set_ylabel("Answer accuracy (%)")
    ax.set_title("The price of honouring the outage contract (eps_out = 0.05)")
    ax.grid(linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig_reliability_price.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved -> %s/fig_latency_accuracy.png, fig_reliability_price.png"
          % args.out_dir)


if __name__ == "__main__":
    main()
