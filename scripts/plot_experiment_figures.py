#!/usr/bin/env python
"""Generate the result figures of the paper from the recorded channel runs.

Figures (written to paper/images/ as vector PDFs):
  fig_snr_sweep.pdf   accuracy vs SNR (0/5/10/15/20 dB), fading-channel mean,
                      fair receiver protocol (v6_norestore); 0/15 dB from the
                      main run, 5/10/20 dB from the SNR sweep run. The main-line
                      runs are selected with --main-prefix (default 4B->8B, i.e.
                      outputs/channel_main_4b8b_*; pass channel_main_2b4b for the
                      earlier reduced deployment).
  fig_budget_csi.pdf  two panels: accuracy vs bandwidth (0.25/0.5/1/2 MHz) and
                      accuracy vs CSI-error variance (0.01/0.05/0.10).
  fig_model_axis.pdf  full/proposed across the five terminal-side configurations
                      (2B, 2B+distil, 4B with 4B edge, 4B with 8B edge, 8B).

All numbers come from the recorded per-observation CSVs; the 8B run covered the
150-question pool and is subset to the 60-question filtered set here so every
curve uses the identical question set.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.io import read_jsonl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "images"
FADE = ["rayleigh", "rician", "frequency_selective"]
STYLE = {
    "full": dict(label="Prompt-SC (uncompressed)", color="#444444", marker="s", ls="--"),
    "plugin_capacity": dict(label="Capacity oracle (violating)", color="#888888",
                            marker="v", ls=":"),
    "proposed": dict(label="Proposed", color="#d62728", marker="o", ls="-"),
    "adaptive_sc": dict(label="Adaptive-SC", color="#1f77b4", marker="D", ls="-"),
    "tokcom_uep": dict(label="TokCom-UEP (2025)", color="#2ca02c", marker="^", ls="-"),
    "answer_loo": dict(label="Answer-LOO (gold std.)", color="#9467bd", marker="x", ls="-"),
}
METHODS = list(STYLE)


def load(path: str, keep: set[str] | None = None) -> pd.DataFrame:
    frame = pd.read_csv(ROOT / path)
    if keep is not None:
        frame = frame[frame["sample_id"].astype(str).isin(keep)]
    return frame[frame["channel"].isin(FADE)]


def acc_by(frame: pd.DataFrame, methods, key: str, values) -> dict[str, list[float]]:
    out = {}
    for method in methods:
        sub = frame[frame["method"] == method]
        out[method] = [float(sub[sub[key] == v]["correct"].mean()) * 100 for v in values]
    return out


def fading_mean(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["channel"].isin(FADE)]


def line_panel(ax, xs, data, xlabel, methods=METHODS, xlog=False, xticks=None):
    for method in methods:
        style = STYLE[method]
        ax.plot(xs, data[method], color=style["color"], marker=style["marker"],
                ls=style["ls"], lw=1.6, ms=5, label=style["label"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    if xlog:
        ax.set_xscale("log")
    if xticks is not None:
        ax.set_xticks(xticks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-prefix", default="channel_main_4b8b",
                        help="run-name prefix of the main sweep family "
                             "(outputs/<prefix>_{fair,snr,b025,b050,b200,"
                             "csise005,csise010})")
    parser.add_argument("--out-dir", default=str(OUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir).resolve()
    prefix = args.main_prefix
    out.mkdir(parents=True, exist_ok=True)
    filt = {str(r["sample_id"]) for r in
            read_jsonl(ROOT / "data/manifests/v2xqa_test_filtered_8b.jsonl")}

    base = load(f"outputs/{prefix}_fair/channel_results.csv")
    snr = load(f"outputs/{prefix}_snr/channel_results.csv")
    b025 = load(f"outputs/{prefix}_b025/channel_results.csv")
    b050 = load(f"outputs/{prefix}_b050/channel_results.csv")
    b200 = load(f"outputs/{prefix}_b200/channel_results.csv")
    csi005 = load(f"outputs/{prefix}_csise005/channel_results.csv")
    csi010 = load(f"outputs/{prefix}_csise010/channel_results.csv")

    # ---- figure 1: SNR sweep -------------------------------------------------
    snrs = [0.0, 5.0, 10.0, 15.0, 20.0]
    data = acc_by(fading_mean(pd.concat([base, snr])), METHODS, "snr_db", snrs)
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    line_panel(ax, snrs, data, "SNR (dB)")
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out / "fig_snr_sweep.pdf")
    plt.close(fig)

    # ---- figure 2: budget (left) and CSI error (right) -----------------------
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    budgets = [0.25, 0.5, 1.0, 2.0]
    bdata = acc_by(fading_mean(pd.concat([b025, b050, base, b200])),
                   METHODS, None, None) if False else {}
    for method in METHODS:
        vals = []
        for frame in (b025, b050, base, b200):
            sub = fading_mean(frame)
            vals.append(float(sub[sub["method"] == method]["correct"].mean()) * 100)
        bdata[method] = vals
    line_panel(axes[0], budgets, bdata, "Bandwidth (MHz)", xticks=budgets)
    axes[0].set_title("(a) Bandwidth sweep", fontsize=9)
    csis = [0.01, 0.05, 0.10]
    cdata = {}
    for method in METHODS:
        vals = []
        for frame in (base, csi005, csi010):
            sub = fading_mean(frame)
            vals.append(float(sub[sub["method"] == method]["correct"].mean()) * 100)
        cdata[method] = vals
    line_panel(axes[1], csis, cdata, r"CSI error variance $\sigma_e^2$")
    axes[1].set_title("(b) CSI error sweep", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.04), framealpha=0.9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "fig_budget_csi.pdf")
    plt.close(fig)

    # ---- figure 3: model axis ------------------------------------------------
    arms = [
        ("2B", load("outputs/channel_main_2b4b_fair/channel_results.csv")),
        ("2B+distil", load("outputs/channel_main_2b4b_distill_fair/channel_results.csv")),
        ("4B", load("outputs/channel_main_4b4b_fair/channel_results.csv")),
        ("4B*", load("outputs/channel_main_4b8b_fair/channel_results.csv")),
        ("8B", load("outputs/channel_8b_fair/channel_results.csv", keep=filt)),
    ]
    labels = ["2B\n(edge 4B)", "2B+distil\n(edge 4B)", "4B\n(edge 4B)",
              "4B\n(edge 8B)", "8B\n(edge 4B)"]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    xs = range(len(arms))
    for method, style in (("full", STYLE["full"]), ("proposed", STYLE["proposed"]),
                          ("adaptive_sc", STYLE["adaptive_sc"])):
        vals = [float(fading_mean(frame)[frame["method"] == method]["correct"].mean()) * 100
                for _, frame in arms]
        ax.plot(xs, vals, color=style["color"], marker=style["marker"],
                ls=style["ls"], lw=1.6, ms=6, label=style["label"])
        for x, v in zip(xs, vals):
            ax.annotate(f"{v:.1f}", (x, v), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=6.5)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(40, 95)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax.set_xlabel("Terminal-side model (edge coordinator in parentheses)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_model_axis.pdf")
    plt.close(fig)

    for name in ("fig_snr_sweep.pdf", "fig_budget_csi.pdf", "fig_model_axis.pdf"):
        print(f"written {out / name}")


if __name__ == "__main__":
    main()
