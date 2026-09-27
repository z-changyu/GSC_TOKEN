#!/usr/bin/env python
"""Behavioural walk-through figure: what the scheduler does to one real report.

The paper's architecture figure is static; reviewers of a *scheduling* paper want
to see the scheduler react to the channel. This script replays the exact main-line
signal path of ``scripts/run_channel_experiment.py`` for a single report and draws,
for the two main-report SNR points (0 dB and 15 dB):

  * the pragmatic utility profile of the report (normalised, top row);
  * the protection level the scheduler assigns to every token
    (white = not transmitted, light grey = r_L, mid grey = r_M, black = r_H);
  * which of the selected tokens actually survive the channel
    (black tick = received, white = erased).

No model is loaded: everything comes from the recorded 2B pragmatic score cache
and the analytic channel model, so the figure is reproducible on CPU in seconds.

Usage
-----
    python scripts/plot_scheduler_walkthrough.py \
        --score-cache outputs/cache/v2xqa_scores_2b.jsonl \
        --manifest data/manifests/v2xqa_test_filtered_8b.jsonl \
        --out paper/images/fig_walkthrough.pdf
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from src.channel import build_channel, decoding_probability  # noqa: E402
from src.channel.link import ProtectionCodebook  # noqa: E402
from src.channel.reliable_capacity import reliable_spectral_efficiency  # noqa: E402
from src.experiments.scoring import SampleScores, combine_scores  # noqa: E402
from src.metrics.communication import token_retention_ratio  # noqa: E402
from src.scheduling import select_level  # noqa: E402
from src.utils.io import read_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-cache", default="outputs/cache/v2xqa_scores_2b.jsonl")
    parser.add_argument("--manifest", default="data/manifests/v2xqa_test_filtered_8b.jsonl")
    parser.add_argument("--channel", default="configs/channels/rayleigh.yaml")
    parser.add_argument("--snr-db", type=float, nargs="*", default=[0.0, 15.0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weights", type=float, nargs=3, default=[0.25, 0.30, 0.45])
    parser.add_argument("--t-max", type=int, default=8)
    parser.add_argument("--sample", default="median",
                        help="'median' report length, an index into the manifest, "
                             "or a sample_id")
    parser.add_argument("--tokenizer", default="models/Qwen3-VL-2B-Instruct")
    parser.add_argument("--num-words", type=int, default=6,
                        help="how many top-utility tokens to label with their text")
    parser.add_argument("--out", default="paper/images/fig_walkthrough.pdf")
    return parser.parse_args()


def decode_token_words(tokenizer, token_ids: list[int]) -> list[str]:
    """Readable single-token strings (byte-level BPE markers stripped)."""
    words = []
    for tid in token_ids:
        piece = tokenizer.decode([int(tid)], skip_special_tokens=True)
        piece = piece.replace("\n", " ").strip()
        words.append(piece if piece else "\u00b7")
    return words


def main() -> None:
    args = parse_args()
    codebook = ProtectionCodebook()
    levels = torch.tensor(codebook.levels, dtype=torch.float32)
    bandwidth, slot_duration, eps_out, beta = 1e6, 1e-3, 0.05, 0.25

    cache = {str(r["sample_id"]): SampleScores.from_dict(r)
             for r in read_jsonl(args.score_cache)}
    manifest = [str(e["sample_id"]) for e in read_jsonl(args.manifest)]
    manifest = [s for s in manifest if s in cache]

    lengths = {s: cache[s].num_tokens for s in manifest}
    if args.sample == "median":
        ordered = sorted(manifest, key=lambda s: lengths[s])
        sample_id = ordered[len(ordered) // 2]
    elif args.sample.isdigit():
        sample_id = manifest[int(args.sample)]
    else:
        sample_id = args.sample
    scores = cache[sample_id]
    utility = combine_scores(scores.attention, scores.semantic, scores.gradient,
                             args.weights).float()
    num_tokens = int(utility.numel())
    coupling = torch.zeros(num_tokens, num_tokens)

    from transformers import AutoProcessor
    tokenizer = AutoProcessor.from_pretrained(args.tokenizer,
                                              local_files_only=True).tokenizer
    words = decode_token_words(tokenizer, scores.token_ids)

    spec = yaml.safe_load(Path(args.channel).read_text())
    rows: list[dict] = []
    for snr_db in args.snr_db:
        channel = build_channel({**spec, "snr_db": float(snr_db), "seed": int(args.seed)})
        state = channel.sample((1,))
        c_eps = reliable_spectral_efficiency(state, eps_out)
        reliable = slot_duration * bandwidth * float(c_eps[0])
        realized = math.log2(1.0 + float(state.snr_linear) * float(state.h.abs().pow(2)[0]))
        q = decoding_probability(state, codebook).reshape(-1)
        level = select_level("proposed", utility, coupling, beta, q, levels,
                             reliable, args.t_max,
                             random.Random(int(args.seed) * 1000003))
        selected = [i for i in range(num_tokens) if int(level[i]) > 0]
        symbols = sum(float(levels[int(level[i]) - 1]) for i in selected)
        rng = random.Random(int(args.seed) * 1000003)
        kept = []
        for token in selected:
            rate = float(codebook.code_rates[int(level[token]) - 1])
            dropped = rng.random() < float(state.packet_loss_rate)
            if realized >= rate and not dropped:
                kept.append(token)
        counts = [0, 0, 0]
        for token in selected:
            counts[int(level[token]) - 1] += 1
        rows.append({
            "snr_db": float(snr_db), "budget": reliable, "symbols": symbols,
            "selected": len(selected), "received": len(kept),
            "kept_rH": counts[0], "kept_rM": counts[1], "kept_rL": counts[2],
            "trr": token_retention_ratio(len(selected), num_tokens),
            "received_rate": len(kept) / len(selected) if selected else 0.0,
            "level": level.tolist(), "kept": kept, "q": [float(v) for v in q],
            "realized_efficiency": realized,
        })
        print("SNR %5.1f dB | budget %7.1f | symbols %7.1f | selected %3d "
              "(rH %2d / rM %2d / rL %2d) | received %3d (%.0f%%)"
              % (snr_db, reliable, symbols, len(selected), counts[0], counts[1],
                 counts[2], len(kept), 100.0 * len(kept) / max(1, len(selected))))

    # ------------------------------------------------------------------- figure
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.7,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    n_rows = len(rows)
    band = 2.0                                   # 1 level row + 1 barcode row
    height = band * n_rows
    fig = plt.figure(figsize=(7.0, 0.34 * height + 1.25))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.42 * height], hspace=0.30)

    ax_u = fig.add_subplot(gs[0])
    ax_u.plot(np.arange(num_tokens), utility.numpy(), color="black", linewidth=0.7)
    ax_u.fill_between(np.arange(num_tokens), utility.numpy(), color="0.75")
    ax_u.set_xlim(-0.5, num_tokens - 0.5)
    ax_u.set_ylim(0, float(utility.max()) * 1.55)
    ax_u.set_ylabel("pragmatic utility", fontsize=8)
    ax_u.set_xticks([])
    candidates = [i for i in torch.topk(utility, k=min(4 * args.num_words,
                                                    num_tokens)).indices.tolist()
                  if any(ch.isalnum() for ch in words[i])]
    top: list[int] = []
    for index in candidates:                      # greedily keep well-separated tokens
        if all(abs(index - other) >= 12 for other in top):
            top.append(index)
        if len(top) == args.num_words:
            break
    top = sorted(top)
    y_max = float(utility.max())
    for rank, index in enumerate(top, start=1):
        ax_u.plot([index], [float(utility[index])], marker="o", markersize=2.4,
                  color="black")
        ax_u.text(index, float(utility[index]) + 0.05 * y_max, str(rank),
                  fontsize=6.0, ha="center", color="black")
    ax_u.text(num_tokens + 4.0, y_max * 1.52,
              "\n".join("%d  \u201c%s\u201d" % (rank, words[index])
                        for rank, index in enumerate(top, start=1)),
              fontsize=6.2, va="top", ha="left", linespacing=1.35)
    for spine in ("top", "right"):
        ax_u.spines[spine].set_visible(False)

    ax_l = fig.add_subplot(gs[1], sharex=ax_u)
    matrix = np.full((int(band) * n_rows, num_tokens), np.nan)
    for row, rec in enumerate(rows):
        for token in range(num_tokens):
            level_index = int(rec["level"][token])
            matrix[int(band) * row, token] = np.nan if level_index == 0 else level_index - 1
    ax_l.imshow(matrix, aspect="auto", interpolation="nearest",
                cmap=ListedColormap(["black", "0.42", "0.70"]).with_extremes(bad="white"),
                vmin=-0.5, vmax=2.5,
                extent=(-0.5, num_tokens - 0.5, band * n_rows - 0.5, -0.5))
    for row, rec in enumerate(rows):
        center = band * row + 1.0
        ax_l.vlines(rec["kept"], center - 0.32, center + 0.32, color="black",
                    linewidth=0.65, zorder=3)
    ax_l.set_yticks(sum([[band * r, band * r + 1.0] for r in range(n_rows)], []))
    ax_l.set_yticklabels(sum([["%g dB\nlevel" % rec["snr_db"],
                               "received"] for rec in rows], []), fontsize=7.0)
    ax_l.set_ylim(band * n_rows - 0.5, -0.5)
    ax_l.set_xlabel("Token index in the semantic report")
    ax_l.set_xticks(np.arange(0, num_tokens + 1, 25))
    for row, rec in enumerate(rows):
        ax_l.text(num_tokens + 2.0, band * row + 0.5,
                  "budget %.0f, spend %.0f\nkept %d / %d selected (TRR %.2f)"
                  % (rec["budget"], rec["symbols"], rec["received"], rec["selected"],
                     rec["trr"]),
                  fontsize=6.5, va="center", ha="left")
    ax_l.set_xlim(-0.5, num_tokens + 62)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="black", edgecolor="black", label="rH (48 symb/tok)"),
               Patch(facecolor="0.42", edgecolor="black", label="rM (32)"),
               Patch(facecolor="0.70", edgecolor="black", label="rL (16)"),
               Patch(facecolor="white", edgecolor="0.6", label="not sent"),
               Line2D([0], [0], color="black", linewidth=0.9, label="received")]
    ax_l.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.28),
                ncol=5, frameon=False, fontsize=6.5, handlelength=1.1,
                columnspacing=1.0, handletextpad=0.4)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print("wrote %s (sample %s, %d tokens, %d words labelled)"
          % (out, sample_id, num_tokens, len(top)))

    sidecar = out.parent / "fig_walkthrough.json"
    sidecar.write_text(json.dumps({
        "sample_id": sample_id, "num_tokens": num_tokens,
        "channel": Path(args.channel).stem, "seed": int(args.seed),
        "weights": list(args.weights), "epsilon_out": eps_out, "beta": beta,
        "rows": [{k: v for k, v in rec.items() if k not in ("level", "kept", "q")}
                 for rec in rows],
    }, indent=2, ensure_ascii=False))
    print("wrote %s" % sidecar)


if __name__ == "__main__":
    main()
