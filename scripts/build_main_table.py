#!/usr/bin/env python
"""Build the single canonical main table and mark the superseded sources.

Three generations of main-line results coexist in ``results/``:

``main_8b8b_maskall``    current implementation + body mask on every method
``main_8b8b_full14``     current implementation, mask off
``channel_main_4b8b_*``  older implementation (predicted-target scoring before the
                         coupling variant B / beta 0.13 recalibration)

Reporting numbers from different generations side by side invites exactly the kind
of error this project has already hit (a 2B->4B result quoted against a 4B->8B
one). This script promotes one source to canonical, emits the tables the paper
should quote, and records in the summary which sources are superseded and why, so
the choice is auditable rather than a matter of memory.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def wilson_halfwidth(accuracy: float, n: int) -> float:
    """95% half-width, normal approximation on the pooled proportion."""
    return 1.96 * math.sqrt(max(accuracy * (1.0 - accuracy), 1e-12) / max(n, 1))


def build(source: Path) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(source / "channel_results.csv", low_memory=False)
    coverage = {
        "samples": int(frame.sample_id.nunique()),
        "channels": sorted(frame.channel.unique().tolist()),
        "snr_db": sorted(float(v) for v in frame.snr_db.unique()),
        "seeds": sorted(int(v) for v in frame.channel_seed.unique()),
        "methods": sorted(frame.method.unique().tolist()),
    }
    rows = []
    for (snr, method), group in frame.groupby(["snr_db", "method"]):
        accuracy = float(group.correct.mean())
        symbols = float(group.transmitted_symbols.mean())
        rows.append({
            "snr_db": float(snr),
            "method": method,
            "accuracy": accuracy,
            "ci95": wilson_halfwidth(accuracy, len(group)),
            "mean_symbols": symbols,
            "acc_per_1k_symbols": accuracy * 1000.0 / symbols if symbols > 0 else float("nan"),
            "mean_trr": float(group.TRR.mean()),
            "n": int(len(group)),
        })
    table = pd.DataFrame(rows).sort_values(["snr_db", "accuracy"], ascending=[True, False])
    return table, coverage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", default="outputs/main_8b8b_maskall")
    parser.add_argument("--mask-off", default="outputs/main_8b8b_full14")
    parser.add_argument("--superseded", nargs="*", default=[
        "results/channel_main_4b8b_fair", "results/channel_main_4b4b_fair"])
    parser.add_argument("--output-dir", default="results/canonical")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    table, coverage = build(Path(args.canonical))
    table.to_csv(out / "main_table.csv", index=False)

    comparison = None
    if Path(args.mask_off).exists():
        off, _ = build(Path(args.mask_off))
        comparison = off.merge(table, on=["snr_db", "method"], how="outer",
                               suffixes=("_mask_off", "_mask_all"))
        comparison.to_csv(out / "mask_effect.csv", index=False)

    summary = {
        "canonical_source": args.canonical,
        "canonical_reason": (
            "current implementation (predicted scoring target, joint reception, "
            "coupling variant B with beta=0.13, context-aware LP) with the body mask "
            "applied to every method, so all methods are compared under the same "
            "information condition: none of them may spend symbols re-transmitting "
            "the question and options the receiver already holds."),
        "coverage": coverage,
        "appendix_sources": {
            args.mask_off: "mask off; kept only to quantify the mask's effect",
            **{path: "older implementation / older budget口径; superseded, must not "
                       "be subtracted from or compared against the canonical table"
               for path in args.superseded},
        },
        "caveats": [
            "Accuracy is on the filtered manifest (full = 100%), a controlled "
            "comparison set; absolute accuracy on the unfiltered pool is ~40pp lower.",
            "The body mask lifts the lexical baselines far more than proposed, so the "
            "margin over tokcom_uep (+0.9pp, p=0.29) is not significant under it.",
            "multi-terminal (VI-B-5) numbers use a different answer prompt and a 1/K "
            "budget per terminal; they are not comparable to these rows.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for snr in coverage["snr_db"]:
        block = table[table.snr_db == snr]
        print(f"\n=== SNR {int(snr)} dB（规范主表，3 信道 × 3 种子 × 60 题）===")
        print(f"{'method':20s} {'acc':>7s} {'±95%':>6s} {'symbols':>8s} {'acc/1k':>7s} {'TRR':>6s}")
        for row in block.itertuples():
            print(f"{row.method:20s} {row.accuracy*100:7.1f} {row.ci95*100:6.1f} "
                  f"{row.mean_symbols:8.0f} {row.acc_per_1k_symbols:7.2f} {row.mean_trr:6.3f}")
    print(f"\n[done] canonical={args.canonical} -> {out}")


if __name__ == "__main__":
    main()
