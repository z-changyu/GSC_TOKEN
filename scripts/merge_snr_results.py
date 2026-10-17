#!/usr/bin/env python
"""Merge the interrupted SNR sweep with the cells that were re-run.

The five-point SNR sweep (5/10/15/20 dB plus the original 0 dB) was killed after 15
of 27 (channel, snr, seed) cells. The driver has no resume logic -- it rewrites
``channel_results.partial.csv`` on every condition -- so the completed cells are
salvaged here and the missing ones are re-run into separate directories, then
concatenated.

The summary is recomputed from the merged per-sample records using the same schema
and the same CI formula as ``run_channel_experiment.py`` (Wilson-free normal
approximation over all observations of a condition), so downstream analysis cannot
tell the difference.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="channel_results CSVs, in merge order")
    parser.add_argument("--reference-summary",
                        default="outputs/main_8b8b_full14/summary.json",
                        help="summary.json whose provenance fields are copied")
    parser.add_argument("--output-dir", default="outputs/main_8b8b_snr")
    parser.add_argument("--expected-cells", type=int, default=27,
                        help="channels x snr x seeds that must all be present")
    args = parser.parse_args()

    frames = []
    for path in args.inputs:
        frame = pd.read_csv(path)
        frame["_source"] = Path(path).parent.name
        frames.append(frame)
        print(f"  + {Path(path).parent.name:28s} {len(frame):6d} rows")
    merged = pd.concat(frames, ignore_index=True)
    key = ["sample_id", "channel", "snr_db", "channel_seed", "method"]
    before = len(merged)
    merged = merged.drop_duplicates(subset=key, keep="first")
    print(f"merged {before} -> {len(merged)} rows after de-duplication")

    cells = merged.groupby(["channel", "snr_db", "channel_seed"]).ngroups
    print(f"cells present: {cells}/{args.expected_cells}")
    if cells != args.expected_cells:
        raise SystemExit(f"expected {args.expected_cells} cells, found {cells}; "
                         f"re-run the missing (channel, snr, seed) combinations")

    reference = json.loads(Path(args.reference_summary).read_text(encoding="utf-8"))
    conditions = []
    for (channel, snr, method), group in merged.groupby(["channel", "snr_db", "method"]):
        accuracy = float(group.correct.mean())
        n = len(group)
        conditions.append({
            "channel": channel,
            "snr_db": float(snr),
            "method": method,
            "observations": int(n),
            "accuracy": accuracy,
            "accuracy_ci95": 1.96 * math.sqrt(max(accuracy * (1 - accuracy), 1e-12) / n),
            "mean_TRR": float(group.TRR.mean()),
            "mean_symbols": float(group.transmitted_symbols.mean()),
            "capacity_outage_rate": float(group.capacity_outage.mean()),
            "plugin_outage_rate": float(group.plugin_outage.mean()),
            "mean_token_erasure_rate": float(group.token_erasure_rate.mean()),
            "mean_schedule_seconds": float(group.schedule_seconds.mean()),
            "mean_receive_seconds": float(group.receive_seconds.mean()),
            "mean_gpu_peak_memory_delta_gib": float(group.gpu_peak_memory_delta_gib.mean()),
            "mean_energy_j": float("nan"),
        })
    conditions.sort(key=lambda c: (c["channel"], c["snr_db"], c["method"]))

    summary = {
        "samples": int(merged.sample_id.nunique()),
        "channels": sorted(merged.channel.unique().tolist()),
        "snr_db": sorted(float(v) for v in merged.snr_db.unique()),
        "channel_seeds": sorted(int(v) for v in merged.channel_seed.unique()),
        "epsilon_out": reference.get("epsilon_out"),
        "methods": sorted(merged.method.unique().tolist()),
        "weights": reference.get("weights"),
        "beta": reference.get("beta"),
        "codebook_levels": reference.get("codebook_levels"),
        "code_rates": reference.get("code_rates"),
        "energy_backend": reference.get("energy_backend"),
        "conditions": conditions,
        "merged_from": [str(Path(p).parent.name) for p in args.inputs],
    }

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    merged.drop(columns=["_source"], errors="ignore").to_csv(
        out / "channel_results.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] {len(merged)} rows, {len(conditions)} conditions -> {out}")


if __name__ == "__main__":
    main()
