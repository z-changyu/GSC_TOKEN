#!/usr/bin/env python
"""Section VI-B-2: does the reliable capacity of Eq. (5)-(6) actually hold epsilon_out?

Eq. (6) sets the symbol budget from ``C_tau = T_s B c_eps(h_hat)`` where
``c_eps = log2(1 + gamma_bar * Q_{epsilon_out}(mu, nu))`` is a lower quantile of the
*true* channel gain given the noisy estimate. The claim is that the realized
spectral efficiency ``log2(1 + gamma_bar |h|^2)`` falls below this budget with
probability at most ``epsilon_out``.

This draws many channel realizations per (channel, SNR), computes both the
quantile-based budget and the realized rate, and reports the empirical outage.
The plug-in capacity ``log2(1 + gamma_bar |h_hat|^2)`` is measured alongside it:
it is the internal baseline the paper argues against, and it is expected to
*overshoot* the target -- it ignores the CSI error, so it cannot certify the
guarantee.

CPU only: the channel model is analytic, no model inference involved.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", nargs="*", default=[
        "configs/channels/rayleigh.yaml",
        "configs/channels/rician.yaml",
        "configs/channels/frequency_selective.yaml"])
    parser.add_argument("--snr-db", type=float, nargs="*", default=(0.0, 5.0, 10.0, 15.0, 20.0))
    parser.add_argument("--epsilon-grid", type=float, nargs="*", default=(0.01, 0.05, 0.10))
    parser.add_argument("--realizations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", default="outputs/capacity_calibration")
    args = parser.parse_args()

    import torch
    from src.channel import build_channel
    from src.channel.reliable_capacity import reliable_spectral_efficiency

    rows = []
    for channel_path in args.channels:
        spec = yaml.safe_load(Path(channel_path).read_text(encoding="utf-8"))
        for snr_db in args.snr_db:
            channel = build_channel({**spec, "snr_db": float(snr_db), "seed": args.seed})
            state = channel.sample((args.realizations,))
            realized = torch.log2(1.0 + float(state.snr_linear) * state.h.abs().pow(2))
            plugin = torch.log2(1.0 + float(state.snr_linear) * state.h_hat.abs().pow(2))
            for epsilon_out in args.epsilon_grid:
                budget = reliable_spectral_efficiency(state, epsilon_out)
                outage_ours = float((realized < budget).float().mean())
                outage_plugin = float((realized < plugin).float().mean())
                rows.append({
                    "channel": channel.name,
                    "snr_db": float(snr_db),
                    "epsilon_out": float(epsilon_out),
                    # Mean over realizations, not a single draw: the budget adapts to
                    # the estimate, so the outage is the fraction where the realized
                    # rate misses it.
                    "mean_budget": float(budget.mean()),
                    "mean_realized": float(realized.mean()),
                    "mean_plugin": float(plugin.mean()),
                    "outage_ours": outage_ours,
                    "outage_plugin": outage_plugin,
                    "target_met": outage_ours <= epsilon_out + 1e-9,
                })
            print(f"{channel.name:20s} snr={int(snr_db):2d}  "
                  + "  ".join(f"eps={r['epsilon_out']:.2f}: ours={r['outage_ours']:.4f} "
                              f"plugin={r['outage_plugin']:.4f}"
                              for r in rows[-len(args.epsilon_grid):]), flush=True)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(
        {"realizations": args.realizations, "seed": args.seed,
         "snr_db": list(args.snr_db), "epsilon_grid": list(args.epsilon_grid),
         "rows": rows}, indent=2), encoding="utf-8")
    met = sum(1 for r in rows if r["target_met"])
    print(f"\n[summary] target met on {met}/{len(rows)} conditions; "
          f"plugin baseline violates in {sum(1 for r in rows if r['outage_plugin'] > r['epsilon_out'])}"
          f"/{len(rows)}")


if __name__ == "__main__":
    main()
