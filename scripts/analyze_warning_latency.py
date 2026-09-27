#!/usr/bin/env python
"""Warning-latency budget for the intersection warning use case (Route A).

Route A moves the rich semantic report off the warning critical path: the
vehicle's native perception stack (radar / ADAS detector) fires the trigger, the
short semantic demand goes uplink, and the edge coordinator answers with a
single forward pass (``v6_letter``). This script composes that path from the
measured stage latencies and evaluates it against the V2X warning requirements.

Stage provenance
----------------
* trigger (10-30 ms) -- ASSUMPTION: the vehicle's own detection stack, not the
  VLM. Measured separately: the 2B VLM vision tower alone costs 1.06 s, which is
  exactly why the semantic report cannot sit on the critical path.
* uplink (1 ms) -- the reliable-budget model of the paper: every method that
  respects Eq. (6) fits in one slot of T_s = 1 ms.
* scheduling -- measured, after the batched solver (``schedule_seconds``).
* edge judgement -- measured, ``receive_seconds`` with ``v6_letter`` (single
  forward pass) and with ``v6_norestore`` (24-token autoregressive answer).
* downlink -- control-plane sized, taken as <1 ms.

Outputs
-------
``paper/images/fig_warning_latency.pdf``
    (a) stage-wise decomposition of both pipelines on a log axis, with the
        100 ms warning budget marked;
    (b) the tolerable warning latency versus conflict distance for three
        approach speeds, with the Route A distribution overlaid.
``outputs/warning_latency/warning_latency.json``
    per-stage statistics and the timely-delivery rate grid.

Usage
-----
    python scripts/analyze_warning_latency.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# --------------------------------------------------------------- scenario model
SPEEDS_KMH = (30.0, 50.0, 60.0)
REACTION_S = 1.0          # driver perception-reaction time
DECEL_MS2 = 6.0           # emergency deceleration on dry asphalt
TRIGGER_LOW_MS = 10.0     # native perception trigger, ASSUMED
TRIGGER_HIGH_MS = 30.0
UPLINK_MS = 1.0           # one slot, Eq. (6) budget respected
DOWNLINK_MS = 0.5
TRANSMIT_VISION_MS = 1062.5      # measured: 2B VLM vision tower
TRANSMIT_REPORT_MS = 3110.3      # measured: full report, P50 (129-320 tokens)
WARNING_BUDGET_MS = 100.0        # GB/ETSI-class awareness and warning budget
CONTROL_BUDGET_MS = 10.0         # 3GPP eV2X cooperative-driving class


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--autoregressive", default="outputs/latency_v6_norestore/channel_results.csv")
    parser.add_argument("--letter", default="outputs/latency_v6_letter/channel_results.csv")
    parser.add_argument("--out-figure", default="paper/images/fig_warning_latency.pdf")
    parser.add_argument("--out-json", default="outputs/warning_latency/warning_latency.json")
    return parser.parse_args()


def stopping_distance(speed_kmh: float) -> float:
    """Perception-reaction plus braking distance [m]."""
    speed = speed_kmh / 3.6
    return speed * REACTION_S + speed ** 2 / (2.0 * DECEL_MS2)


def tolerable_latency(distance_m: float, speed_kmh: float) -> float:
    """Latest arrival time [ms] that still allows a full stop before contact."""
    speed = speed_kmh / 3.6
    return max(0.0, (distance_m - stopping_distance(speed_kmh)) / speed * 1e3)


def main() -> None:
    args = parse_args()
    auto = pd.read_csv(args.autoregressive)
    letter = pd.read_csv(args.letter)

    # ---------------------------------------------- stage statistics (Route A)
    schedule_ms = letter.schedule_seconds * 1e3
    judge_ms = letter.receive_seconds * 1e3
    route_a_ms = TRIGGER_LOW_MS + UPLINK_MS + schedule_ms + judge_ms + DOWNLINK_MS
    route_a_high = (TRIGGER_HIGH_MS + UPLINK_MS + schedule_ms + judge_ms + DOWNLINK_MS)
    legacy_ms = (TRANSMIT_VISION_MS + TRANSMIT_REPORT_MS + UPLINK_MS + DOWNLINK_MS
                 + auto.schedule_seconds * 1e3 + auto.receive_seconds * 1e3)

    def pct(series, q):
        return float(np.percentile(series, q))

    stages = {
        "端侧触发（原生感知，假设）": {"low": TRIGGER_LOW_MS, "high": TRIGGER_HIGH_MS,
                                     "source": "assumption"},
        "端侧视觉塔（实测）": {"ms": TRANSMIT_VISION_MS, "source": "measured"},
        "端侧报告生成（实测，异步）": {"ms": TRANSMIT_REPORT_MS, "source": "measured"},
        "上行（建模）": {"ms": UPLINK_MS, "source": "model"},
        "边侧调度（实测，优化后）": {"p50": float(schedule_ms.median()),
                                    "p95": pct(schedule_ms, 95), "source": "measured"},
        "边侧判定-判别式（实测）": {"p50": float(judge_ms.median()),
                                   "p95": pct(judge_ms, 95), "source": "measured"},
        "边侧判定-自回归（实测）": {"p50": float((auto.receive_seconds * 1e3).median()),
                                    "p95": float(np.percentile(auto.receive_seconds * 1e3, 95)),
                                    "source": "measured"},
        "回传（建模）": {"ms": DOWNLINK_MS, "source": "model"},
    }

    # ---------------------------------------------------- timely-delivery grid
    grid = []
    distances = [10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0]
    for speed in SPEEDS_KMH:
        for distance in distances:
            budget = tolerable_latency(distance, speed)
            grid.append({
                "speed_kmh": speed, "distance_m": distance,
                "tolerable_ms": budget,
                "timely_rate_low_trigger": float((route_a_ms <= budget).mean()) if budget > 0 else 0.0,
                "timely_rate_high_trigger": float((route_a_high <= budget).mean()) if budget > 0 else 0.0,
                "feasible": bool(budget > 0),
            })

    summary = {
        "warning_budget_ms": WARNING_BUDGET_MS,
        "control_budget_ms": CONTROL_BUDGET_MS,
        "route_a_ms": {"trigger_low": {"p50": pct(route_a_ms, 50), "p95": pct(route_a_ms, 95)},
                       "trigger_high": {"p50": pct(route_a_high, 50), "p95": pct(route_a_high, 95)}},
        "legacy_ms": {"p50": pct(legacy_ms, 50), "p95": pct(legacy_ms, 95)},
        "stages": stages, "timely_grid": grid,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("=== 关键路径（Route A）===")
    for tag, series in (("触发 10 ms", route_a_ms), ("触发 30 ms", route_a_high)):
        print("  %-10s P50 %6.1f ms | P95 %6.1f ms | P99 %6.1f ms"
              % (tag, pct(series, 50), pct(series, 95), pct(series, 99)))
    print("=== 当前主线（报告在关键路径）===")
    print("  P50 %.0f ms | P95 %.0f ms" % (pct(legacy_ms, 50), pct(legacy_ms, 95)))
    print("=== 及时率（触发 30 ms 保守口径）===")
    print("  %-6s %-8s %-10s %-10s" % ("速度", "距离", "允许时延", "及时率"))
    for row in grid:
        print("  %-6.0f %-8.0f %-10.0f %.3f%s"
              % (row["speed_kmh"], row["distance_m"], row["tolerable_ms"],
                 row["timely_rate_high_trigger"],
                 "" if row["feasible"] else "   (制动距离已超过该距离)"))

    # ---------------------------------------------------------------- figure
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.7,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.75))

    ax = axes[0]
    # One row per stage: a stacked bar on a log axis would misrepresent the sum,
    # so every stage is drawn on its own row with its own bar, and the two totals
    # are given their own rows at the bottom.
    rows = [
        ("vehicle trigger (assumed)", TRIGGER_LOW_MS, TRIGGER_HIGH_MS, "0.55", "//"),
        ("uplink (one slot)", UPLINK_MS, UPLINK_MS, "0.85", ""),
        ("edge scheduling (measured)", pct(schedule_ms, 50), pct(schedule_ms, 95),
         "0.30", ""),
        ("edge judgement (measured)", pct(judge_ms, 50), pct(judge_ms, 95), "0.30", "\\\\"),
        ("downlink", DOWNLINK_MS, DOWNLINK_MS, "0.85", ""),
        ("Route A total", pct(route_a_high, 50), pct(route_a_high, 95), "black", ""),
        ("VLM vision tower (async in Route A)", TRANSMIT_VISION_MS, TRANSMIT_VISION_MS, "black", "xx"),
        ("VLM report generation (async in Route A)", TRANSMIT_REPORT_MS,
         TRANSMIT_REPORT_MS, "black", "xx"),
        ("current pipeline total", pct(legacy_ms, 50), pct(legacy_ms, 95), "black", ""),
    ]
    order = list(range(len(rows)))[::-1]
    for position, index in enumerate(order):
        label, low, high, color, hatch = rows[index]
        ax.barh(position, high, left=0, height=0.55, color=color, hatch=hatch,
                edgecolor="black", linewidth=0.5, zorder=3)
        if high > low * 1.01:                       # show the measured range
            ax.plot([low, high], [position, position], color="white", linewidth=1.6,
                    zorder=4)
        ax.text(high * 1.25, position, "%.1f" % high if high < 10 else "%.0f" % high,
                va="center", fontsize=6.2)
    ax.axvline(WARNING_BUDGET_MS, color="black", linestyle="--", linewidth=0.9, zorder=5)
    ax.axvline(CONTROL_BUDGET_MS, color="0.35", linestyle=":", linewidth=0.9, zorder=5)
    ax.set_xscale("log")
    ax.set_xlim(0.2, 4e4)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([rows[i][0] for i in order], fontsize=6.2)
    ax.set_xlabel("Latency per stage (ms, log scale)")
    ax.set_title("(a) Warning critical path", fontsize=8.5)
    ax.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.4)
    ax.set_axisbelow(True)
    for value, label, dx, align in ((CONTROL_BUDGET_MS, "10 ms control", -2, "right"),
                                    (WARNING_BUDGET_MS, "100 ms warning budget", 2, "left")):
        ax.annotate(label, xy=(value, len(rows) - 0.35), xytext=(dx, -4),
                    textcoords="offset points", fontsize=5.8, ha=align, va="top")

    ax = axes[1]
    distances = np.linspace(8.0, 60.0, 200)
    for speed, style in zip(SPEEDS_KMH, ("-", "--", ":")):
        values = [tolerable_latency(d, speed) for d in distances]
        ax.plot(distances, values, color="black", linestyle=style, linewidth=1.2,
                label="%d km/h" % speed)
    ax.axhline(WARNING_BUDGET_MS, color="black", linestyle="--", linewidth=0.9)
    ax.axhspan(pct(route_a_high, 50), pct(route_a_high, 95), color="0.6", alpha=0.45,
               zorder=1)
    ax.text(59.0, pct(route_a_high, 95) * 1.15, "Route A\n(30 ms trigger)",
            fontsize=5.8, color="black", ha="right", va="bottom")
    ax.set_yscale("log")
    ax.set_ylim(1.0, 5e3)
    ax.set_xlim(8.0, 60.0)
    ax.set_xlabel("Distance to the conflict point (m)")
    ax.set_ylabel("Tolerable warning latency (ms)")
    ax.set_title("(b) Timeliness margin", fontsize=8.5)
    ax.grid(linestyle="--", linewidth=0.4, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.0, frameon=False, loc="upper left", title="approach speed",
              title_fontsize=6.0, handletextpad=0.5)

    out = Path(args.out_figure)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.4)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    print("wrote %s and %s" % (out, out_json))


if __name__ == "__main__":
    main()
