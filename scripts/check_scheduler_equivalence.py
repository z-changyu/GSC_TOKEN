#!/usr/bin/env python
"""Equivalence and latency check for the scheduling solver.

The batched dual search in ``solve_base_lp`` must return *exactly* the same
extreme point as the scalar version it replaced. Because the decision depends on
which breakpoint attains the minimum of the convex dual, a one-ulp change in the
arithmetic can flip a decision, so the check is done on decision fingerprints
(MD5 of the per-token level vector) rather than on aggregate metrics.

Usage
-----
    # before touching the solver
    python scripts/check_scheduler_equivalence.py --dump /tmp/sched_before.json
    # after the change
    python scripts/check_scheduler_equivalence.py --dump /tmp/sched_after.json
    python scripts/check_scheduler_equivalence.py --compare /tmp/sched_before.json \
        /tmp/sched_after.json

The dump covers every cached sample under three fading channels x two SNRs x two
seeds (720 scheduler calls), and records the per-call latency so the speed-up is
measured on the same workload.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.channel import build_channel, decoding_probability  # noqa: E402
from src.channel.link import ProtectionCodebook  # noqa: E402
from src.channel.reliable_capacity import reliable_spectral_efficiency  # noqa: E402
from src.experiments.scoring import SampleScores, combine_scores  # noqa: E402
from src.scheduling import select_level  # noqa: E402
from src.utils.io import read_jsonl  # noqa: E402

CHANNELS = ("rayleigh", "rician", "frequency_selective")
SNRS = (0.0, 15.0)
SEEDS = (0, 1)
WEIGHTS = (0.25, 0.30, 0.45)
BETA = 0.25
T_MAX = 8


def dump(score_cache: str, manifest_path: str, out_path: str) -> None:
    cache = {str(r["sample_id"]): SampleScores.from_dict(r)
             for r in read_jsonl(score_cache)}
    manifest = [str(e["sample_id"]) for e in read_jsonl(manifest_path)]
    manifest = [s for s in manifest if s in cache]
    codebook = ProtectionCodebook()
    levels = torch.tensor(codebook.levels, dtype=torch.float32)

    records, times = [], []
    for channel_name in CHANNELS:
        spec = yaml.safe_load(Path("configs/channels/%s.yaml" % channel_name).read_text())
        for snr_db in SNRS:
            for seed in SEEDS:
                channel = build_channel({**spec, "snr_db": snr_db, "seed": seed})
                for sample_id in manifest:
                    scores = cache[sample_id]
                    utility = combine_scores(scores.attention, scores.semantic,
                                             scores.gradient, WEIGHTS).float()
                    num_tokens = int(utility.numel())
                    state = channel.sample((1,))
                    budget = 1e-3 * 1e6 * float(
                        reliable_spectral_efficiency(state, 0.05)[0])
                    q = decoding_probability(state, codebook).reshape(-1)
                    start = time.perf_counter()
                    level = select_level("proposed", utility,
                                         torch.zeros(num_tokens, num_tokens), BETA, q,
                                         levels, budget, T_MAX,
                                         random.Random(seed * 1000003))
                    times.append((time.perf_counter() - start) * 1e3)
                    selected = [i for i in range(num_tokens) if int(level[i]) > 0]
                    records.append({
                        "key": "%s/%g/%d/%s" % (channel_name, snr_db, seed, sample_id),
                        "hash": hashlib.md5(level.numpy().tobytes()).hexdigest()[:16],
                        "selected": len(selected),
                        "symbols": float(sum(float(levels[int(level[i]) - 1])
                                             for i in selected)),
                    })
    times.sort()
    payload = {"calls": len(records), "p50_ms": statistics.median(times),
               "p95_ms": times[min(len(times) - 1, int(0.95 * len(times)))],
               "max_ms": max(times), "records": records}
    Path(out_path).write_text(json.dumps(payload, indent=1))
    print("%s: %d calls | P50 %.3f ms | P95 %.3f ms | max %.3f ms"
          % (out_path, len(records), payload["p50_ms"], payload["p95_ms"],
             payload["max_ms"]))


def compare(before_path: str, after_path: str) -> None:
    before = json.loads(Path(before_path).read_text())
    after = json.loads(Path(after_path).read_text())
    old = {r["key"]: r for r in before["records"]}
    new = {r["key"]: r for r in after["records"]}
    if set(old) != set(new):
        raise SystemExit("the two dumps cover different workloads")
    mismatched = [k for k in old if old[k]["hash"] != new[k]["hash"]]
    print("配置数 %d | 决策一致 %d | 不一致 %d"
          % (len(old), len(old) - len(mismatched), len(mismatched)))
    for key in mismatched[:5]:
        print("  %s: selected %d->%d, symbols %.1f->%.1f"
              % (key, old[key].get("selected", old[key].get("sel", -1)),
                 new[key].get("selected", new[key].get("sel", -1)),
                 old[key].get("symbols", old[key].get("sym", -1)), new[key].get("symbols", new[key].get("sym", -1))))
    print("时延 P50 %.3f -> %.3f ms | P95 %.3f -> %.3f ms | max %.3f -> %.3f ms"
          % (before["p50_ms"], after["p50_ms"], before["p95_ms"], after["p95_ms"],
             before["max_ms"], after["max_ms"]))
    speedup = before["p50_ms"] / max(after["p50_ms"], 1e-9)
    print("P50 加速比 %.1fx" % speedup)
    if mismatched:
        raise SystemExit("决策发生改变，不能用该优化")
    print("✅ 全部决策逐位一致，且时延下降 %.1fx" % speedup)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", metavar="OUT_JSON")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    parser.add_argument("--score-cache", default="outputs/cache/v2xqa_scores_2b.jsonl")
    parser.add_argument("--manifest", default="data/manifests/v2xqa_test_filtered_8b.jsonl")
    args = parser.parse_args()
    if args.dump:
        dump(args.score_cache, args.manifest, args.dump)
    elif args.compare:
        compare(*args.compare)
    else:
        parser.error("pass --dump OUT or --compare BEFORE AFTER")


if __name__ == "__main__":
    main()
