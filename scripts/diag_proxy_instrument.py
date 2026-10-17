#!/usr/bin/env python
"""Is the proxy wrong, or is the answer-NLL instrument wrong?

``run_proxy_validation.py`` validates ``F_T`` against the transmitter-side answer
NLL, and the 2026-10 rerun shows that instrument carries almost no signal:
deleting 30% of the prompt moves the answer NLL by +0.06..+0.5 nats with a sign
that flips between samples, while the proxy spans 2..17. Correlating a signal-free
quantity against a wide one produces noise, so the -0.39 within-sample Spearman is
not evidence that the proxy is misordered.

This script swaps in the instrument the method actually optimises: receiver
accuracy. For each sample it deletes the same number of tokens in two orders --
by the proxy (highest first) and uniformly at random -- and asks the deployed
receiver to answer. If proxy-ranked deletions damage accuracy more than random
deletions of the same size, the proxy is a valid importance order on the metric
that matters, regardless of its behaviour on NLL.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
from pathlib import Path

import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/v2x_edge_edge_8b_rx8b.yaml")
    parser.add_argument("--score-cache", default="outputs/cache/v2xqa_scores_8b_pred_cpl.jsonl")
    parser.add_argument("--manifest", default="data/manifests/v2xqa_test_filtered_8b.jsonl")
    parser.add_argument("--processed", default="data/processed/v2xqa_test.jsonl")
    parser.add_argument("--weights", type=float, nargs=3, default=(0.25, 0.30, 0.45))
    parser.add_argument("--fracs", type=float, nargs="+", default=(0.1, 0.2, 0.3))
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/proxy_instrument")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    visible = str(config["experiment"].get("cuda_visible_devices", "0"))
    os.environ["CUDA_VISIBLE_DEVICES"] = visible

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    from src.experiments.prompts import ANSWER_REQUEST
    from src.experiments.receiver_protocol import answer_with_protocol
    from src.experiments.scoring import SampleScores, combine_scores
    from src.metrics.qa import exact_match
    from src.utils.io import atomic_json, read_jsonl

    cache = {row["sample_id"]: SampleScores.from_dict(row) for row in read_jsonl(args.score_cache)}
    processed = {str(row["row_idx"]): row for row in read_jsonl(args.processed)}
    manifest = [str(e["sample_id"]) for e in read_jsonl(args.manifest)]
    manifest = [sid for sid in manifest if sid in cache][: args.num_samples]

    rx_path = config["model"]["receiver_path"]
    rx_tokenizer = AutoProcessor.from_pretrained(rx_path, local_files_only=True).tokenizer
    try:
        receiver = AutoModelForCausalLM.from_pretrained(
            rx_path, dtype=torch.bfloat16, attn_implementation="eager",
            local_files_only=True).to("cuda:0").eval()
    except Exception:                                    # noqa: BLE001
        from transformers import AutoModelForImageTextToText
        receiver = AutoModelForImageTextToText.from_pretrained(
            rx_path, dtype=torch.bfloat16, attn_implementation="eager",
            local_files_only=True).to("cuda:0").eval()
    print(f"[receiver] {rx_path}  samples={len(manifest)}", flush=True)

    rows = []
    for sample_id in manifest:
        scores = cache[sample_id]
        row = processed[sample_id]
        n = scores.num_tokens
        u_prime = combine_scores(scores.attention, scores.semantic, scores.gradient,
                                 args.weights)
        order = sorted(range(n), key=lambda i: -u_prime[i])
        truth = str(row["choices"][int(row["answer"])])
        for frac in (0.0, *args.fracs):
            m = max(1, int(round(frac * n))) if frac > 0 else 0
            plans = {"baseline": []} if m == 0 else {
                "proxy": order[:m],
                "random": sorted(random.Random(f"{args.seed}:{sample_id}:{frac}").sample(
                    range(n), m)),
            }
            for arm, deleted in plans.items():
                drop = set(deleted)
                kept = [scores.token_ids[j] for j in range(n) if j not in drop]
                text = rx_tokenizer.decode(kept, skip_special_tokens=True)
                result = answer_with_protocol(
                    model=receiver, tokenizer=rx_tokenizer, row=row, received=text,
                    survived_tokens=1, total_tokens=1,        # ideal link: never restore
                    answer_request=ANSWER_REQUEST,
                    max_new_tokens=int(config["receiver"]["max_new_tokens"]),
                    protocol="v6", restore_threshold=0.0)
                rows.append({"sample_id": sample_id, "frac": frac, "arm": arm,
                             "deleted": len(deleted), "tokens": n,
                             "correct": int(exact_match(result["prediction"], truth)),
                             "prediction": result["prediction"], "truth": truth})
                gc.collect(); torch.cuda.empty_cache()
        print(f"[sample] {sample_id} n={n} done", flush=True)

    summary = {"samples": len(manifest), "weights": list(args.weights),
               "fracs": list(args.fracs), "receiver": rx_path, "rows": rows}
    by_arm: dict = {}
    for r in rows:
        by_arm.setdefault((r["frac"], r["arm"]), []).append(r["correct"])
    summary["accuracy"] = {f"{f}:{a}": sum(v) / len(v) for (f, a), v in by_arm.items()}
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "summary.json", summary)

    print("\n=== 接收端精度：代理排序删除 vs 随机删除等量 token ===")
    print(f"{'删除比例':>8s} {'代理排序':>9s} {'随机':>7s} {'Δ':>7s}")
    base = summary["accuracy"].get("0.0:baseline")
    for frac in args.fracs:
        p = summary["accuracy"].get(f"{frac}:proxy")
        r = summary["accuracy"].get(f"{frac}:random")
        if p is None or r is None:
            continue
        print(f"{frac*100:7.0f}% {p*100:9.1f} {r*100:7.1f} {(p-r)*100:+7.1f}")
    if base is not None:
        print(f"  基线（未删除）= {base*100:.1f}%")


if __name__ == "__main__":
    main()
