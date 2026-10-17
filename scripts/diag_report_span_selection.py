#!/usr/bin/env python
"""Does the duplicated question/option region waste transmit budget?

The scored payload ``semantic_prompt`` is question + option list + report, but the
receiver builds its own prompt with ``build_question_block(row)``, so the first
part of the payload repeats text the RSU already has. Measurement shows that
region is 30% of the payload yet receives 44% of the top-10% utility mass.

This script turns that observation into an accuracy test. At a fixed token
payload, it compares

  arm "all"    keep the m highest-utility tokens over the whole payload, and
  arm "body"   keep the m highest-utility tokens drawn only from the report body,

then asks the deployed receiver to answer. If "body" beats "all", the utility
mass spent on the duplicated region is a measurable loss and the selection should
be restricted to the report span.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from src.experiments.report_span import report_span_start  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/v2x_edge_edge_8b_rx8b.yaml")
    parser.add_argument("--score-cache", default="outputs/cache/v2xqa_scores_8b_pred_cpl.jsonl")
    parser.add_argument("--manifest", default="data/manifests/v2xqa_test_filtered_8b.jsonl")
    parser.add_argument("--processed", default="data/processed/v2xqa_test.jsonl")
    parser.add_argument("--weights", type=float, nargs=3, default=(0.25, 0.30, 0.45))
    parser.add_argument("--retentions", type=float, nargs="+", default=(0.25, 0.35))
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--output-dir", default="outputs/report_span_selection")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["experiment"].get("cuda_visible_devices", "0"))

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    from src.experiments.prompts import ANSWER_REQUEST
    from src.experiments.receiver_protocol import answer_with_protocol
    from src.experiments.scoring import combine_scores
    from src.metrics.qa import exact_match
    from src.utils.io import atomic_json, read_jsonl

    cache = {json.loads(line)["sample_id"]: json.loads(line)
             for line in Path(args.score_cache).read_text(encoding="utf-8").splitlines() if line.strip()}
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
    print(f"[receiver] {rx_path} samples={len(manifest)}", flush=True)

    rows = []
    for sample_id in manifest:
        row = cache[sample_id]
        dataset_row = processed[sample_id]
        token_ids = row["token_ids"]
        total = len(token_ids)
        utility = combine_scores(row["attention"], row["semantic"], row["gradient"],
                                 args.weights)
        body_start = report_span_start(row["token_strings"],
                                       dataset_row.get("choices"))
        truth = str(dataset_row["choices"][int(dataset_row["answer"])])
        for retention in args.retentions:
            keep = max(1, int(round(retention * total)))
            plans = {
                "all": sorted(torch.topk(utility, keep).indices.tolist()),
                "body": sorted(
                    (body_start + int(i) for i in torch.topk(
                        utility[body_start:], min(keep, total - body_start)).indices.tolist())),
            }
            for arm, kept in plans.items():
                text = rx_tokenizer.decode([token_ids[j] for j in kept],
                                           skip_special_tokens=True)
                result = answer_with_protocol(
                    model=receiver, tokenizer=rx_tokenizer, row=dataset_row, received=text,
                    survived_tokens=1, total_tokens=1,
                    answer_request=ANSWER_REQUEST,
                    max_new_tokens=int(config["receiver"]["max_new_tokens"]),
                    protocol="v6", restore_threshold=0.0)
                rows.append({"sample_id": sample_id, "retention": retention, "arm": arm,
                             "kept": len(kept), "total": total, "body_start": body_start,
                             "correct": int(exact_match(result["prediction"], truth)),
                             "prediction": result["prediction"], "truth": truth})
                gc.collect(); torch.cuda.empty_cache()
        print(f"[sample] {sample_id} n={total} body_start={body_start} done", flush=True)

    accuracy = {}
    for r in rows:
        accuracy.setdefault(f"{r['retention']}:{r['arm']}", []).append(r["correct"])
    accuracy = {k: sum(v) / len(v) for k, v in accuracy.items()}
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "summary.json", {
        "samples": len(manifest), "weights": list(args.weights),
        "retentions": list(args.retentions), "receiver": rx_path,
        "accuracy": accuracy, "rows": rows})

    print("\n=== 固定 token 载荷：全 prompt 选择 vs 仅报告正文选择 ===")
    print(f"{'保留比':>7s} {'全prompt':>9s} {'仅正文':>7s} {'Δ':>7s}")
    for retention in args.retentions:
        a = accuracy.get(f"{retention}:all")
        b = accuracy.get(f"{retention}:body")
        if a is None or b is None:
            continue
        print(f"{retention*100:6.0f}% {a*100:9.1f} {b*100:7.1f} {(b-a)*100:+7.1f}")


if __name__ == "__main__":
    main()
