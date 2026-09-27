#!/usr/bin/env python
"""Stage-1 pre-filter for the edge-edge V2X-QA experiments (方案 A).

Run the *full* (uncompressed) semantic report through the receiver once, and keep
only the samples the receiver answers correctly. Downstream channel experiments then
run on this filtered manifest, so the `full` prompt method reports 100% accuracy on
the retained set — isolating compression / channel loss from task difficulty (the
earlier "low full-prompt accuracy" symptom).

The reports are already in the score cache, so this stage needs the **receiver** but
no VLM re-run:

    1. run_cross_validation.py --score-only   (builds the semantic-report cache)
    2. filter_by_full.py                     (this script -> filtered manifest)
    3. run_channel_experiment.py --manifest <filtered>

The filtered subset is exactly the set the receiver can answer from the full report,
so the `full` reference line equals 100% by construction.
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
    parser.add_argument("--config", default="configs/experiments/v2x_edge_edge.yaml")
    parser.add_argument("--score-cache", default=None, help="default: config['score_cache']")
    parser.add_argument("--manifest", default=None, help="default: config['dataset']['manifest']")
    parser.add_argument("--receiver-path", default=None, help="default: config['model']['receiver_path']")
    parser.add_argument("--output", default=None,
                        help="filtered manifest (default: <manifest stem>_filtered.jsonl)")
    parser.add_argument("--report", default=None,
                        help="JSON report (default: <output>.report.json)")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    visible = str(config["experiment"].get("cuda_visible_devices", "0"))
    if "," in visible:
        raise SystemExit("single GPU only")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.experiments.prompts import ANSWER_REQUEST, canonical_prediction
    from src.metrics.qa import exact_match
    from src.utils.io import atomic_json, read_jsonl, write_jsonl

    device = args.device or "cuda:0"
    score_cache_path = args.score_cache or config["score_cache"]
    manifest_path = args.manifest or config["dataset"]["manifest"]
    processed_path = config["dataset"]["processed"]
    rx_path = args.receiver_path or config["model"]["receiver_path"]
    if not Path(rx_path).exists():
        raise SystemExit(
            f"receiver not found at {rx_path}\n"
            f"Fetch Qwen3-4B (or set config model.receiver_path / --receiver-path to an "
            f"available text model such as models/Qwen3-1.7B).")

    rx_tokenizer = AutoTokenizer.from_pretrained(rx_path, local_files_only=True)
    receiver = AutoModelForCausalLM.from_pretrained(
        rx_path, dtype=torch.bfloat16, attn_implementation="eager",
        local_files_only=True).to(device).eval()

    cache = {str(r["sample_id"]): r for r in read_jsonl(score_cache_path)}
    processed = {str(r["row_idx"]): r for r in read_jsonl(processed_path)}
    manifest = list(read_jsonl(manifest_path))

    records, filtered = [], []
    kept = 0
    for entry in manifest:
        if args.max_samples is not None and len(records) >= args.max_samples:
            break
        sid = str(entry["sample_id"])
        sc = cache.get(sid)
        row = processed.get(sid)
        if sc is None or row is None:
            records.append({"sample_id": sid, "status": "missing", "correct": None})
            continue
        full_report = sc.get("semantic_prompt") or ""
        choices = list(row["choices"])
        answer = int(row["answer"])
        ground_truth = str(choices[answer]) if 0 <= answer < len(choices) else ""
        prompt = rx_tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{ANSWER_REQUEST}\n\n{full_report}"}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inp = rx_tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = receiver.generate(
                **inp, max_new_tokens=int(config["receiver"]["max_new_tokens"]),
                do_sample=False)
        raw = rx_tokenizer.decode(out[0, inp.input_ids.shape[1]:],
                                  skip_special_tokens=True).strip()
        pred, _ = canonical_prediction(raw, choices)
        correct = bool(exact_match(pred, ground_truth))
        kept += int(correct)
        records.append({"sample_id": sid, "status": "scored", "correct": correct,
                        "pred": pred, "truth": ground_truth})
        if correct:
            filtered.append({"sample_id": sid, "dataset": entry.get("dataset"),
                             "split": entry.get("split")})

    total = len(records)
    acc = kept / total if total else 0.0
    man_stem = Path(manifest_path).stem
    out_path = Path(args.output) if args.output else Path(manifest_path).parent / f"{man_stem}_filtered.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, filtered)
    report_path = Path(args.report) if args.report else out_path.with_suffix(".json").with_name(
        out_path.stem + ".report.json")
    report = {
        "input_manifest": manifest_path,
        "score_cache": score_cache_path,
        "receiver": rx_path,
        "total": total,
        "kept": kept,
        "dropped": total - kept,
        "full_prompt_accuracy_before_filter": acc,
        "full_prompt_accuracy_after_filter": 1.0,
        "output_manifest": str(out_path),
    }
    atomic_json(report_path, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
