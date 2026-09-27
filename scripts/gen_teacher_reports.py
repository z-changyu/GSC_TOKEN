#!/usr/bin/env python
"""Experiment B (step 1): generate *teacher* semantic reports for distillation.

Why: the terminal's bottleneck is report quality, not receiver capacity. The same
questions give full-prompt accuracy 60.0% with a 2B terminal and 86.1% with an 8B
terminal, so B distils the strong terminal's reports into a small one.

This script only generates report text (no pragmatic scoring, no gradients) using
the *exact* transmitter prompt of the main experiment, so the distilled student can
later be dropped into the same pipeline unchanged.

Output JSONL fields:
``sample_id, question, choices, answer, image, image_path_is, report``
plus the generation config for provenance.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/v2x_edge_edge_main_2b4b_gpu1.yaml",
                        help="supplies the transmitter prompt/caps/weights")
    parser.add_argument("--transmitter", default="models/Qwen3-VL-8B-Instruct",
                        help="teacher checkpoint")
    parser.add_argument("--processed", default="data/processed/v2xqa_train.jsonl")
    parser.add_argument("--manifest", default="data/manifests/v2xqa_train.jsonl")
    parser.add_argument("--output", default="outputs/distill/teacher_reports_8b.jsonl")
    parser.add_argument("--device", default=None, help="override cuda_visible_devices")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    visible = str(args.device or config["experiment"].get("cuda_visible_devices", "0"))
    if "," in visible:
        raise SystemExit("single GPU only")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    from transformers import (AutoImageProcessor, AutoProcessor,
                              Qwen3VLForConditionalGeneration)
    from src.experiments.prompts import build_choice_block, select_transmitter_request
    from src.experiments.scoring import PragmaticScorer, load_image
    from src.utils.io import read_jsonl, write_jsonl

    request_template = select_transmitter_request(config)
    processed = {str(r["row_idx"]): r for r in read_jsonl(args.processed)}
    for row in processed.values():
        row["_request"] = request_template.format(
            question=row["question"], choices=build_choice_block(row["choices"]))
    needed = [str(e["sample_id"]) for e in read_jsonl(args.manifest)
              if str(e["sample_id"]) in processed]
    if args.max_samples:
        needed = needed[:args.max_samples]
    print(f"teacher={args.transmitter}  samples={len(needed)}  prompt="
          f"{config.get('transmitter', {}).get('prompt_version', 'v1')}")

    processor = AutoProcessor.from_pretrained(args.transmitter, local_files_only=True)
    processor.image_processor = AutoImageProcessor.from_pretrained(
        args.transmitter, min_pixels=config["model"]["image_min_pixels"],
        max_pixels=config["model"]["image_max_pixels"], local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.transmitter, dtype=torch.bfloat16, attn_implementation="eager",
        local_files_only=True).to("cuda:0").eval()
    scorer = PragmaticScorer(model, processor, config)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {str(r["sample_id"]) for r in read_jsonl(out_path)}
        print(f"resuming: {len(done)} already generated")
    pending = [sid for sid in needed if sid not in done]

    records = []
    if done:
        records = list(read_jsonl(out_path))
    for pos, sid in enumerate(pending, 1):
        row = processed[sid]
        image = load_image(row.get("image"))
        report = scorer.generate_prompt(row, image)
        records.append({
            "sample_id": sid, "question": row["question"], "choices": list(row["choices"]),
            "answer": int(row["answer"]), "image": row.get("image"),
            "image_path_is": row.get("image_path_is"), "report": report,
            "teacher": args.transmitter,
            "max_new_tokens": int(config["transmitter"]["max_new_tokens"]),
        })
        if pos % 50 == 0 or pos == len(pending):
            write_jsonl(out_path, records)
            print(f"generated {pos}/{len(pending)} (last id={sid}, "
                  f"{len(report.split())} words)", flush=True)
        del image
        gc.collect()
        torch.cuda.empty_cache()

    write_jsonl(out_path, records)
    print(f"DONE: {len(records)} teacher reports -> {out_path}")


if __name__ == "__main__":
    main()
