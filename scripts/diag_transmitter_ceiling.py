#!/usr/bin/env python
"""Localise the ~27% full-report accuracy: measure the transmitter VLM's own
accuracy when it answers the question *directly from the image* (no report, no
text-only receiver). This is the semantic-communication ceiling for the
transmitter. Comparing it with the 27% receiver-from-report accuracy separates
"the VLM can't read V2X scenes" (model ceiling low) from "the report format /
truncation loses the discriminative evidence" (report bottleneck).
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
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    visible = str(config["experiment"].get("cuda_visible_devices", "0"))
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import torch
    from transformers import (AutoImageProcessor, AutoProcessor,
                              Qwen3VLForConditionalGeneration)
    from src.experiments.prompts import build_choice_block, canonical_prediction
    from src.metrics.qa import normalize_answer
    from src.utils.io import read_jsonl

    device = args.device or "cuda:0"
    processed = {str(r["row_idx"]): r for r in read_jsonl(config["dataset"]["processed"])}
    manifest = list(read_jsonl(config["dataset"]["manifest"]))

    tx_path = config["model"]["transmitter_path"]
    processor = AutoProcessor.from_pretrained(tx_path, local_files_only=True)
    processor.image_processor = AutoImageProcessor.from_pretrained(
        tx_path, min_pixels=config["model"]["image_min_pixels"],
        max_pixels=config["model"]["image_max_pixels"], local_files_only=True)
    transmitter = Qwen3VLForConditionalGeneration.from_pretrained(
        tx_path, dtype=torch.bfloat16, attn_implementation="eager",
        local_files_only=True).to(device).eval()
    tokenizer = processor.tokenizer

    qa_prompt = ("Question: {question}\n\nCandidate choices:\n{choices}\n\n"
                 "Answer with the exact option text only, no explanation.")

    ok = 0
    n = 0
    records = []
    from PIL import Image
    for entry in manifest:
        if args.max_samples is not None and n >= args.max_samples:
            break
        sid = str(entry["sample_id"])
        row = processed.get(sid)
        if row is None:
            continue
        n += 1
        choices = list(row["choices"])
        answer = int(row["answer"])
        truth = str(choices[answer])
        image = Image.open(row["image"]).convert("RGB") if row.get("image") else None
        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": qa_prompt.format(
            question=row["question"], choices=build_choice_block(choices))})
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        inp = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = transmitter.generate(**inp, max_new_tokens=24, do_sample=False)
        raw = tokenizer.decode(out[0, inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
        pred, _ = canonical_prediction(raw, choices)
        correct = normalize_answer(pred) == normalize_answer(truth)
        ok += int(correct)
        records.append({"sample_id": sid, "truth": truth, "raw": raw,
                        "pred": pred, "correct": correct})

    out_path = Path("outputs/diag_ceiling.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"transmitter direct-from-image accuracy = {ok/n:.3f}  (n={n})")
    print(f"(receiver-from-report was 0.267; the gap is the report/compression loss)")


if __name__ == "__main__":
    main()
