#!/usr/bin/env python
"""Diagnostic: multimodal upper bound for the receiver.

The deployed receiver is Qwen3-1.7B (text-only) and only ever reads the VLM's
textual semantic description, never the image. This script measures how accurate
the *transmitter* VLM (Qwen3-VL-2B) is when it answers the ScienceQA question
directly WITH the image -- i.e. the accuracy ceiling if the receiver could see.

Comparing this upper bound U with the pipeline's ideal `full` accuracy (0.640)
quantifies the text-only / description penalty that caps every experiment.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.prompts import build_choice_block, canonical_prediction
from src.metrics.qa import exact_match
from src.utils.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/channel_scienceqa.yaml")
    parser.add_argument("--num-samples", type=int, default=0, help="0 = all in score cache")
    parser.add_argument("--output", default="outputs/diag_multimodal_upper_bound.json")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    os_environ = __import__("os")
    os_environ.environ["CUDA_VISIBLE_DEVICES"] = str(
        config["experiment"].get("cuda_visible_devices", "0"))
    os_environ.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import torch
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    processed = {str(r["row_idx"]): r for r in read_jsonl(config["dataset"]["processed"])}
    cache_ids = [str(r["sample_id"]) for r in read_jsonl(config["score_cache"])]
    ids = cache_ids[: args.num_samples] if args.num_samples else cache_ids
    ids = [i for i in ids if i in processed]
    print(f"multimodal upper bound on {len(ids)} samples")

    model_path = config["model"]["transmitter_path"]
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="eager",
        local_files_only=True).to("cuda:0").eval()

    correct = 0
    records = []
    t0 = time.perf_counter()
    for n, sid in enumerate(ids, 1):
        row = processed[sid]
        image = Image.open(row["image"]).convert("RGB") if row.get("image") else None
        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text":
            f"Question: {row['question']}\n\nCandidate choices:\n"
            f"{build_choice_block(row['choices'])}\n\nAnswer with the exact option text only, with no explanation."})
        messages = [
            {"role": "system", "content": [{"type": "text",
                "text": "You answer a multiple-choice science question about an image. Reply with the exact option text only."}]},
            {"role": "user", "content": content}]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=config["receiver"]["max_new_tokens"],
                                 do_sample=False)
        raw = processor.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        pred, mode = canonical_prediction(raw, row["choices"])
        gt = str(row["choices"][row["answer"]])
        ok = exact_match(pred, gt)
        correct += int(ok)
        records.append({"sample_id": sid, "prediction": pred, "mode": mode,
                        "correct": ok, "raw": raw[:120]})
        if n % 10 == 0:
            print(f"  {n}/{len(ids)} acc={correct/n:.3f}")

    acc = correct / len(ids)
    summary = {"samples": len(ids), "upper_bound_accuracy": acc,
               "pipeline_ideal_full_accuracy": 0.640,
               "text_only_penalty": round(acc - 0.640, 3),
               "seconds": round(time.perf_counter() - t0, 1), "records": records}
    Path(args.output).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
