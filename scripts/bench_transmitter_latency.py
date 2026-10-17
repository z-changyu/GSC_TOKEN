#!/usr/bin/env python
"""Terminal-side latency ledger for the V2X latency budget.

The V2X budget in this project (``outputs/warning_latency/warning_latency.json``) is
100 ms for a warning / 10 ms for control, and the legacy generative terminal blows
through it: vision tower ~1.06 s + report generation ~3.11 s. This script measures
every terminal-side stage that a designer can actually trade off, so the report can
state which knob buys how many milliseconds:

  vision   : processor + ``get_image_features``  (scales with image pixels)
  prefill  : one forward pass over image + prompt tokens
  decode   : autoregressive steps (scales with report length x weight bytes)
  total    : the number the deployment cares about

Sweeps attention implementation x image resolution x report length.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/v2x_edge_edge_main_4b8b.yaml")
    parser.add_argument("--manifest", default="data/manifests/v2xqa_test_filtered_8b.jsonl")
    parser.add_argument("--models", nargs="+", default=["models/Qwen3-VL-4B"])
    parser.add_argument("--attn", nargs="+", default=["eager", "sdpa"])
    parser.add_argument("--max-pixels", type=int, nargs="+",
                        default=[200704, 100352, 50176])
    parser.add_argument("--max-new-tokens", type=int, nargs="+", default=[128, 64, 32, 16])
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--output", default="outputs/bench_transmitter_latency.json")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import torch
    import yaml
    from PIL import Image
    from transformers import (AutoImageProcessor, AutoModelForImageTextToText,
                              AutoProcessor)

    from src.experiments.prompts import (build_choice_block,
                                         build_transmitter_messages,
                                         select_transmitter_request)
    from src.utils.io import read_jsonl

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    request_template = select_transmitter_request(config)
    processed = {str(r["row_idx"]): r for r in read_jsonl(config["dataset"]["processed"])}
    manifest = [str(e["sample_id"]) for e in read_jsonl(args.manifest)
                if str(e["sample_id"]) in processed][:args.num_samples]
    print(f"[data] {len(manifest)} samples | prompt={Path(args.config).name}")

    results = []
    for model_path in args.models:
        for attn in args.attn:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            model = AutoModelForImageTextToText.from_pretrained(
                model_path, dtype=torch.bfloat16, attn_implementation=attn,
                local_files_only=True).to("cuda:0").eval()
            base_peak = torch.cuda.max_memory_allocated() / 2 ** 30
            for max_pixels in args.max_pixels:
                processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
                processor.image_processor = AutoImageProcessor.from_pretrained(
                    model_path, min_pixels=config["model"]["image_min_pixels"],
                    max_pixels=max_pixels, local_files_only=True)
                # -------- vision + prefill (independent of report length)
                vision_t, prefill_t = [], []
                prepared = []
                for sid in manifest:
                    row = processed[sid]
                    image = Image.open(row["image"]).convert("RGB") if row.get("image") else None
                    request = request_template.format(
                        question=row["question"], choices=build_choice_block(row["choices"]))
                    content = [{"type": "image", "image": image}] if image is not None else []
                    content.append({"type": "text", "text": request})
                    inputs = processor.apply_chat_template(
                        build_transmitter_messages(content), tokenize=True,
                        add_generation_prompt=True, return_dict=True, return_tensors="pt",
                        enable_thinking=False)
                    inputs = {k: (v.to("cuda:0") if hasattr(v, "to") else v)
                              for k, v in inputs.items()}
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        feats = model.model.get_image_features(inputs["pixel_values"],
                                                               inputs["image_grid_thw"])
                    torch.cuda.synchronize()
                    vision_t.append(time.perf_counter() - t0)
                    del feats
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        out = model(**inputs, use_cache=True)
                    torch.cuda.synchronize()
                    prefill_t.append(time.perf_counter() - t0)
                    del out
                    prepared.append(inputs)
                    gc.collect(); torch.cuda.empty_cache()

                row_result = {
                    "model": model_path, "attn": attn, "max_pixels": max_pixels,
                    "vision_ms": round(1000 * sum(vision_t) / len(vision_t), 1),
                    "prefill_ms": round(1000 * sum(prefill_t) / len(prefill_t), 1),
                    "peak_vram_gib": round(base_peak, 2),
                }
                print(f"[{Path(model_path).name}/{attn}/px{max_pixels}] "
                      f"vision {row_result['vision_ms']:7.1f} ms  "
                      f"prefill {row_result['prefill_ms']:6.1f} ms")

                # -------- decode (depends on report length)
                for n_new in args.max_new_tokens:
                    times, actual = [], []
                    for inputs in prepared:
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        with torch.no_grad():
                            out = model.generate(**inputs, max_new_tokens=n_new,
                                                 do_sample=False)
                        torch.cuda.synchronize()
                        times.append(time.perf_counter() - t0)
                        actual.append(int(out.shape[1] - inputs["input_ids"].shape[1]))
                        del out
                        gc.collect(); torch.cuda.empty_cache()
                    gen_s = sum(times) / len(times)
                    mean_tok = sum(actual) / len(actual)
                    entry = dict(row_result, max_new_tokens=n_new,
                                 decode_ms=round(1000 * gen_s, 1),
                                 ms_per_token=round(1000 * gen_s / max(1.0, mean_tok), 2),
                                 mean_new_tokens=round(mean_tok, 1),
                                 gen_seconds_per_report=round(gen_s, 2),
                                 terminal_total_ms=round(row_result["vision_ms"]
                                                         + row_result["prefill_ms"]
                                                         + 1000 * gen_s, 1))
                    results.append(entry)
                    print(f"    n={n_new:4d}  decode {entry['decode_ms']:8.1f} ms "
                          f"({entry['ms_per_token']:5.2f} ms/tok, {mean_tok:5.1f} tok)  "
                          f"END-TO-END {entry['terminal_total_ms']:7.1f} ms")
                for inputs in prepared:
                    del inputs
                del processor
                gc.collect(); torch.cuda.empty_cache()
            del model
            gc.collect(); torch.cuda.empty_cache()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"samples": len(manifest), "results": results},
                              indent=2), encoding="utf-8")
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
