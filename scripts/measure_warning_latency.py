#!/usr/bin/env python
"""Measure the transmitter-side stages of the warning critical path (Route A).

Route A keeps the rich semantic report *off* the warning critical path: a
lightweight on-board trigger fires, the short semantic demand goes uplink, and
the edge answers from its cached/asynchronous context. To budget that path
honestly we need the transmitter-side costs separated, because the end-to-end
time of the full pipeline (3.45 s) is dominated by autoregressive report
generation, which Route A moves out of the critical path.

Three transmit-side stages are timed, warm, on the main-line 2B VLM:

``vision``
    Vision tower only (image -> features). This is the floor for any frame-level
    trigger that has to look at the image.
``prefill``
    One full forward pass over image + prompt (no decoding): the cost of a
    discriminative on-board decision that consumes the frame once.
``report``
    Autoregressive generation of the report (the stage Route A makes
    asynchronous; the same stage costs 3.45 s median in the main run).

Usage
-----
    python scripts/measure_warning_latency.py --samples 3 --repeats 5 \
        --out outputs/warning_latency/transmitter.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from PIL import Image  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/v2x_edge_edge_main_2b4b.yaml")
    parser.add_argument("--model", default=None,
                        help="defaults to model.transmitter_path of --config")
    parser.add_argument("--processed", default="data/processed/v2xqa_test.jsonl")
    parser.add_argument("--manifest", default="data/manifests/v2xqa_test_filtered_8b.jsonl")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--out", default="outputs/warning_latency/transmitter.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import yaml
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    from src.experiments.prompts import (build_choice_block, build_transmitter_messages,
                                         select_transmitter_request)
    from src.utils.io import read_jsonl

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    request_template = select_transmitter_request(config)
    processed = {str(r["row_idx"]): r for r in read_jsonl(args.processed)}
    for row in processed.values():                 # same request the scorer uses
        row["_request"] = request_template.format(
            question=row["question"], choices=build_choice_block(row["choices"]))
    samples = [str(e["sample_id"]) for e in read_jsonl(args.manifest)]
    samples = [s for s in samples if s in processed and processed[s].get("image")]
    samples = samples[:args.samples]

    args.model = args.model or config["model"]["transmitter_path"]
    print("transmitter model: %s" % args.model)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="eager").to("cuda:0").eval()

    stages: dict[str, list[float]] = {"vision": [], "prefill": [], "report": []}
    generated_tokens: list[int] = []
    for sample_id in samples:
        row = processed[sample_id]
        image = Image.open(row["image"]).convert("RGB")
        inputs = processor.apply_chat_template(
            build_transmitter_messages([{"type": "image", "image": image},
                                        {"type": "text", "text": row["_request"]}]),
            tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt")
        inputs = {k: (v.to("cuda:0") if hasattr(v, "to") else v) for k, v in inputs.items()}

        def vision_only():
            with torch.no_grad():
                return model.model.get_image_features(inputs["pixel_values"],
                                                      inputs["image_grid_thw"])

        def one_forward():
            with torch.no_grad():
                return model(**inputs)

        def full_report():
            with torch.no_grad():
                return model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                      do_sample=False)

        for name, call in (("vision", vision_only), ("prefill", one_forward)):
            call()                                            # warm up
            times = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                output = call()
                times.append((time.perf_counter() - start) * 1e3)
                del output
            stages[name].extend(times)
            print("%-8s sample %-4s P50 %7.2f ms" % (name, sample_id,
                                                     statistics.median(times)))

        start = time.perf_counter()
        output = full_report()
        elapsed = (time.perf_counter() - start) * 1e3
        new_tokens = int(output.shape[1] - inputs["input_ids"].shape[1])
        stages["report"].append(elapsed)
        generated_tokens.append(new_tokens)
        print("report   sample %-4s %7.1f ms (%d tokens)" % (sample_id, elapsed, new_tokens))
        del inputs, output
        torch.cuda.empty_cache()

    summary = {
        "model": args.model, "samples": samples, "repeats": args.repeats,
        "generated_tokens": generated_tokens,
        "ms": {name: {"p50": statistics.median(values),
                      "mean": statistics.fmean(values),
                      "min": min(values), "max": max(values)}
               for name, values in stages.items() if values},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n%-8s %10s %10s" % ("stage", "P50 ms", "mean ms"))
    for name, stats in summary["ms"].items():
        print("%-8s %10.2f %10.2f" % (name, stats["p50"], stats["mean"]))
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
