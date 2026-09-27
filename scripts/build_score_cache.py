#!/usr/bin/env python
"""Build the SampleScores cache for the dataset named in an experiment config.

`run_cross_validation.py` is ScienceQA-5-fold specific (its dataset/folds/cache all
default to ScienceQA), so it cannot produce a V2X-QA cache. This driver is driven
purely by ``config['dataset']`` (manifest + processed) and writes ``SampleScores``
rows to ``config['score_cache']`` — the exact format consumed by
`scripts/run_channel_experiment.py` and `scripts/filter_by_full.py`.

It resumes from an existing cache, so re-running only scores the missing samples.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/v2x_edge_edge.yaml")
    parser.add_argument("--manifest", default=None, help="override config dataset.manifest")
    parser.add_argument("--processed", default=None, help="override config dataset.processed")
    parser.add_argument("--score-cache", default=None, help="override config score_cache")
    parser.add_argument("--max-samples", type=int, default=None, help="cap for smoke runs")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    visible = str(config["experiment"].get("cuda_visible_devices", "0"))
    if "," in visible:
        raise SystemExit("This driver is restricted to exactly one visible GPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import torch
    from transformers import (AutoImageProcessor, AutoProcessor,
                              Qwen3VLForConditionalGeneration)
    from src.experiments.prompts import (build_choice_block,
                                         detect_meta_leak,
                                         select_transmitter_request)
    from src.experiments.scoring import PragmaticScorer, SampleScores, load_image
    from src.utils.io import read_jsonl, write_jsonl

    manifest_path = args.manifest or config["dataset"]["manifest"]
    processed_path = args.processed or config["dataset"]["processed"]
    cache_path = Path(args.score_cache or config["score_cache"])

    # processed keyed by str(row_idx); inject the transmitter request the scorer needs.
    request_template = select_transmitter_request(config)
    print(f"transmitter prompt version: "
          f"{config.get('transmitter', {}).get('prompt_version', 'v1')}")
    processed = {str(r["row_idx"]): r for r in read_jsonl(processed_path)}
    for row in processed.values():
        row["_request"] = request_template.format(
            question=row["question"], choices=build_choice_block(row["choices"]))
    needed = [str(e["sample_id"]) for e in read_jsonl(manifest_path)
              if str(e["sample_id"]) in processed]
    print(f"manifest -> {len(needed)} samples in processed pool")

    # Resume from any existing cache.
    cache: dict[str, SampleScores] = {}
    if cache_path.exists():
        for row in read_jsonl(cache_path):
            cache[row["sample_id"]] = SampleScores.from_dict(row)
    print(f"cache has {len(cache)}/{len(needed)}; building the rest")

    tx_path = config["model"]["transmitter_path"]
    processor = AutoProcessor.from_pretrained(tx_path, local_files_only=True)
    processor.image_processor = AutoImageProcessor.from_pretrained(
        tx_path, min_pixels=config["model"]["image_min_pixels"],
        max_pixels=config["model"]["image_max_pixels"], local_files_only=True)
    transmitter = Qwen3VLForConditionalGeneration.from_pretrained(
        tx_path, dtype=torch.bfloat16, attn_implementation="eager",
        local_files_only=True).to("cuda:0").eval()
    if config.get("transmitter", {}).get("gradient_checkpointing", False):
        transmitter.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        # HF applies checkpointing only when self.training is True; Qwen3 has no
        # dropout, so train() is numerically identical to eval() here.
        transmitter.train()
        print("gradient checkpointing: ON (train mode)")
    scorer = PragmaticScorer(transmitter, processor, config)

    pending = [sid for sid in needed if sid not in cache]
    if args.max_samples:
        pending = pending[:args.max_samples]
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    for pos, sid in enumerate(pending, 1):
        row = processed[sid]
        image = load_image(row.get("image"))
        scores = scorer.score(row, image)
        scores.prompt_meta_leak = detect_meta_leak(scores.semantic_prompt)
        cache[sid] = scores
        if pos % 10 == 0 or pos == len(pending):
            write_jsonl(cache_path,
                        [cache[s].to_dict() for s in sorted(cache, key=int)])
            print(f"scored {pos}/{len(pending)} id={sid} tokens={scores.num_tokens} "
                  f"nll={scores.answer_nll:.3f} leak={scores.prompt_meta_leak}")
        gc.collect(); torch.cuda.empty_cache()

    write_jsonl(cache_path, [cache[s].to_dict() for s in sorted(cache, key=int)])
    print(f"DONE: {len(cache)} samples -> {cache_path}")


if __name__ == "__main__":
    main()
