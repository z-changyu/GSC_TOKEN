#!/usr/bin/env python
"""Sidecar of per-token embeddings for the report tokens in a score cache.

Both redundancy-aware coupling variants need to know whether two report tokens say
the *same thing*:

    A (conservative)  M_ij <- a_ij * (1 - cos(e_i, e_j))
    B (penalty)       F_T(T) = sum_i u'_i - gamma sum_{i<j} sim(e_i, e_j) u'_i u'_j

The embeddings stored here are the transmitter's **input-token embeddings** (a pure
lookup, no forward pass), which is the cheap and reproducible choice: they capture
token identity, which is what "the same detail said twice" is about. They are
deliberately not the contextual hidden states of the scoring forward, so that the
sidecar can be rebuilt in seconds and does not depend on attention settings.

Saved as float16 tensors keyed by sample_id in a single ``.pt`` file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/v2x_edge_edge_8b_rx8b_smoke.yaml")
    parser.add_argument("--score-cache", default=None)
    parser.add_argument("--output", default="outputs/cache/v2xqa_token_embeddings_8b.pt")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    import yaml
    from transformers import AutoModelForImageTextToText

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cache_path = args.score_cache or config["score_cache"]
    rows = [json.loads(line) for line in open(cache_path, encoding="utf-8")]
    print(f"{len(rows)} samples from {cache_path}")

    path = config["model"]["transmitter_path"]
    model = AutoModelForImageTextToText.from_pretrained(
        path, dtype=torch.bfloat16, local_files_only=True).to("cuda:0").eval()
    embed = model.get_input_embeddings()

    out: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for row in rows:
            ids = torch.tensor([row["token_ids"]], device="cuda:0")
            out[str(row["sample_id"])] = embed(ids)[0].detach().to(torch.float16).cpu()
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, target)
    dims = {tuple(v.shape) for v in out.values()}
    print(f"saved {len(out)} embedding matrices, shapes {sorted(dims)[:4]} -> {target}")
    print(f"size {target.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
