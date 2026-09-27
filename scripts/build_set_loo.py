#!/usr/bin/env python
"""Set-level vs marginal importance: why LOO top-k fails under compression.

For m in {25%, 50%, 75%} of the report, jointly delete the m tokens with the
highest individual LOO scores and compare:

    ratio(m) = [NLL(y* | Q, P minus top-m) - NLL(y* | Q, P)] / sum(top-m LOO)

ratio << 1 means the top-m tokens are mutually redundant (submodular): their
joint value is far below the sum of their marginal values, which is exactly why
a marginal-importance top-k selector collapses once the retention ratio drops.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments/v2x_edge_edge_8b.yaml")
    ap.add_argument("--output", default="outputs/cache/v2xqa_set_loo.jsonl")
    args = ap.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(
        config["experiment"].get("cuda_visible_devices", "0"))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.experiments.prompts import ANSWER_REQUEST
    from src.utils.io import read_jsonl, write_jsonl

    cache_path = Path(config["score_cache"])
    aware = {str(r["sample_id"]): r for r in read_jsonl(
        Path(config["answer_score_cache"]))}
    processed = {str(r["row_idx"]): r for r in
                 read_jsonl(config["dataset"]["processed"])}
    rows = list(read_jsonl(cache_path))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import torch  # noqa: F811
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["receiver_path"], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["receiver_path"], dtype=torch.bfloat16,
        local_files_only=True).to("cuda:0").eval()
    eos_id = tokenizer.eos_token_id

    def nll(ids, alen):
        labels = torch.full_like(ids, -100)
        labels[0, -alen:] = ids[0, -alen:]
        with torch.no_grad():
            return float(model(input_ids=ids, labels=labels).loss.detach())

    out = []
    for pos, row in enumerate(rows, 1):
        sid = str(row["sample_id"])
        loo = aware.get(sid, {}).get("answer_loo", [])
        if not loo or len(loo) != len(row["token_ids"]):
            continue
        text = row["semantic_prompt"]
        sample = processed.get(sid, {})
        gold = str(sample["choices"][sample["answer"]])
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{ANSWER_REQUEST}\n\n{text}"}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompt_ids = tokenizer(chat, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(gold, add_special_tokens=False)["input_ids"] + [eos_id]
        report_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        start = len(prompt_ids) - len(report_ids)
        end = start + len(report_ids)
        base = torch.tensor([prompt_ids + answer_ids], device="cuda:0")
        n_full = nll(base, len(answer_ids))

        order = sorted(range(len(loo)), key=lambda i: -loo[i])
        rec = {"sample_id": sid, "num_tokens": len(loo)}
        for frac in (0.25, 0.5, 0.75):
            m = max(1, round(frac * len(loo)))
            top = order[:m]
            kept = prompt_ids[:start] + [pid for j, pid in
                                         enumerate(prompt_ids[start:end])
                                         if j not in set(top)] + prompt_ids[end:]
            joint = nll(torch.tensor([kept + answer_ids], device="cuda:0"),
                        len(answer_ids)) - n_full
            s_ind = sum(loo[i] for i in top)
            rec["m%d_ratio" % round(frac * 100)] = joint / s_ind if s_ind > 1e-9 else None
            rec["m%d_joint" % round(frac * 100)] = joint
            rec["m%d_sum" % round(frac * 100)] = s_ind
        out.append(rec)
        if pos % 25 == 0 or pos == len(rows):
            write_jsonl(args.output, out)
            print(f"{pos}/{len(rows)} id={sid}")
    write_jsonl(args.output, out)
    print(f"DONE -> {args.output}")


if __name__ == "__main__":
    main()
