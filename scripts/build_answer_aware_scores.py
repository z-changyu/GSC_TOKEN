#!/usr/bin/env python
"""Answer-aware token importance, measured by the *consumer* of the report.

Two training-free, answer-aware signals from the receiver LM (Qwen3-4B):

``answer_loo``
  Leave-one-out importance -- the gold standard of task-aware token scoring:
      loo_i = NLL(y* | Q, P \\ x_i) - NLL(y* | Q, P)
  i.e. how much the receiver's answer likelihood degrades when report token i is
  deleted (deletion mirrors the channel, which drops lost tokens). This is the
  exact quantity the paper's gradient proxy G_i (Eq. 14) first-order
  approximates, at ~131 forward passes per sample instead of one backward pass.

``answer_attn``
  Answer-grounded attention: in the same full forward, the attention mass the
  gold-answer tokens spend on each report token.

Output sidecar rows {"sample_id", "answer_loo": [...], "answer_attn": [...]}
aligned token-for-token with the score cache's ``token_ids``.
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments/v2x_edge_edge_8b.yaml")
    ap.add_argument("--score-cache", default=None)
    ap.add_argument("--processed", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    visible = str(config["experiment"].get("cuda_visible_devices", "0"))
    if "," in visible:
        raise SystemExit("This driver is restricted to exactly one visible GPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.experiments.prompts import ANSWER_REQUEST, locate_span
    from src.utils.io import read_jsonl, write_jsonl

    cache_path = Path(args.score_cache or config["score_cache"])
    out_path = Path(args.output or
                    cache_path.with_name(cache_path.stem + "_answer_aware.jsonl"))

    rows = list(read_jsonl(cache_path))
    processed = {str(r["row_idx"]): r for r in
                 read_jsonl(args.processed or config["dataset"]["processed"])}
    print(f"cache: {len(rows)} samples -> {cache_path}")

    done: dict[str, dict[str, list[float]]] = {}
    if out_path.exists():
        for row in read_jsonl(out_path):
            done[str(row["sample_id"])] = {
                "answer_loo": list(row.get("answer_loo", [])),
                "answer_attn": list(row.get("answer_attn", []))}
        print(f"resuming: {len(done)} rows on disk")

    def complete(sid, n):
        e = done.get(sid)
        return bool(e) and len(e["answer_loo"]) == n and len(e["answer_attn"]) == n

    pending = [r for r in rows if not complete(str(r["sample_id"]), len(r["token_ids"]))]
    if args.max_samples:
        pending = pending[:args.max_samples]
    if not pending:
        print("nothing to do")
        return

    rx_path = config["model"]["receiver_path"]
    tokenizer = AutoTokenizer.from_pretrained(rx_path, local_files_only=True)
    # Text-only causal LMs load directly; a vision-language coordinator (8B edge
    # coordinator in text-only mode) falls back to the image-text-to-text class,
    # mirroring run_channel_experiment._load_receiver.
    rx_kwargs = dict(dtype=torch.bfloat16, local_files_only=True,
                     attn_implementation="eager")
    try:
        model = AutoModelForCausalLM.from_pretrained(rx_path, **rx_kwargs).to("cuda:0").eval()
    except (ValueError, KeyError, RuntimeError):
        # gc is imported at module level; a local import here would shadow it and
        # break the later gc.collect() call when this branch is not taken.
        gc.collect(); torch.cuda.empty_cache()
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            rx_path, **rx_kwargs).to("cuda:0").eval()
    print(f"scorer (receiver LM): {rx_path}")
    eos_id = tokenizer.eos_token_id

    def answer_nll(ids, answer_len):
        labels = torch.full_like(ids, -100)
        labels[0, -answer_len:] = ids[0, -answer_len:]
        with torch.no_grad():
            out = model(input_ids=ids, labels=labels)
        return float(out.loss.detach())

    def flush():
        write_jsonl(out_path, [dict(sample_id=k, **done[k])
                               for k in sorted(done, key=lambda s: int(s))])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    for pos, row in enumerate(pending, 1):
        sid = str(row["sample_id"])
        token_ids = list(row["token_ids"])
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
        if tokenizer.convert_ids_to_tokens(
                prompt_ids[start:start + len(report_ids)]) != \
                tokenizer.convert_ids_to_tokens(report_ids):
            found = locate_span(prompt_ids, report_ids,
                                len(prompt_ids) - len(report_ids))
            if found < 0:
                print(f"  !! {sid}: cannot locate the report span; skipped")
                continue
            start = found
        end = start + len(report_ids)

        base = torch.tensor([prompt_ids + answer_ids], device="cuda:0")
        n_full = answer_nll(base, len(answer_ids))
        with torch.no_grad():
            att = torch.stack(model(input_ids=base,
                                    output_attentions=True).attentions)[:, 0].float()
        # answer queries look back at report keys (causal-compatible direction)
        a0 = len(prompt_ids)
        attn_score = att[:, :, a0:, start:end].sum(dim=2).mean(dim=(0, 1))

        loo = []
        for i in range(len(report_ids)):
            kept = prompt_ids[:start + i] + prompt_ids[start + i + 1:]
            ids = torch.tensor([kept + answer_ids], device="cuda:0")
            loo.append(answer_nll(ids, len(answer_ids)) - n_full)

        done[sid] = {"answer_loo": loo,
                     "answer_attn": [float(v) for v in attn_score]}
        del att, base
        if pos % 10 == 0 or pos == len(pending):
            flush()
            mx = max(loo) if loo else 0.0
            print(f"{pos}/{len(pending)} id={sid} tok={len(token_ids)} "
                  f"loo_max={mx:.3f} [{time.strftime('%H:%M:%S')}]")
        gc.collect()
        torch.cuda.empty_cache()

    flush()
    print(f"DONE: {len(done)} samples -> {out_path}")


if __name__ == "__main__":
    main()
