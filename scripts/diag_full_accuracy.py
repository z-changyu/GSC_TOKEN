#!/usr/bin/env python
"""Diagnose why full-report accuracy on V2X-QA is only ~27%.

Re-runs the receiver on the *full* semantic reports already cached, and records,
per sample: raw generation, parse mode, strict correctness, plus a *lenient*
correctness that treats a prediction as right when its text contains (or is
contained by) the ground-truth option. The gap between strict and lenient tells
us how much of the 27% is a parsing/instruction-following problem versus a
genuinely uninformative report.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def strict_match(pred, truth):
    from src.metrics.qa import normalize_answer
    return normalize_answer(pred) == normalize_answer(truth)


def lenient_match(raw, truth):
    """True if raw contains the option text (or vice-versa), ignoring a/an/the."""
    from src.metrics.qa import normalize_answer
    r, t = normalize_answer(raw), normalize_answer(truth)
    return (t and t in r) or (r and r in t)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/v2x_edge_edge.yaml")
    parser.add_argument("--score-cache", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    visible = str(config["experiment"].get("cuda_visible_devices", "0"))
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.experiments.prompts import ANSWER_REQUEST, canonical_prediction
    from src.utils.io import read_jsonl

    device = args.device or "cuda:0"
    cache = {str(r["sample_id"]): r
             for r in read_jsonl(args.score_cache or config["score_cache"])}
    processed = {str(r["row_idx"]): r for r in read_jsonl(config["dataset"]["processed"])}
    manifest = list(read_jsonl(args.manifest or config["dataset"]["manifest"]))

    rx_path = config["model"]["receiver_path"]
    rx_tokenizer = AutoTokenizer.from_pretrained(rx_path, local_files_only=True)
    receiver = AutoModelForCausalLM.from_pretrained(
        rx_path, dtype=torch.bfloat16, attn_implementation="eager",
        local_files_only=True).to(device).eval()

    mode_counter: Counter = Counter()
    pred_letter: Counter = Counter()
    strict_ok = lenient_ok = 0
    records = []
    n = 0
    for entry in manifest:
        if args.max_samples is not None and n >= args.max_samples:
            break
        sid = str(entry["sample_id"])
        sc = cache.get(sid)
        row = processed.get(sid)
        if sc is None or row is None:
            continue
        n += 1
        choices = list(row["choices"])
        answer = int(row["answer"])
        truth = str(choices[answer])
        report = sc.get("semantic_prompt") or ""
        prompt = rx_tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{ANSWER_REQUEST}\n\n{report}"}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inp = rx_tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = receiver.generate(**inp, max_new_tokens=int(config["receiver"]["max_new_tokens"]),
                                    do_sample=False)
        raw = rx_tokenizer.decode(out[0, inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
        pred, mode = canonical_prediction(raw, choices)
        mode_counter[mode] += 1
        if mode in ("exact", "leading_letter", "unique_substring"):
            idx = choices.index(pred) if pred in choices else -1
            pred_letter[chr(65 + idx) if idx >= 0 else "?"] += 1
        else:
            pred_letter["RAW"] += 1
        s_ok = strict_match(pred, truth)
        l_ok = lenient_match(raw, truth)
        strict_ok += int(s_ok)
        lenient_ok += int(l_ok)
        records.append({
            "sample_id": sid, "question": row["question"], "choices": choices,
            "truth": truth, "raw": raw, "pred": pred, "mode": mode,
            "strict": s_ok, "lenient": l_ok,
            "report": report[:400],
        })

    out_path = Path("outputs/diag_full.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"total={n}")
    print(f"strict_accuracy = {strict_ok/n:.3f}  (current pipeline)")
    print(f"lenient_accuracy = {lenient_ok/n:.3f}  (if we matched by text containment)")
    print(f"parse modes: {dict(mode_counter)}")
    print(f"predicted letters: {dict(pred_letter)}")
    # Show a few strict-wrong-but-lenient-right (parsing failures) and a few genuine wrong.
    flip = [r for r in records if not r["strict"] and r["lenient"]][:3]
    genuine = [r for r in records if not r["strict"] and not r["lenient"]][:3]
    print("\n--- parsing-failure examples (wrong by strict, right by lenient) ---")
    for r in flip:
        print(f"[s{r['sample_id']}] truth={r['truth']!r} pred(raw)={r['raw']!r}")
    print("\n--- genuine wrong examples (wrong by both) ---")
    for r in genuine:
        print(f"[s{r['sample_id']}] Q={r['question']!r} truth={r['truth']!r} raw={r['raw']!r}")


if __name__ == "__main__":
    main()
