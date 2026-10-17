#!/usr/bin/env python
"""Receiver-in-the-loop curation of the distillation target set.

Motivation
----------
Plain sequence-level distillation on 3000 teacher reports failed for the 4B student
(76.7% -> 65.0% at lr 1e-4, 78.3% at lr 2e-5, i.e. noise). The teacher's report is a
*good* target on average, but the student already solves most samples, so the
gradient is dominated by examples it has nothing to learn from, while the style
imitation (teacher reports are ~20% longer) actively dilutes the decisive evidence.

This script builds a *verified* target set instead of trusting the teacher blindly:

1. verify each teacher report with the deployed receiver (text-only coordinator,
   ``v6_norestore`` protocol) -> ``teacher_correct``;
2. generate the current student's report for the same samples and verify it with
   the same receiver -> ``student_correct``;
3. expose three selectable subsets:
   - ``teacher_correct``: every target the receiver can actually use;
   - ``hard``: ``teacher_correct and not student_correct`` -- exactly the samples
     where the teacher demonstrably knows something the student does not;
   - ``all``: the unfiltered set (what plain distillation used).

The receiver never sees the ground truth in its own prompt, so no answer is leaked
into the target reports.

Usage
-----
Run with an interpreter that can load both the student and the receiver (the repo
env is fine; the student here is Qwen3-VL-4B, not Qwen3.5).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/v2x_edge_edge_main_4b8b.yaml")
    parser.add_argument("--reports", default="outputs/distill/teacher_reports_8b.jsonl")
    parser.add_argument("--student", default="models/Qwen3-VL-4B")
    parser.add_argument("--num-samples", type=int, default=600)
    parser.add_argument("--output-dir", default="outputs/distill/curated")
    parser.add_argument("--student-reports", default=None,
                        help="cache of already-generated student reports (JSONL)")
    args = parser.parse_args()

    config_path = Path(args.config)
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["experiment"].get("cuda_visible_devices", "0"))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import torch
    from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoProcessor

    from src.experiments.prompts import (ANSWER_REQUEST, build_choice_block,
                                         build_transmitter_messages,
                                         canonical_prediction,
                                         select_transmitter_request)
    from src.experiments.receiver_protocol import answer_with_protocol
    from src.metrics.qa import exact_match
    from src.utils.io import read_jsonl

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [r for r in read_jsonl(args.reports) if r.get("report")][:args.num_samples]
    print(f"[data] {len(rows)} teacher reports from {args.reports}")

    request_template = select_transmitter_request(config)
    max_prompt_tokens = int(config["transmitter"]["max_prompt_tokens"])
    max_new_tokens = int(config["transmitter"]["max_new_tokens"])
    rx_max_new_tokens = int(config["receiver"]["max_new_tokens"])

    # ------------------------------------------- phase 1: student reports
    stu_cache_path = Path(args.student_reports) if args.student_reports else (
        out_dir / f"student_reports_{Path(args.student).name}.jsonl")
    student_reports: dict[str, str] = {}
    if stu_cache_path.exists():
        student_reports = {str(r["sample_id"]): r["report"] for r in read_jsonl(stu_cache_path)}
        print(f"[phase1] reused {len(student_reports)} cached student reports")
    if len(student_reports) < len(rows):
        processor = AutoProcessor.from_pretrained(args.student, local_files_only=True)
        processor.image_processor = AutoImageProcessor.from_pretrained(
            args.student, min_pixels=config["model"]["image_min_pixels"],
            max_pixels=config["model"]["image_max_pixels"], local_files_only=True)
        student = None
        import transformers as _tf
        for cls_name in ("Qwen3VLForConditionalGeneration", "AutoModelForImageTextToText"):
            cls = getattr(_tf, cls_name, None)
            if cls is None:
                continue
            try:
                student = cls.from_pretrained(
                    args.student, dtype=torch.bfloat16, attn_implementation="eager",
                    local_files_only=True)
                break
            except Exception as exc:                 # noqa: BLE001
                print(f"[warn] student load via {cls_name} failed: {type(exc).__name__}")
        if student is None:
            raise SystemExit(f"could not load student {args.student}")
        try:
            student = student.to("cuda:0")
        except (ValueError, RuntimeError):
            pass
        student = student.eval()
        tok = processor.tokenizer
        from PIL import Image

        t0 = time.perf_counter()
        for n, row in enumerate(rows, 1):
            sid = str(row["sample_id"])
            if sid in student_reports:
                continue
            request = request_template.format(
                question=row["question"], choices=build_choice_block(row["choices"]))
            image = Image.open(row["image"]).convert("RGB") if row.get("image") else None
            content = []
            if image is not None:
                content.append({"type": "image", "image": image})
            content.append({"type": "text", "text": request})
            inputs = processor.apply_chat_template(
                build_transmitter_messages(content), tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt", enable_thinking=False)
            inputs = {k: (v.to("cuda:0") if hasattr(v, "to") else v) for k, v in inputs.items()}
            with torch.no_grad():
                out = student.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            text = tok.decode(out[0, inputs["input_ids"].shape[1]:],
                              skip_special_tokens=True).strip()
            ids = tok(text, add_special_tokens=False)["input_ids"]
            if len(ids) > max_prompt_tokens:
                text = tok.decode(ids[:max_prompt_tokens], skip_special_tokens=True).strip()
            student_reports[sid] = text
            del inputs, out
            gc.collect(); torch.cuda.empty_cache()
            if n % 25 == 0 or n == len(rows):
                rate = (time.perf_counter() - t0) / max(1, n)
                print(f"  student report {n}/{len(rows)} ({rate:.1f}s each)")
        with stu_cache_path.open("w", encoding="utf-8") as fh:
            for row in rows:
                sid = str(row["sample_id"])
                fh.write(json.dumps({"sample_id": sid, "report": student_reports[sid]},
                                    ensure_ascii=False) + "\n")
        del student, processor
        gc.collect(); torch.cuda.empty_cache()
        print(f"[phase1] student reports -> {stu_cache_path}")

    # ------------------------------------------- phase 2: receiver verification
    rx_path = config["model"]["receiver_path"]
    rx_tokenizer = AutoProcessor.from_pretrained(rx_path, local_files_only=True).tokenizer
    try:
        receiver = AutoModelForCausalLM.from_pretrained(
            rx_path, dtype=torch.bfloat16, attn_implementation="eager",
            local_files_only=True).to("cuda:0").eval()
    except Exception:                                # noqa: BLE001
        from transformers import AutoModelForImageTextToText
        receiver = AutoModelForImageTextToText.from_pretrained(
            rx_path, dtype=torch.bfloat16, attn_implementation="eager",
            local_files_only=True).to("cuda:0").eval()
    print(f"[phase2] receiver loaded: {rx_path}")

    def verify(row: dict, report: str) -> tuple[bool, str, str]:
        result = answer_with_protocol(
            model=receiver, tokenizer=rx_tokenizer, row=row, received=report,
            survived_tokens=1, total_tokens=1,          # ideal channel
            answer_request=ANSWER_REQUEST, max_new_tokens=rx_max_new_tokens,
            protocol="v6", restore_threshold=0.0)
        truth = str(row["choices"][int(row["answer"])])
        ok = bool(exact_match(result["prediction"], truth))
        gc.collect(); torch.cuda.empty_cache()
        return ok, result["prediction"], truth

    curated = []
    counts = Counter()
    for n, row in enumerate(rows, 1):
        sid = str(row["sample_id"])
        t_ok, t_pred, truth = verify(row, row["report"])
        s_ok, s_pred, _ = verify(row, student_reports[sid])
        counts["teacher_correct"] += int(t_ok)
        counts["student_correct"] += int(s_ok)
        counts["hard"] += int(t_ok and not s_ok)
        curated.append({"sample_id": sid, "question": row["question"],
                        "choices": row["choices"], "answer": row["answer"],
                        "image": row["image"], "report": row["report"],
                        "teacher_correct": int(t_ok), "teacher_prediction": t_pred,
                        "student_correct": int(s_ok), "student_prediction": s_pred,
                        "ground_truth": truth})
        if n % 25 == 0 or n == len(rows):
            print(f"  verified {n}/{len(rows)}  {dict(counts)}")

    with (out_dir / "curated.jsonl").open("w", encoding="utf-8") as fh:
        for row in curated:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    for mode, keep in (("teacher_correct", lambda r: r["teacher_correct"]),
                       ("hard", lambda r: r["teacher_correct"] and not r["student_correct"]),
                       ("all", lambda r: True)):
        subset = [r for r in curated if keep(r)]
        with (out_dir / f"targets_{mode}.jsonl").open("w", encoding="utf-8") as fh:
            for row in subset:
                fh.write(json.dumps({k: row[k] for k in
                                     ("sample_id", "question", "choices", "answer",
                                      "image", "report")}, ensure_ascii=False) + "\n")
        print(f"[write] targets_{mode}.jsonl  n={len(subset)}")

    summary = {"reports": args.reports, "student": args.student,
               "receiver_path": rx_path, "samples": len(rows), **counts}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
