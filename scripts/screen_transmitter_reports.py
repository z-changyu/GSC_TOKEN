#!/usr/bin/env python
"""Screen a candidate transmitter VLM by report quality, before paying for the
full pragmatic-scoring pipeline.

Why this exists
---------------
`run_channel_experiment.py` never loads the transmitter: it replays reports out
of a pre-built ``SampleScores`` cache. Swapping the transmitter therefore costs a
full cache rebuild (three sidecars, ~20-40 min on 150 questions) *and* the
scoring code currently assumes Qwen3-VL internals (``get_rope_index`` signature,
attention weights on every layer). A newer hybrid-architecture model such as
Qwen3.5-4B breaks those assumptions.

This script answers the only question that decides whether that investment is
worth it: **does the candidate write reports that the deployed receiver answers
more accurately?** It gives every arm the full (undamaged) report -- i.e. the
``full`` method at an ideal channel -- so the comparison is purely about report
content.

Arms
----
``baseline``  reports already cached from the incumbent transmitter
              (``--baseline-cache``, e.g. outputs/cache/v2xqa_scores_4b.jsonl).
``candidate`` reports generated on the fly by ``--transmitter``.

Both arms answer under the fair protocol (``v6_norestore``: RSU-issued task
intent + option-letter output, restoration off), so numbers are comparable to
``results/channel_main_4b8b_fair``.

Usage
-----
Run it with an interpreter whose ``transformers`` can load the candidate. For
Qwen3.5 models that means transformers 5.x, which needs the ``torch.accelerator``
shim injected before ``transformers`` is imported; on torch >= 2.6 the shim is a
no-op.
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


# ---------------------------------------------------------------- torch compat
def _install_torch_compat() -> bool:
    """Inject the torch>=2.6 Accelerator API that transformers 5.x requires."""
    import torch

    if hasattr(torch, "accelerator"):
        return False

    class _Accelerator:
        @staticmethod
        def current_accelerator(check_available: bool = False):
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")

        @staticmethod
        def is_available() -> bool:
            return torch.cuda.is_available()

        @staticmethod
        def device_count() -> int:
            return torch.cuda.device_count()

        @staticmethod
        def current_device() -> int:
            return torch.cuda.current_device()

        @staticmethod
        def set_device(index) -> None:
            torch.cuda.set_device(index)

    torch.accelerator = _Accelerator()
    return True


_SHIM_INSTALLED = _install_torch_compat()

import yaml  # noqa: E402
from PIL import Image  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/v2x_edge_edge_main_4b8b.yaml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--processed", default=None)
    parser.add_argument("--baseline-cache", default="outputs/cache/v2xqa_scores_4b.jsonl")
    parser.add_argument("--transmitter", default="models/Qwen3.5-4B",
                        help="candidate transmitter VLM path ('' to skip generation)")
    parser.add_argument("--candidate-reports", default=None,
                        help="JSONL of pre-generated candidate reports ({sample_id, report}); "
                             "skips local generation (use for hosted-API teachers)")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--prompt-version", default=None,
                        help="override transmitter.prompt_version (v1/v2/v3)")
    parser.add_argument("--output", default="outputs/screen_report_quality.json")
    parser.add_argument("--reports-out", default=None,
                        help="JSONL of candidate reports, for reuse/inspection")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.prompt_version:
        config.setdefault("transmitter", {})["prompt_version"] = args.prompt_version
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["experiment"].get("cuda_visible_devices", "0"))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import torch
    from transformers import (AutoImageProcessor, AutoModelForCausalLM,
                              AutoProcessor)

    from src.experiments.prompts import (ANSWER_REQUEST, build_choice_block,
                                         build_transmitter_messages,
                                         canonical_prediction,
                                         select_transmitter_request)
    from src.experiments.receiver_protocol import answer_with_protocol
    from src.metrics.qa import exact_match
    from src.utils.io import read_jsonl

    manifest_path = args.manifest or config["dataset"]["manifest"]
    processed_path = args.processed or config["dataset"]["processed"]
    request_template = select_transmitter_request(config)
    max_prompt_tokens = int(config["transmitter"]["max_prompt_tokens"])
    max_new_tokens = int(config["transmitter"]["max_new_tokens"])
    rx_max_new_tokens = int(config["receiver"]["max_new_tokens"])

    print(f"[env] transformers shim installed: {_SHIM_INSTALLED}")
    processed = {str(r["row_idx"]): r for r in read_jsonl(processed_path)}
    for row in processed.values():
        row["_request"] = request_template.format(
            question=row["question"], choices=build_choice_block(row["choices"]))

    baseline = {str(r["sample_id"]): r for r in read_jsonl(args.baseline_cache)}
    manifest = [str(e["sample_id"]) for e in read_jsonl(manifest_path)
                if str(e["sample_id"]) in processed and str(e["sample_id"]) in baseline]
    if args.num_samples:
        manifest = manifest[:args.num_samples]
    print(f"[data] {len(manifest)} samples "
          f"(manifest={Path(manifest_path).name}, baseline={Path(args.baseline_cache).name})")

    # ---------------------------------------------------- phase 1: candidate reports
    reports: dict[str, str] = {}
    gen_seconds: list[float] = []
    if args.candidate_reports:
        loaded = {str(r["sample_id"]): r["report"] for r in read_jsonl(args.candidate_reports)}
        reports = {sid: loaded[sid] for sid in manifest if sid in loaded}
        print(f"[phase0] loaded {len(reports)}/{len(manifest)} pre-generated reports "
              f"from {args.candidate_reports}")
    if args.transmitter and not reports:
        processor = AutoProcessor.from_pretrained(args.transmitter, local_files_only=True)
        try:
            processor.image_processor = AutoImageProcessor.from_pretrained(
                args.transmitter, min_pixels=config["model"]["image_min_pixels"],
                max_pixels=config["model"]["image_max_pixels"], local_files_only=True)
        except Exception as exc:                     # noqa: BLE001
            print(f"[warn] image_processor override skipped: {type(exc).__name__}: {exc}")
        model = None
        for cls_name in ("AutoModelForImageTextToText",):
            import transformers
            cls = getattr(transformers, cls_name)
            try:
                model = cls.from_pretrained(args.transmitter, dtype=torch.bfloat16,
                                            attn_implementation="eager",
                                            local_files_only=True).to("cuda:0").eval()
                break
            except Exception as exc:                 # noqa: BLE001
                print(f"[warn] {cls_name} load failed: {type(exc).__name__}: {exc}")
        if model is None:
            raise SystemExit("could not load the candidate transmitter")
        tokenizer = processor.tokenizer

        for n, sid in enumerate(manifest, 1):
            row = processed[sid]
            image = Image.open(row["image"]).convert("RGB") if row.get("image") else None
            content = []
            if image is not None:
                content.append({"type": "image", "image": image})
            content.append({"type": "text", "text": row["_request"]})
            inputs = processor.apply_chat_template(
                build_transmitter_messages(content), tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt", enable_thinking=False)
            inputs = {k: (v.to("cuda:0") if hasattr(v, "to") else v) for k, v in inputs.items()}
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            text = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True).strip()
            gen_seconds.append(time.perf_counter() - t0)
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(ids) > max_prompt_tokens:
                text = tokenizer.decode(ids[:max_prompt_tokens], skip_special_tokens=True).strip()
            reports[sid] = text
            del inputs, out
            gc.collect(); torch.cuda.empty_cache()
            if n % 5 == 0 or n == len(manifest):
                print(f"  report {n}/{len(manifest)} id={sid} chars={len(text)} "
                      f"t={gen_seconds[-1]:.1f}s")

        if args.reports_out:
            Path(args.reports_out).parent.mkdir(parents=True, exist_ok=True)
            with Path(args.reports_out).open("w", encoding="utf-8") as fh:
                for sid in manifest:
                    fh.write(json.dumps({"sample_id": sid, "report": reports[sid]},
                                        ensure_ascii=False) + "\n")

        del model, processor
        gc.collect(); torch.cuda.empty_cache()
        print(f"[phase1] {len(reports)} candidate reports, "
              f"mean {sum(gen_seconds)/len(gen_seconds):.1f}s/report")

    # ---------------------------------------------------- phase 2: receiver scoring
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

    records = []
    arms = {"baseline": lambda sid: baseline[sid]["semantic_prompt"]}
    if reports:
        arms["candidate"] = lambda sid: reports[sid]
    for arm, getter in arms.items():
        ok = 0
        for sid in manifest:
            row = processed[sid]
            report = getter(sid)
            result = answer_with_protocol(
                model=receiver, tokenizer=rx_tokenizer, row=row, received=report,
                survived_tokens=1, total_tokens=1,       # ideal channel: never restore
                answer_request=ANSWER_REQUEST, max_new_tokens=rx_max_new_tokens,
                protocol="v6", restore_threshold=0.0)
            truth = str(row["choices"][row["answer"]])
            correct = exact_match(result["prediction"], truth)
            ok += int(correct)
            records.append({"arm": arm, "sample_id": sid, "correct": int(correct),
                            "prediction": result["prediction"], "truth": truth,
                            "parse_mode": result["parse_mode"], "raw": result["raw"]})
            gc.collect(); torch.cuda.empty_cache()
        print(f"[{arm}] full-report accuracy = {ok/len(manifest):.3f}  (n={len(manifest)})")

    summary = {
        "manifest": str(manifest_path),
        "baseline_cache": str(args.baseline_cache),
        "candidate_transmitter": args.transmitter,
        "receiver_path": rx_path,
        "protocol": "v6_norestore (task intent + letter output, no restoration)",
        "channel": "ideal (full undamaged report)",
        "samples": len(manifest),
        "accuracy": {arm: sum(r["correct"] for r in records if r["arm"] == arm) / len(manifest)
                     for arm in arms},
        "mean_report_seconds": (sum(gen_seconds) / len(gen_seconds)) if gen_seconds else None,
        "records": records,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"},
                     indent=2, ensure_ascii=False))
    print(f"[done] -> {out_path}")


if __name__ == "__main__":
    main()
