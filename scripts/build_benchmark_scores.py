#!/usr/bin/env python
"""Pre-compute the per-token ranking signals of every external baseline.

Four published families are supported. All of them rank the report's tokens and
never look at the pragmatic A/S/G scores, so the channel experiment isolates the
selection criterion.

``selective_context`` (Li et al., EMNLP 2023) and ``llmlingua`` (Jiang et al., ACL 2023)
    self-information  I(x_i) = -log p(x_i | x_1..x_{i-1}) of the *receiver* LM;
``longllmlingua`` (Jiang et al., 2024)
    question-aware contrastive relevance  I(x_i) - I(x_i | Q);
``llmlingua2`` (Xiao et al., ACL 2024 Findings)
    the keep-probability of a *trained* token classifier distilled from GPT-4
    (``microsoft/llmlingua-2-*``), aggregated from sub-words to report tokens;
``tokcom`` (Qiao et al., arXiv 2502.12096, 2025)
    cross-modal context attention: how much attention the *non-report* context
    (image + question + template) of the transmitter VLM pays to each report
    token -- the signal TokCom uses to decide which tokens to mask.

The receiver LM scores the first three families because it is the consumer of the
compressed prompt; TokCom is scored by the transmitter VLM because its cross-modal
context lives there.

Output is a JSONL sidecar, one row per sample::

    {"sample_id": str,
     "surprisal":    [...],   # context-only self-information
     "surprisal_q":  [...],   # question-conditioned self-information
     "llmlingua2":   [...],   # trained keep-probability
     "tokcom":       [...]}   # cross-modal context attention

All lists are aligned token-for-token with the sample's ``token_ids`` in the score
cache.  Models are loaded in two phases (receiver LM + classifier, then the VLM)
so the three never share the GPU.
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

FIELDS = ("surprisal", "surprisal_q", "llmlingua2", "tokcom")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/v2x_edge_edge_8b.yaml")
    parser.add_argument("--score-cache", default=None, help="override config score_cache")
    parser.add_argument("--processed", default=None, help="override config dataset.processed")
    parser.add_argument("--output", default=None,
                        help="default: alongside the cache as <stem>_surprisal.jsonl")
    parser.add_argument("--llmlingua2-path", default="models/llmlingua2-xlmr")
    parser.add_argument("--max-samples", type=int, default=None, help="cap for smoke runs")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    visible = str(config["experiment"].get("cuda_visible_devices", "0"))
    if "," in visible:
        raise SystemExit("This driver is restricted to exactly one visible GPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.experiments.prompts import build_choice_block
    from src.utils.io import read_jsonl, write_jsonl

    cache_path = Path(args.score_cache or config["score_cache"])
    out_path = Path(args.output or cache_path.with_name(cache_path.stem + "_surprisal.jsonl"))

    rows = list(read_jsonl(cache_path))
    processed = {str(r["row_idx"]): r for r in
                 read_jsonl(args.processed or config["dataset"]["processed"])}
    print(f"cache: {len(rows)} samples -> {cache_path}")

    done: dict[str, dict[str, list[float]]] = {}
    if out_path.exists():
        for row in read_jsonl(out_path):
            done[str(row["sample_id"])] = {f: list(row.get(f, [])) for f in FIELDS}
        print(f"resuming: {len(done)} rows on disk")

    def complete(sid: str, n_tokens: int) -> bool:
        entry = done.get(sid)
        return bool(entry) and all(len(entry[f]) == n_tokens for f in FIELDS)

    def incomplete():
        return [r for r in rows
                if not complete(str(r["sample_id"]), len(r["token_ids"]))]

    def flush() -> None:
        write_jsonl(out_path, [dict(sample_id=k, **done[k])
                               for k in sorted(done, key=lambda s: int(s))])

    pending = incomplete()
    if args.max_samples:
        pending = pending[:args.max_samples]
    if not pending:
        print("nothing to do")
        return

    qwen_tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["receiver_path"], local_files_only=True)

    # ------------------------------------------------ phase 1: text-side signals
    need_text = [r for r in pending
                 if not done.get(str(r["sample_id"]), {}).get("surprisal")
                 or not done.get(str(r["sample_id"]), {}).get("surprisal_q")
                 or not done.get(str(r["sample_id"]), {}).get("llmlingua2")]
    if need_text:
        run_text_phase(need_text, processed, config, args, qwen_tokenizer,
                       done, flush, torch)

    gc.collect(); torch.cuda.empty_cache()

    # ------------------------------------------ phase 2: cross-modal (TokCom)
    need_vlm = [r for r in pending
                if not done.get(str(r["sample_id"]), {}).get("tokcom")]
    if need_vlm:
        run_tokcom_phase(need_vlm, processed, config, qwen_tokenizer,
                         done, flush, torch)

    flush()
    print(f"DONE: {len(done)} samples -> {out_path}")


# --------------------------------------------------------------------- helpers
def run_text_phase(rows, processed, config, args, qwen_tokenizer, done, flush, torch):
    from transformers import (AutoModelForCausalLM, AutoModelForTokenClassification,
                              AutoTokenizer)
    from src.experiments.prompts import build_choice_block
    from src.utils.io import read_jsonl, write_jsonl  # noqa: F401  (parity with caller)

    # Text-only causal LMs load directly; a vision-language coordinator (e.g. the
    # 8B edge coordinator used in text-only mode) falls back to the
    # image-text-to-text class, mirroring run_channel_experiment._load_receiver.
    rx_path = config["model"]["receiver_path"]
    rx_kwargs = dict(dtype=torch.bfloat16, local_files_only=True)
    try:
        receiver = AutoModelForCausalLM.from_pretrained(rx_path, **rx_kwargs).to("cuda:0").eval()
    except (ValueError, KeyError, RuntimeError):
        # gc is imported at module level; a local import here would shadow it and
        # break the later gc.collect() calls when this branch is not taken.
        gc.collect(); torch.cuda.empty_cache()
        from transformers import AutoModelForImageTextToText

        receiver = AutoModelForImageTextToText.from_pretrained(
            rx_path, **rx_kwargs).to("cuda:0").eval()
    print(f"[phase 1] receiver LM: {rx_path}")

    classifier = clf_tokenizer = None
    if Path(args.llmlingua2_path).exists():
        clf_tokenizer = AutoTokenizer.from_pretrained(args.llmlingua2_path,
                                                      local_files_only=True)
        classifier = AutoModelForTokenClassification.from_pretrained(
            args.llmlingua2_path, dtype=torch.bfloat16,
            local_files_only=True).to("cuda:0").eval()
        id2label = {int(k): str(v) for k, v in classifier.config.id2label.items()}
        keep_id = next((i for i, name in id2label.items()
                        if name.lower() in ("true", "keep")), 1)
        print(f"[phase 1] llmlingua-2: {args.llmlingua2_path} (keep label {keep_id})")
    else:
        print(f"[phase 1] llmlingua-2 weights missing at {args.llmlingua2_path!r}; "
              f"the field will stay empty")

    def self_information(ids: torch.Tensor, start: int) -> list[float]:
        with torch.no_grad():
            logits = receiver(input_ids=ids).logits[0, start - 1:-1].float()
            values = -torch.log_softmax(logits, dim=-1).gather(
                -1, ids[0, start:][:, None]).squeeze(-1)
        return [float(v) for v in values]

    def keep_probabilities(text: str) -> list[float]:
        """LLMLingua-2 keep-probability per sub-word (specials excluded)."""
        enc = clf_tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=512, return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to("cuda:0") for k, v in enc.items()}
        with torch.no_grad():
            probs = torch.softmax(classifier(**enc).logits[0].float(), dim=-1)[:, keep_id]
        return [p for (s, e), p in zip(offsets, probs.tolist()) if e > s], \
            [span for span in offsets if span[1] > span[0]]

    def to_report_tokens(text: str, sub_probs, sub_offsets):
        """Aggregate sub-word probabilities onto the cached Qwen tokens."""
        enc = qwen_tokenizer(text, add_special_tokens=False,
                             return_offsets_mapping=True)
        spans, ids = enc["offset_mapping"], enc["input_ids"]
        out, cursor = [], 0
        for start, end in spans:
            best = 0.0
            k = cursor
            while k < len(sub_offsets) and sub_offsets[k][0] < end:
                if sub_offsets[k][1] > start and sub_probs[k] > best:
                    best = sub_probs[k]
                k += 1
            cursor = max(cursor, 0) if end > start else cursor
            out.append(best if end > start else 0.0)
        return ids, out

    for pos, row in enumerate(rows, 1):
        sid = str(row["sample_id"])
        token_ids = list(row["token_ids"])
        text = row["semantic_prompt"]

        ids = torch.tensor([token_ids], device="cuda:0")
        values = self_information(ids, start=1)
        values = [sum(values) / len(values)] + values

        sample = processed.get(sid, {})
        prefix = "Question: {q}\n\nCandidate choices:\n{c}".format(
            q=sample.get("question", ""), c=build_choice_block(sample.get("choices")))
        prefix_ids = qwen_tokenizer(prefix, add_special_tokens=False)["input_ids"]
        ids_q = torch.tensor([prefix_ids + token_ids], device="cuda:0")
        values_q = self_information(ids_q, start=len(prefix_ids))

        entry = done.setdefault(sid, {f: [] for f in FIELDS})
        entry["surprisal"], entry["surprisal_q"] = values, values_q

        if classifier is not None:
            sub_probs, sub_offsets = keep_probabilities(text)
            _, mapped = to_report_tokens(text, sub_probs, sub_offsets)
            if len(mapped) == len(token_ids):
                entry["llmlingua2"] = mapped
            else:
                print(f"  !! {sid}: llmlingua-2 re-tokenisation mismatch "
                      f"({len(mapped)} vs {len(token_ids)}); field left empty")

        del ids, ids_q
        if pos % 25 == 0 or pos == len(rows):
            flush()
            print(f"[phase 1] {pos}/{len(rows)} id={sid} [{time.strftime('%H:%M:%S')}]")
        gc.collect(); torch.cuda.empty_cache()

    del receiver
    if classifier is not None:
        del classifier, clf_tokenizer
    gc.collect(); torch.cuda.empty_cache()


def run_tokcom_phase(rows, processed, config, qwen_tokenizer, done, flush, torch):
    """TokCom (arXiv 2502.12096): cross-modal context attention per report token."""
    from transformers import AutoImageProcessor, AutoProcessor, Qwen3VLForConditionalGeneration
    from src.experiments.prompts import build_scoring_messages, locate_span
    from src.experiments.scoring import load_image

    tx_path = config["model"]["transmitter_path"]
    processor = AutoProcessor.from_pretrained(tx_path, local_files_only=True)
    processor.image_processor = AutoImageProcessor.from_pretrained(
        tx_path, min_pixels=config["model"]["image_min_pixels"],
        max_pixels=config["model"]["image_max_pixels"], local_files_only=True)
    transmitter = Qwen3VLForConditionalGeneration.from_pretrained(
        tx_path, dtype=torch.bfloat16, attn_implementation="eager",
        local_files_only=True).to("cuda:0").eval()
    print(f"[phase 2] transmitter VLM: {tx_path}")

    for pos, row in enumerate(rows, 1):
        sid = str(row["sample_id"])
        token_ids = list(row["token_ids"])
        text = row["semantic_prompt"]
        sample = processed.get(sid, {})

        image = load_image(sample.get("image"))
        messages = build_scoring_messages(image, sample.get("question", ""),
                                          sample.get("choices"), text)
        batch = processor.apply_chat_template(messages, tokenize=True,
                                              add_generation_prompt=True,
                                              return_dict=True, return_tensors="pt")
        sequence = batch["input_ids"][0].tolist()
        expected = qwen_tokenizer(text, add_special_tokens=False)["input_ids"]
        hint = len(processor.apply_chat_template(
            messages[:1], tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt")["input_ids"][0])
        p_start = locate_span(sequence, expected, hint)
        if p_start < 0:
            print(f"  !! {sid}: cannot locate the report span; skipped")
            continue
        p_end = p_start + len(expected)

        inputs = {k: (v.to("cuda:0") if hasattr(v, "to") else v)
                  for k, v in batch.items()}
        with torch.no_grad():
            out = transmitter(output_attentions=True, return_dict=True, **inputs)
        # attention: [layers, heads, query, key] (batch squeezed). Causal masking
        # means the context can never attend to the report, so the computable
        # cross-modal direction is report-token queries looking *back* at the
        # multimodal context (image + question + template): a token that is
        # grounded in the image/question spends much of its attention there, while
        # report-internal filler attends to its own neighbours instead.
        att = torch.stack(out.attentions)[:, 0].float()
        context_mass = att[:, :, p_start:p_end, :p_start].sum(dim=3).mean(dim=(0, 1))

        entry = done.setdefault(sid, {f: [] for f in FIELDS})
        if p_end - p_start == len(token_ids):
            entry["tokcom"] = [float(v) for v in context_mass]
        else:
            print(f"  !! {sid}: report span {p_end - p_start} != {len(token_ids)} tokens")

        del out, att, batch, inputs, image
        if pos % 10 == 0 or pos == len(rows):
            flush()
            print(f"[phase 2] {pos}/{len(rows)} id={sid} "
                  f"mean={sum(entry['tokcom'])/max(1,len(entry['tokcom'])):.4f} "
                  f"[{time.strftime('%H:%M:%S')}]")
        gc.collect(); torch.cuda.empty_cache()

    del transmitter
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
