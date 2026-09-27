#!/usr/bin/env python
"""E3: tiered cloud-edge-terminal pipeline smoke experiment.

Data flow (方案 A 三级架构, docs/场景重构与端到端数据流.md §3):

  V (端, vehicle)  Qwen3-VL-2B looks at the image and writes a SHORT coarse
                   semantic report (resource-limited device).
  U (上行)         the coarse report crosses a lossy low-rate uplink
                   (uniform strongest protection, realized-rate erasure +
                   packet loss). Some tokens are erased.
  C (云, cloud)    Qwen3-VL-8B (text-only) repairs and refines the damaged
                   report into a clean self-contained semantic report, then
                   scores its own output with the pragmatic scorer
                   (text-only: the cloud never sees the image).
  D (下行)         the refined report is broadcast through the standard
                   reliable-budget downlink (full / adaptive_sc / proposed).
  A (端, terminal) Qwen3-1.7B in-vehicle LLM answers from the surviving text.

Direct baseline: the same pipeline with the cloud stage skipped -- the coarse
report itself is scored by the vehicle (2B, with image) and broadcast through
the identical downlink. The comparison isolates the value of cloud refinement.

All model loads are strictly sequential (load -> run phase -> free) so the
whole pipeline fits one 24 GiB GPU.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/tiered_pipeline_smoke.yaml")
    parser.add_argument("--num-samples", type=int, help="override for smoke testing")
    parser.add_argument("--output-dir", default="outputs/tiered_pipeline")
    parser.add_argument("--methods", nargs="*")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    visible = str(config["experiment"].get("cuda_visible_devices", "0"))
    if "," in visible:
        raise SystemExit("single GPU only")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    from transformers import (AutoModelForCausalLM, AutoProcessor, AutoTokenizer,
                              AutoModelForImageTextToText)

    from src.channel import build_channel, decoding_probability
    from src.channel.link import ProtectionCodebook
    from src.channel.reliable_capacity import reliable_spectral_efficiency
    from src.experiments.prompts import (ANSWER_REQUEST, build_choice_block,
                                         build_transmitter_messages,
                                         canonical_prediction, locate_span)
    from src.experiments.scoring import PragmaticScorer, combine_scores, load_image
    from src.metrics.communication import token_retention_ratio
    from src.metrics.qa import exact_match
    from src.scheduling import select_level
    from src.utils.io import atomic_json, read_jsonl
    from src.utils.logging import configure_logging, create_run_directory

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run = create_run_directory("outputs/runs", "tiered_pipeline", config)
    logger = configure_logging(run / "run.log")

    def load_channel_spec(path: str) -> dict:
        return yaml.safe_load(Path(path).read_text())

    def gpu_gib() -> float:
        return torch.cuda.memory_allocated() / 2 ** 30

    def collect() -> None:
        gc.collect()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ data
    processed = {str(row["row_idx"]): row for row in read_jsonl(config["dataset"]["processed"])}
    manifest = [str(entry["sample_id"])
                for entry in read_jsonl(config["dataset"]["manifest"])]
    manifest = [sid for sid in manifest if processed.get(sid, {}).get("image")]
    limit = args.num_samples or int(config["experiment"]["num_samples"])
    manifest = manifest[:limit]
    logger.info("tiered pipeline: %d samples", len(manifest))

    codebook = ProtectionCodebook(
        levels=tuple(float(v) for v in config["radio"]["protection"]["levels"]),
        information_bits_per_token=float(config["radio"]["protection"]["information_bits_per_token"]))
    bandwidth = float(config["radio"]["bandwidth"])
    slot_duration = float(config["radio"]["slot_duration"])
    epsilon_out = float(config["radio"]["epsilon_out"])
    t_max = int(config["scheduling"]["t_max"])
    do_refine = bool(config["scheduling"].get("refine", True))
    weights = tuple(float(w) for w in config["pragmatic"]["weights"])
    beta = float(config["pragmatic"]["beta"])
    methods = args.methods or list(config["methods"])
    img_kw = dict(min_pixels=int(config["model"]["image_min_pixels"]),
                  max_pixels=int(config["model"]["image_max_pixels"]))

    # Tiered vehicle prompt: the coarse report must stay short so the evidence
    # is never cut by the max_new_tokens cap (smoke run showed cap truncation).
    VEHICLE_REQUEST = (
        "Question: {question}\n\nCandidate choices:\n{choices}\n\n"
        "Write a short semantic description that a text-only model can use to "
        "answer the question. Requirements, in this order: restate the question "
        "in a few words; list every candidate choice verbatim; for EACH choice "
        "give ONE short sentence of visible evidence that supports or contradicts "
        "it. Be extremely concise: the WHOLE description must stay under 80 "
        "tokens. Do not state or imply which choice is correct. Output the "
        "description only.")

    def transmitter_request(row) -> str:
        return VEHICLE_REQUEST.format(question=row["question"],
                                      choices=build_choice_block(row["choices"]))

    def load_vlm(path: str):
        proc = AutoProcessor.from_pretrained(path, local_files_only=True, **img_kw)
        model = AutoModelForImageTextToText.from_pretrained(
            path, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager")
        model = model.to("cuda:0").eval()
        return model, proc

    def generate(model, processor, messages, max_new_tokens: int) -> str:
        # Qwen3-VL processors require structured content parts, even for
        # text-only conversations.
        fixed = [{"role": m["role"],
                  "content": m["content"] if isinstance(m["content"], list)
                  else [{"type": "text", "text": m["content"]}]}
                 for m in messages]
        inputs = processor.apply_chat_template(fixed, tokenize=True,
                                               add_generation_prompt=True,
                                               return_dict=True, return_tensors="pt")
        inputs = {k: (v.to("cuda:0") if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = processor.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:],
                                          skip_special_tokens=True).strip()
        del out, inputs
        torch.cuda.empty_cache()
        return text

    def score_given_text(scorer: PragmaticScorer, row: dict, image, prompt_text: str):
        """Pragmatic A/S/G scores for a GIVEN report text (no regeneration).

        Mirrors PragmaticScorer.score() from the scoring step onward, so the
        cloud can score its refined text and the vehicle can score its coarse
        text without re-generating anything.
        """
        from src.experiments.prompts import build_scoring_messages

        tokenizer = scorer.tokenizer
        token_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        max_prompt = scorer.max_prompt_tokens
        if len(token_ids) > max_prompt:
            token_ids = token_ids[:max_prompt]
            prompt_text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
            token_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        if not token_ids:
            raise RuntimeError(f"empty report for sample {row.get('row_idx')}")
        token_strings = tokenizer.convert_ids_to_tokens(token_ids)

        torch.cuda.reset_peak_memory_stats()
        messages = build_scoring_messages(image, row["question"], row["choices"], prompt_text)
        batch = scorer.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt")
        expected = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        sequence_list = batch["input_ids"][0].tolist()
        hint = len(scorer.processor.apply_chat_template(
            messages[:1], tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt")["input_ids"][0])
        p_start = locate_span(sequence_list, expected, hint)
        if p_start < 0:
            raise RuntimeError(f"cannot locate report span for sample {row.get('row_idx')}")
        p_end = p_start + len(expected)

        input_ids = batch["input_ids"].to("cuda:0")
        attention_mask = batch["attention_mask"].to("cuda:0")
        answer_ids = tokenizer(str(row["choices"][row["answer"]]),
                               add_special_tokens=False)["input_ids"] + [scorer.im_end_id]
        tail = torch.tensor([answer_ids], device="cuda:0")
        input_ids = torch.cat([input_ids, tail], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(tail)], dim=1)
        labels = torch.full_like(input_ids, -100)
        labels[0, -len(answer_ids):] = tail[0]
        pixel_values = batch["pixel_values"].to("cuda:0") if "pixel_values" in batch else None
        grid = batch["image_grid_thw"].to("cuda:0") if "image_grid_thw" in batch else None
        position_ids, _ = scorer.transmitter.model.get_rope_index(
            input_ids, grid, None, attention_mask=attention_mask)
        embeddings = scorer.transmitter.get_input_embeddings()(input_ids).detach().to(torch.bfloat16)
        embeddings.requires_grad_(True)
        output = scorer.transmitter(inputs_embeds=embeddings, attention_mask=attention_mask,
                                    position_ids=position_ids, pixel_values=pixel_values,
                                    image_grid_thw=grid, labels=labels,
                                    output_attentions=True, return_dict=True)
        prompt_slice = slice(p_start, p_end)
        prompt_attentions = tuple(layer[:, :, prompt_slice, prompt_slice]
                                  for layer in output.attentions)
        attention = scorer.attention_scorer.score(prompt_attentions, token_ids, token_strings)
        semantic = scorer.semantic_scorer.score(embeddings.detach()[0, prompt_slice].float(),
                                                attention.scores.float(), token_ids, token_strings)
        gradient = scorer.gradient_scorer.score(output.loss, embeddings, token_ids, token_strings,
                                                mask_embedding=scorer.mask_embedding,
                                                positions=prompt_slice)
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        seq_len = int(input_ids.shape[1])
        del output, embeddings, batch, input_ids, labels, position_ids
        torch.cuda.empty_cache()
        return {"token_ids": token_ids, "num_tokens": len(token_ids),
                "attention": [float(v) for v in attention.scores],
                "semantic": [float(v) for v in semantic.scores],
                "gradient": [float(v) for v in gradient.scores],
                "sequence_length": seq_len, "peak_memory_gib": peak}

    # ======================================================================
    # Phase V (端): vehicle 2B writes the coarse report and scores it
    # ======================================================================
    logger.info("[phase V] vehicle VLM: %s", config["model"]["vehicle_path"])
    vehicle, v_proc = load_vlm(config["model"]["vehicle_path"])
    v_cfg = dict(config)
    v_cfg["transmitter"] = dict(config["vehicle"])
    v_scorer = PragmaticScorer(vehicle, v_proc, v_cfg)

    stages = []
    for sid in manifest:
        row = processed[sid]
        image = load_image(row["image"])
        row = dict(row)
        row["_request"] = transmitter_request(row)
        coarse = generate(vehicle, v_proc, build_transmitter_messages(
            [{"type": "image", "image": image},
             {"type": "text", "text": row["_request"]}]),
            int(config["vehicle"]["max_new_tokens"]))
        coarse_scored = score_given_text(v_scorer, row, image, coarse)
        stages.append({"sample_id": sid, "coarse": coarse, "coarse_scores": coarse_scored})
        logger.info("  [V] %s coarse=%d tokens", sid, coarse_scored["num_tokens"])
    # Drop caller-scope references explicitly: deleting a function-local name
    # inside a helper cannot release main()'s bindings.
    v_scorer = vehicle = v_proc = None
    collect()
    logger.info("[phase V] freed, %.2f GiB still allocated", gpu_gib())

    # ======================================================================
    # Phase U (上行): lossy vehicle -> cloud uplink, uniform strongest protection
    # ======================================================================
    up_spec = load_channel_spec(config["uplink"]["channel"])
    up_channel = build_channel({**up_spec, "snr_db": float(config["uplink"]["snr_db"]),
                                "seed": int(config["experiment"]["channel_seed"])})
    up_level = int(config["uplink"]["protection_level"])
    up_rate = float(codebook.code_rates[up_level - 1])
    up_tokenizer = AutoProcessor.from_pretrained(
        config["model"]["vehicle_path"], local_files_only=True).tokenizer
    for st in stages:
        state = up_channel.sample((1,))
        realized = math.log2(1.0 + float(state.snr_linear) * float(state.h.abs().pow(2)[0]))
        rng = random.Random(int(config["experiment"]["channel_seed"]) * 7919
                            + int(st["sample_id"]))
        kept, lost = [], 0
        for token in st["coarse_scores"]["token_ids"]:
            dropped = rng.random() < float(state.packet_loss_rate)
            if realized >= up_rate and not dropped:
                kept.append(token)
            else:
                lost += 1
        st["uplink_realized_efficiency"] = realized
        st["uplink_erasure_rate"] = lost / max(1, len(st["coarse_scores"]["token_ids"]))
        st["damaged"] = up_tokenizer.decode(kept, skip_special_tokens=True).strip()
        logger.info("  [U] %s erasure=%.3f damaged=%d chars",
                    st["sample_id"], st["uplink_erasure_rate"], len(st["damaged"]))

    # ======================================================================
    # Phase C (云): cloud 8B repairs/refines the damaged report, then scores it
    # ======================================================================
    logger.info("[phase C] cloud VLM (text-only): %s", config["model"]["cloud_path"])
    cloud, c_proc = load_vlm(config["model"]["cloud_path"])
    c_cfg = dict(config)
    c_cfg["transmitter"] = dict(config["cloud"])
    c_scorer = PragmaticScorer(cloud, c_proc, c_cfg)
    refine_system = ("You are the cloud semantic coordinator of a vehicular "
                     "semantic communication system. Vehicles send short, possibly "
                     "damaged reports over a lossy uplink; you restore and refine "
                     "them for a text-only downstream receiver.")
    for st in stages:
        row = processed[st["sample_id"]]
        refine_request = (
            f"Question: {row['question']}\n\nCandidate choices:\n"
            f"{build_choice_block(row['choices'])}\n\n"
            f"Damaged vehicle report (characters may be missing):\n\"\"\"{st['damaged']}\"\"\"\n\n"
            "Rewrite it as one clean, self-contained semantic description that a "
            "text-only model can use to answer the question. Requirements: restate "
            "the question; list every candidate choice; describe the visual and "
            "contextual evidence that discriminates the choices; repair missing or "
            "corrupted words silently using context -- never mention or quote the "
            "damage. The rewrite MUST be SHORTER than the damaged original (keep "
            "it under 100 tokens): refinement means condensing, not padding. Do "
            "not state or imply which choice is correct. Output the description "
            "only.")
        refined = generate(cloud, c_proc,
                           [{"role": "system", "content": refine_system},
                            {"role": "user", "content": refine_request}],
                           int(config["cloud"]["max_new_tokens"]))
        st["refined"] = refined
        st["refined_scores"] = score_given_text(c_scorer, row, None, refined)
        logger.info("  [C] %s refined=%d tokens", st["sample_id"],
                    st["refined_scores"]["num_tokens"])
    c_scorer = cloud = c_proc = None
    collect()
    logger.info("[phase C] freed, %.2f GiB still allocated", gpu_gib())

    # ======================================================================
    # Phase D (下行): reliable-budget downlink scheduling + erasure, two payloads
    # ======================================================================
    # payload "pipeline" = cloud-refined report (utilities scored at the cloud);
    # payload "direct" = the vehicle's own coarse report (utilities scored at
    # the vehicle). Both go through the identical downlink protocol.
    payloads = {"pipeline": "refined_scores", "direct": "coarse_scores"}
    cloud_tokenizer = AutoTokenizer.from_pretrained(config["model"]["cloud_path"],
                                                    local_files_only=True)
    channel_paths = config["channels"]
    snr_values = [float(v) for v in config["snr_db"]]
    seeds = [int(v) for v in config["experiment"]["channel_seeds"]]

    records = []
    for channel_path in channel_paths:
        base_spec = load_channel_spec(channel_path)
        for snr_db in snr_values:
            for seed in seeds:
                channel = build_channel({**base_spec, "snr_db": float(snr_db), "seed": seed})
                logger.info("downlink channel=%s snr=%s seed=%s", channel.name, snr_db, seed)
                for position, st in enumerate(stages):
                    sid = st["sample_id"]
                    row = processed[sid]
                    state = channel.sample((1,))
                    c_eps = reliable_spectral_efficiency(state, epsilon_out)
                    budget = slot_duration * bandwidth * float(c_eps[0])
                    realized = math.log2(1.0 + float(state.snr_linear)
                                         * float(state.h.abs().pow(2)[0]))
                    q = decoding_probability(state, codebook).reshape(-1)
                    rng = random.Random(seed * 1000003 + position)
                    for payload_name, score_field in payloads.items():
                        scores = st[score_field]
                        utilities = [float(u) for u in combine_scores(
                            scores["attention"], scores["semantic"], scores["gradient"],
                            weights)]
                        utility = torch.tensor(utilities, dtype=torch.float32)
                        n_tokens = scores["num_tokens"]
                        coupling = torch.zeros(n_tokens, n_tokens)
                        text_tokenizer = (up_tokenizer if payload_name == "direct"
                                          else cloud_tokenizer)
                        for method in methods:
                            if method == "full":
                                level = torch.ones(n_tokens, dtype=torch.long)
                            else:
                                level = select_level(
                                    method, utility, coupling, beta, q,
                                    torch.tensor(codebook.levels, dtype=torch.float32),
                                    budget, t_max if do_refine else 0, rng)
                            selected = [i for i in range(n_tokens) if level[i] > 0]
                            kept, lost = [], 0
                            for token in selected:
                                rate = float(codebook.code_rates[int(level[token]) - 1])
                                dropped = rng.random() < float(state.packet_loss_rate)
                                if realized >= rate and not dropped:
                                    kept.append(token)
                                else:
                                    lost += 1
                            received = channel.transmit(text_tokenizer.decode(
                                [scores["token_ids"][i] for i in kept],
                                skip_special_tokens=True))
                            symbols = sum(float(codebook.levels[int(level[i]) - 1])
                                          for i in selected)
                            records.append({
                                "sample_id": sid, "payload": payload_name,
                                "channel": channel.name, "snr_db": float(snr_db),
                                "channel_seed": seed, "method": method,
                                "original_tokens": n_tokens,
                                "selected_tokens": len(selected),
                                "received_tokens": len(kept),
                                "TRR": token_retention_ratio(len(selected), n_tokens),
                                "transmitted_symbols": symbols, "budget": budget,
                                "reliable_efficiency": float(c_eps[0]),
                                "realized_efficiency": realized,
                                "over_budget": int(symbols > budget),
                                "token_erasure_rate": lost / len(selected) if selected else 0.0,
                                "received_text": received})
                    logger.info("  [D] %s done", sid)

    # ======================================================================
    # Phase A (端): terminal 1.7B answers from the surviving downlink text
    # ======================================================================
    logger.info("[phase A] terminal LLM: %s", config["model"]["terminal_path"])
    term_tokenizer = AutoTokenizer.from_pretrained(config["model"]["terminal_path"],
                                                   local_files_only=True)
    terminal = AutoModelForCausalLM.from_pretrained(
        config["model"]["terminal_path"], local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="eager").to("cuda:0").eval()
    for rec in records:
        row = processed[rec["sample_id"]]
        prompt = term_tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{ANSWER_REQUEST}\n\n{rec['received_text']}"}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = term_tokenizer(prompt, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            output = terminal.generate(**inputs,
                                       max_new_tokens=int(config["receiver"]["max_new_tokens"]),
                                       do_sample=False)
        raw = term_tokenizer.decode(output[0, inputs.input_ids.shape[1]:],
                                    skip_special_tokens=True).strip()
        prediction, parse_mode = canonical_prediction(raw, row["choices"])
        rec["prediction"] = prediction
        rec["parse_mode"] = parse_mode
        rec["correct"] = exact_match(prediction, str(row["choices"][row["answer"]]))
    terminal = None
    collect()

    # ======================================================================
    # summary
    # ======================================================================
    def agg(sub):
        if not sub:
            return {}
        n = len(sub)
        return {"n": n,
                "accuracy": sum(r["correct"] for r in sub) / n,
                "mean_symbols": sum(r["transmitted_symbols"] for r in sub) / n,
                "mean_selected": sum(r["selected_tokens"] for r in sub) / n,
                "mean_trr": sum(r["TRR"] for r in sub) / n,
                "over_budget_rate": sum(r["over_budget"] for r in sub) / n,
                "mean_erasure": sum(r["token_erasure_rate"] for r in sub) / n,
                "raw_parse_rate": sum(r["parse_mode"] == "raw" for r in sub) / n}

    conditions = {}
    for rec in records:
        key = (rec["payload"], rec["channel"], float(rec["snr_db"]), rec["method"])
        conditions.setdefault(key, []).append(rec)
    summary = {
        "samples": len(manifest),
        "methods": methods,
        "uplink": {"channel": config["uplink"]["channel"],
                   "snr_db": float(config["uplink"]["snr_db"]),
                   "mean_erasure": sum(st["uplink_erasure_rate"] for st in stages) / len(stages)},
        "stage_tokens": {"coarse_mean": sum(st["coarse_scores"]["num_tokens"] for st in stages) / len(stages),
                         "refined_mean": sum(st["refined_scores"]["num_tokens"] for st in stages) / len(stages)},
        "conditions": [{"payload": k[0], "channel": k[1], "snr_db": k[2], "method": k[3],
                        **agg(v)} for k, v in sorted(conditions.items())],
        "records": records,
        "stages": [{k: st[k] for k in ("sample_id", "coarse", "damaged", "refined",
                                       "uplink_erasure_rate")} for st in stages],
    }
    atomic_json(out_dir / "summary.json", summary)
    logger.info("DONE: %d records -> %s", len(records), out_dir / "summary.json")


if __name__ == "__main__":
    main()
