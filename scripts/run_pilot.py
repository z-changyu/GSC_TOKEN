#!/usr/bin/env python
"""Single-GPU, ideal-channel ScienceQA pilot with paper-faithful pragmatic scoring.

Per-sample pipeline
-------------------
1. Transmit  Qwen3-VL converts (image, question, choices) into a semantic prompt P.
2. Score     Eq. (2): L_T(P) = -log p(y* | I, Q, P) is built by teacher forcing the
             ground-truth answer y* after [user: I, Q][assistant: P][user: request].
             The input embeddings of the P slice are the differentiation target, so
             G_i = ||dL_T/de_i|| * ||e_i - e_mask|| (Eq. 14) is the deletion-aware
             sensitivity of the *answer* to each prompt token. A_i (Eq. 10),
             S_i (Eq. 11) and M_ij (Eq. 16) are read from the same forward pass,
             restricted to the P token slice.
3. Select    full / seeded random / utility-greedy at several retention ratios.
4. Receive   Qwen3-1.7B answers from the retained text only.

Only an ideal channel is implemented at this stage; everything downstream of the
token decision is bit-exact.
"""
from __future__ import annotations

import argparse, gc, json, os, re, sys, time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SYSTEM_PROMPT = (
    "You are the transmitter of a semantic communication system. You convert an image and a "
    "question into a compact text description for a text-only receiver model that cannot see "
    "the image."
)

TRANSMITTER_REQUEST = (
    "Question: {question}\n\nCandidate choices:\n{choices}\n\n"
    "Write a short semantic description that a text-only model can use to answer the question. "
    "Requirements: restate the question; list every candidate choice; describe only the visual "
    "and contextual evidence that discriminates the choices; do not state or imply which choice "
    "is correct. Output the description only, with no preamble, heading, or meta-commentary."
)

ANSWER_REQUEST = (
    "Answer the question using the semantic description below. "
    "Reply with the exact option text only, with no explanation."
)

# Patterns that indicate the transmitter echoed the instruction instead of producing a description.
META_LEAK_PATTERNS = (
    "generate a", "create a", "write a semantic", "write a concise",
    "the prompt should", "semantic prompt", "your task is", "you are asked to",
    "instructions:", "requirements:", "task:", "here is a",
)

# An option letter only counts when it is uppercase and closed by a separator or
# the end of the string, so that ordinary sentences are not read as an option.
LEADING_OPTION = re.compile(r"^\s*\(?\s*([A-Z])\s*(?:[\.\):]\s*|$)")


def build_choice_block(choices):
    return "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices or []))


def detect_meta_leak(text: str) -> bool:
    head = text.strip().lower()[:120]
    return any(pattern in head for pattern in META_LEAK_PATTERNS)


def canonical_prediction(text, choices):
    """Return (prediction, parse_mode). Never searches an explanation for a choice."""
    from src.metrics.qa import normalize_answer

    clean = normalize_answer(text)
    if not choices:
        return text.strip(), "raw"
    norm_choices = [normalize_answer(choice) for choice in choices]

    exact = [choice for choice, norm in zip(choices, norm_choices) if norm == clean]
    if len(exact) == 1:
        return exact[0], "exact"

    match = LEADING_OPTION.match(text)
    if match:
        index = ord(match.group(1)) - 65
        if 0 <= index < len(choices):
            return choices[index], "leading_letter"

    if clean:
        hits = {choice for choice, norm in zip(choices, norm_choices)
                if norm and re.search(rf"\b{re.escape(norm)}\b", clean)}
        if len(hits) == 1:
            return hits.pop(), "unique_substring"
    return text.strip(), "raw"


def prompt_span_hint(processor, messages, semantic_prompt, tokenizer):
    """Return (hint, expected_ids): where the P span should start, and P's own ids.

    The hint is the length of the rendering that stops right after the assistant
    header; the exact start is then confirmed by matching P's token ids, because
    the template's trailing whitespace tokens differ between renderings.
    """
    prefix = processor.apply_chat_template(messages[:1], tokenize=True,
                                           add_generation_prompt=True, return_dict=True,
                                           return_tensors="pt")
    return int(prefix["input_ids"][0].shape[0]), tokenizer(semantic_prompt,
                                                           add_special_tokens=False)["input_ids"]


def locate_span(sequence, needle, hint):
    """First index of ``needle`` in ``sequence``, searched around ``hint`` first."""
    n, m = len(sequence), len(needle)
    if m == 0:
        return -1
    low = max(0, hint - 8)
    high = min(n - m, hint + 24)
    for start, end in ((low, high + 1), (0, n - m + 1)):
        for i in range(start, end):
            if sequence[i:i + m] == needle:
                return i
    return -1


def protection_rates(token_count, protection):
    mode = protection.get("mode", "uniform")
    if mode == "uniform":
        return [float(protection["uniform_symbols_per_token"])] * token_count
    raise ValueError(f"unsupported protection mode {mode!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/pilot_scienceqa.yaml")
    parser.add_argument("--num-samples", type=int, help="Override only for smoke testing")
    parser.add_argument("--manifest", help="Override the frozen sample manifest")
    parser.add_argument("--processed", help="Override the processed sample jsonl")
    parser.add_argument("--output-dir", default="outputs/pilot")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    visible = str(config["experiment"].get("cuda_visible_devices", "0"))
    if "," in visible:
        raise SystemExit("This pilot is restricted to exactly one visible GPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import pandas as pd
    import torch
    from PIL import Image
    from transformers import (AutoImageProcessor, AutoModelForCausalLM, AutoProcessor,
                              AutoTokenizer, Qwen3VLForConditionalGeneration)
    from src.channel.ideal import IdealChannel
    from src.metrics.communication import token_retention_ratio, transmitted_symbol_cost
    from src.metrics.qa import exact_match
    from src.metrics.statistics import mean_std_ci95
    from src.pragmatic.attention_score import AttentionStructuralScorer
    from src.pragmatic.coupling import SemanticCouplingScorer
    from src.pragmatic.gradient_score import GradientScorer
    from src.pragmatic.semantic_score import SemanticDirectionalScorer
    from src.pragmatic.utility import UnifiedUtilityScorer, context_aware_utility, coupling_regime
    from src.selection.random_selector import RandomTokenSelector
    from src.selection.utility_selector import FullPromptSelector, UtilityTokenSelector
    from src.utils.io import atomic_json, read_jsonl
    from src.utils.logging import configure_logging, create_run_directory
    from src.utils.power import PowerSampler
    from src.utils.seed import set_global_seed

    capability = json.loads(Path("outputs/model_capabilities.json").read_text())
    failures = {k: v for k, v in capability.items()
                if isinstance(v, dict) and v.get("status") not in (None, "PASS")}
    if failures:
        raise SystemExit(f"Pilot blocked by capability gates: {sorted(failures)}")

    manifest = read_jsonl(args.manifest or config["dataset"]["manifest"])
    samples = read_jsonl(args.processed or config["dataset"]["processed"])
    by_id = {str(row["row_idx"]): row for row in samples}
    ordered = [by_id[row["sample_id"]] for row in manifest]
    if config["dataset"].get("require_image", False):
        ordered = [row for row in ordered if row.get("image")]
    requested = args.num_samples or config["experiment"]["num_samples"]
    if len(ordered) > requested:
        ordered = ordered[:requested]
    limit = len(ordered)
    set_global_seed(config["experiment"]["model_seed"])
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    run = create_run_directory("outputs/runs", f"scienceqa_qwen3_single_gpu_pilot_{limit}", config)
    logger = configure_logging(run / "run.log")
    if limit != requested:
        logger.warning("requested %d usable samples, only %d available", requested, limit)
    power = PowerSampler()
    logger.info("Power sampling backend: %s", power.backend or "unavailable")
    if not power.available:
        logger.warning("No power backend available; energy columns will be null")

    logger.info("Loading single-GPU BF16/eager models; samples=%d", limit)
    tx_path = config["model"]["transmitter_path"]
    rx_path = config["model"]["receiver_path"]
    processor = AutoProcessor.from_pretrained(tx_path, local_files_only=True)
    # Pin the visual-token budget so that the attention tensors stay bounded.
    processor.image_processor = AutoImageProcessor.from_pretrained(
        tx_path, min_pixels=config["model"]["image_min_pixels"],
        max_pixels=config["model"]["image_max_pixels"], local_files_only=True)
    tokenizer = processor.tokenizer
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    transmitter = Qwen3VLForConditionalGeneration.from_pretrained(
        tx_path, dtype=torch.bfloat16, attn_implementation="eager",
        local_files_only=True).to("cuda:0").eval()
    receiver_tokenizer = AutoTokenizer.from_pretrained(rx_path, local_files_only=True)
    receiver = AutoModelForCausalLM.from_pretrained(
        rx_path, dtype=torch.bfloat16, attn_implementation="eager",
        local_files_only=True).to("cuda:0").eval()

    # Only the prompt-token embeddings are differentiated, so the vision tower is
    # frozen: it saves activation memory and avoids building a graph we never use.
    _vision_features = transmitter.model.get_image_features

    @torch.no_grad()
    def _frozen_vision_features(pixel_values, image_grid_thw):
        return _vision_features(pixel_values, image_grid_thw)

    transmitter.model.get_image_features = _frozen_vision_features

    mask_token_id = tokenizer.convert_tokens_to_ids(config["pragmatic"]["mask_token"])
    if mask_token_id is None or mask_token_id < 0:
        raise SystemExit(f"unknown mask token {config['pragmatic']['mask_token']!r}")
    with torch.no_grad():
        mask_embedding = transmitter.get_input_embeddings()(
            torch.tensor([mask_token_id], device="cuda:0"))[0]

    attention_scorer = AttentionStructuralScorer(config["pragmatic"].get("layer_reduction", "mean"))
    semantic_scorer = SemanticDirectionalScorer()
    coupling_scorer = SemanticCouplingScorer(config["pragmatic"].get("layer_reduction", "mean"))
    gradient_scorer = GradientScorer("eq2_answer_nll_given_I_Q_P", "ground_truth_answer")
    utility_scorer = UnifiedUtilityScorer(config["pragmatic"]["lambda_attention"],
                                          config["pragmatic"]["lambda_semantic"],
                                          config["pragmatic"]["lambda_gradient"])
    beta = config["pragmatic"]["beta"]
    coupling_enabled = bool(config["pragmatic"].get("coupling_enabled", False))
    channel = IdealChannel()
    max_prompt_tokens = config["transmitter"]["max_prompt_tokens"]
    records = []

    for sample_number, sample in enumerate(ordered, 1):
        sample_start = time.perf_counter()
        image = Image.open(sample["image"]).convert("RGB") if sample.get("image") else None
        question_text = TRANSMITTER_REQUEST.format(question=sample["question"],
                                                   choices=build_choice_block(sample["choices"]))

        # ---------------- 1. transmit: generate the semantic prompt ----------------
        t0 = time.perf_counter()
        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": question_text})
        tx_inputs = processor.apply_chat_template(
            [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
             {"role": "user", "content": content}],
            tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        tx_inputs = {k: (v.to("cuda:0") if hasattr(v, "to") else v) for k, v in tx_inputs.items()}
        with torch.no_grad():
            generated = transmitter.generate(**tx_inputs,
                                             max_new_tokens=config["transmitter"]["max_new_tokens"],
                                             do_sample=False)
        semantic_prompt = tokenizer.decode(
            generated[0, tx_inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        transmit_seconds = time.perf_counter() - t0
        del generated, tx_inputs
        gc.collect(); torch.cuda.empty_cache()

        if not semantic_prompt:
            raise RuntimeError(f"empty transmitter prompt for sample {sample['row_idx']}")

        token_ids = tokenizer(semantic_prompt, add_special_tokens=False)["input_ids"]
        truncated = len(token_ids) > max_prompt_tokens
        if truncated:
            token_ids = token_ids[:max_prompt_tokens]
            semantic_prompt = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
            token_ids = tokenizer(semantic_prompt, add_special_tokens=False)["input_ids"]
        token_strings = tokenizer.convert_ids_to_tokens(token_ids)
        if not token_ids:
            raise RuntimeError(f"empty prompt after truncation for sample {sample['row_idx']}")

        # ---------------- 2. score: Eq. (2) answer NLL given (I, Q, P) ----------------
        t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        scoring_messages = [
            {"role": "user", "content": ([] if image is None
                                         else [{"type": "image", "image": image}])
             + [{"type": "text", "text": f"Question: {sample['question']}\n\n"
                                         f"Candidate choices:\n{build_choice_block(sample['choices'])}"}]},
            {"role": "assistant", "content": [{"type": "text", "text": semantic_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": ANSWER_REQUEST}]},
        ]
        hint, expected_ids = prompt_span_hint(processor, scoring_messages,
                                             semantic_prompt, tokenizer)
        batch = processor.apply_chat_template(scoring_messages, tokenize=True,
                                              add_generation_prompt=True, return_dict=True,
                                              return_tensors="pt")
        sequence_ids = batch["input_ids"].to("cuda:0")
        attention_mask = batch["attention_mask"].to("cuda:0")
        sequence_list = sequence_ids[0].tolist()
        p_start = locate_span(sequence_list, expected_ids, hint)
        if p_start < 0:
            raise RuntimeError(f"cannot locate the prompt span for sample {sample['row_idx']}")
        p_end = p_start + len(expected_ids)
        if sequence_list[p_start:p_end] != token_ids:
            raise RuntimeError(f"prompt-token mismatch for sample {sample['row_idx']}")

        answer_text = str(sample["choices"][sample["answer"]])
        answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"] + [im_end_id]
        tail = torch.tensor([answer_ids], device="cuda:0")
        input_ids = torch.cat([sequence_ids, tail], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(tail)], dim=1)
        labels = torch.full_like(input_ids, -100)
        labels[0, -len(answer_ids):] = tail[0]

        pixel_values = batch["pixel_values"].to("cuda:0") if "pixel_values" in batch else None
        image_grid_thw = batch["image_grid_thw"].to("cuda:0") if "image_grid_thw" in batch else None
        position_ids, _ = transmitter.model.get_rope_index(
            input_ids, image_grid_thw, None, attention_mask=attention_mask)
        embeddings = transmitter.get_input_embeddings()(input_ids).detach().to(torch.bfloat16)
        embeddings.requires_grad_(True)
        scoring = transmitter(inputs_embeds=embeddings, attention_mask=attention_mask,
                              position_ids=position_ids, pixel_values=pixel_values,
                              image_grid_thw=image_grid_thw, labels=labels,
                              output_attentions=True, return_dict=True)
        prompt_slice = slice(p_start, p_end)
        prompt_attentions = tuple(layer[:, :, prompt_slice, prompt_slice]
                                  for layer in scoring.attentions)
        attention = attention_scorer.score(prompt_attentions, token_ids, token_strings)
        semantic = semantic_scorer.score(embeddings.detach()[0, prompt_slice].float(),
                                         attention.scores.float(), token_ids, token_strings)
        coupling = coupling_scorer.score(prompt_attentions) if coupling_enabled else None
        # G_i uses the mean answer NLL; the 1/T_y factor is a per-sample constant and
        # G_i is min-max normalised within the sample, so it does not affect ranking.
        gradient = gradient_scorer.score(scoring.loss, embeddings, token_ids, token_strings,
                                         mask_embedding=mask_embedding, positions=prompt_slice)
        base_utility = utility_scorer.score(attention, semantic, gradient).scores.detach()
        if coupling is None:
            utilities = base_utility.float().cpu().tolist()
            regime = {}
        else:
            regime = coupling_regime(coupling, beta)
            utilities = context_aware_utility(base_utility, coupling, beta).detach().float().cpu().tolist()
        scoring_loss = float(scoring.loss.detach())
        score_seconds = time.perf_counter() - t0
        scoring_peak_memory_gib = torch.cuda.max_memory_allocated() / 2**30
        sequence_length = int(input_ids.shape[1])
        del scoring, embeddings, batch, input_ids, labels, position_ids
        gc.collect(); torch.cuda.empty_cache()

        # ---------------- 3/4. select and receive ----------------
        conditions = [("full", 1.0)] + [(method, float(ratio))
                                        for method in ("random", "proposed")
                                        for ratio in config["selection"]["retention_ratios"]]
        for method, ratio in conditions:
            t0 = time.perf_counter()
            if method == "full":
                selection = FullPromptSelector().select(token_ids)
            elif method == "random":
                per_sample_seed = config["experiment"]["model_seed"] + int(sample["row_idx"])
                selection = RandomTokenSelector(per_sample_seed).select(token_ids,
                                                                        retention_ratio=ratio)
            else:
                selection = UtilityTokenSelector().select(token_ids, retention_ratio=ratio,
                                                          utilities=utilities)
            select_seconds = time.perf_counter() - t0

            selected_ids = [token_ids[i] for i in selection.indices]
            received = channel.transmit(tokenizer.decode(selected_ids, skip_special_tokens=True))
            rx_prompt = receiver_tokenizer.apply_chat_template(
                [{"role": "user", "content": f"{ANSWER_REQUEST}\n\n{received}"}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            rx_inputs = receiver_tokenizer(rx_prompt, return_tensors="pt").to("cuda:0")

            # The absolute peak is dominated by the resident weights, so the delta
            # over the pre-condition baseline is what isolates this condition.
            torch.cuda.reset_peak_memory_stats()
            memory_baseline_gib = torch.cuda.memory_allocated() / 2**30
            watts_start = power.watts()
            t1 = time.perf_counter()
            with torch.no_grad():
                rx_output = receiver.generate(**rx_inputs,
                                              max_new_tokens=config["receiver"]["max_new_tokens"],
                                              do_sample=False)
            receive_seconds = time.perf_counter() - t1
            watts_end = power.watts()
            condition_peak_memory_gib = torch.cuda.max_memory_allocated() / 2**30

            raw_prediction = receiver_tokenizer.decode(
                rx_output[0, rx_inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            prediction, parse_mode = canonical_prediction(raw_prediction, sample.get("choices"))
            ground_truth = str(sample["choices"][sample["answer"]])
            selected_count, original_count = len(selected_ids), len(token_ids)
            rates = protection_rates(original_count, config["selection"]["protection"])
            retention_flags = [i in set(selection.indices) for i in range(original_count)]
            records.append({
                "sample_id": str(sample["row_idx"]), "method": method, "retention_ratio": ratio,
                "original_tokens": original_count, "selected_tokens": selected_count,
                "TRR": token_retention_ratio(selected_count, original_count),
                "transmitted_symbols": transmitted_symbol_cost(retention_flags, rates),
                "prediction": prediction, "raw_prediction": raw_prediction,
                "parse_mode": parse_mode, "ground_truth": ground_truth,
                "correct": exact_match(prediction, ground_truth),
                "transmit_seconds": transmit_seconds, "score_seconds": score_seconds,
                "select_seconds": select_seconds, "receive_seconds": receive_seconds,
                "runtime": transmit_seconds + score_seconds + select_seconds + receive_seconds,
                "gpu_peak_memory_gib": condition_peak_memory_gib,
                "gpu_peak_memory_delta_gib": condition_peak_memory_gib - memory_baseline_gib,
                "scoring_peak_memory_gib": scoring_peak_memory_gib,
                "energy_j": PowerSampler.joules(watts_start, watts_end, receive_seconds),
                "semantic_prompt": semantic_prompt, "selected_text": received,
                "prompt_meta_leak": detect_meta_leak(semantic_prompt),
                "prompt_truncated": truncated, "sequence_length": sequence_length,
                "answer_nll": scoring_loss,
                "gradient_loss_type": "eq2_answer_nll_given_I_Q_P",
                "attention_layer_reduction": attention_scorer.layer_reduction,
                "coupling_enabled": coupling_enabled,
                "coupling_kappa": regime.get("kappa"),
                "coupling_d_M": regime.get("d_M"),
                "channel_seed": config["experiment"]["channel_seed"],
            })
            del rx_output, rx_inputs
            gc.collect(); torch.cuda.empty_cache()

        logger.info("sample %d/%d id=%s tokens=%d seq=%d meta_leak=%s runtime=%.2fs",
                    sample_number, limit, sample["row_idx"], len(token_ids), sequence_length,
                    records[-1]["prompt_meta_leak"], time.perf_counter() - sample_start)
        pd.DataFrame(records).to_csv(output_dir / "results.partial.csv", index=False)

    frame = pd.DataFrame(records)
    frame.to_csv(output_dir / "results.csv", index=False)

    summary_rows = []
    for (method, ratio), group in frame.groupby(["method", "retention_ratio"], sort=True):
        stats = mean_std_ci95(group["correct"].astype(float).tolist())
        summary_rows.append({
            "method": method, "retention_ratio": float(ratio), "samples": len(group),
            "accuracy": stats["mean"], "accuracy_std": stats["std"], "accuracy_ci95": stats["ci95"],
            "mean_TRR": float(group["TRR"].mean()),
            "mean_symbols": float(group["transmitted_symbols"].mean()),
            "mean_transmit_seconds": float(group["transmit_seconds"].mean()),
            "mean_score_seconds": float(group["score_seconds"].mean()),
            "mean_select_seconds": float(group["select_seconds"].mean()),
            "mean_receive_seconds": float(group["receive_seconds"].mean()),
            "mean_gpu_peak_memory_gib": float(group["gpu_peak_memory_gib"].mean()),
            "mean_gpu_peak_memory_delta_gib": float(group["gpu_peak_memory_delta_gib"].mean()),
            "mean_energy_j": (float(group["energy_j"].mean())
                              if group["energy_j"].notna().any() else None),
            "parse_mode_distribution": {str(k): int(v)
                                        for k, v in group["parse_mode"].value_counts().items()},
        })
    summary = {
        "num_samples": limit,
        "dataset_seed": config["experiment"]["dataset_seed"],
        "model_seed": config["experiment"]["model_seed"],
        "channel_seed": config["experiment"]["channel_seed"],
        "single_physical_gpu": visible,
        "channel": "ideal",
        "transmitter": tx_path,
        "receiver": rx_path,
        "proposed_scope": ("base_A_S_G_utility_with_coupling" if coupling_enabled
                           else "base_A_S_G_utility_without_coupling_or_swap"),
        "gradient_loss": "Eq. (2) answer NLL given (I, Q, P)",
        "meta_leak_rate": float(frame.drop_duplicates("sample_id")["prompt_meta_leak"].mean()),
        "prompt_truncation_rate": float(frame.drop_duplicates("sample_id")["prompt_truncated"].mean()),
        "energy_measured": power.available,
        "energy_backend": power.backend,
        "conditions": summary_rows,
    }
    atomic_json(output_dir / "summary.json", summary)
    atomic_json(run / "summary.json", summary)
    frame.to_csv(run / "results.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for method, group in pd.DataFrame(summary_rows).groupby("method"):
        plt.plot(group["retention_ratio"], group["accuracy"], marker="o", label=method)
    plt.xlabel("Token retention ratio"); plt.ylabel("QA accuracy"); plt.ylim(0, 1)
    plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(output_dir / "accuracy_vs_trr.png", dpi=180); plt.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
