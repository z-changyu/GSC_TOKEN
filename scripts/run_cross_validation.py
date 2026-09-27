#!/usr/bin/env python
"""Five-fold cross-validation over the frozen ScienceQA folds.

Each fold runs three phases.

1. **Score** — ``A_i``, ``S_i``, ``G_i`` (and ``M_ij`` when coupling is on) are
   computed once per sample and cached. They do not depend on the pragmatic
   weights, so the cache is shared by every fold.
2. **Calibrate** — the weights ``(lambda_1, lambda_2, lambda_3)`` of Eq. (15) are
   grid searched *inside* this fold's training samples, scoring each candidate by
   the transmitter-side answer NLL of the prompt pruned at the calibration
   retention ratio (Eq. 21-23). The receiver is not involved, so a candidate
   costs one forward pass rather than one generation.
3. **Evaluate** — full / seeded random / proposed are run on the held-out fold
   through the receiver.

Fold-level means are reported with the standard deviation and the
normal-approximation 95% confidence interval required by Section VI-B.
"""
from __future__ import annotations

import argparse, gc, json, os, sys, time
from itertools import product
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def simplex_grid(step: float) -> list[tuple[float, float, float]]:
    """Nonnegative weight triples summing to one on a grid of the given step."""
    divisions = int(round(1.0 / step))
    points = []
    for i in range(divisions + 1):
        for j in range(divisions + 1 - i):
            k = divisions - i - j
            points.append((i * step, j * step, k * step))
    return sorted(set(points))


def answer_with_receiver(receiver, receiver_tokenizer, sample, prompt_text, max_new_tokens):
    """Run the receiver once and return 1/0 correctness for the sample's ground truth."""
    import torch
    from src.experiments.prompts import ANSWER_REQUEST, canonical_prediction
    from src.metrics.qa import exact_match

    message = receiver_tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{ANSWER_REQUEST}\n\n{prompt_text}"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = receiver_tokenizer(message, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        output = receiver.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    raw = receiver_tokenizer.decode(output[0, inputs.input_ids.shape[1]:],
                                    skip_special_tokens=True).strip()
    prediction, _ = canonical_prediction(raw, sample["choices"])
    correct = float(exact_match(prediction, str(sample["choices"][sample["answer"]])))
    del output, inputs
    return correct


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/pilot_scienceqa.yaml")
    parser.add_argument("--folds", default="data/manifests/scienceqa_val_local_folds.json")
    parser.add_argument("--processed", default="data/processed/scienceqa_validation_local.jsonl")
    parser.add_argument("--score-cache", default="outputs/cache/scienceqa_scores.jsonl")
    parser.add_argument("--score-only", action="store_true",
                        help="Build or extend the score cache, then stop")
    parser.add_argument("--coupling", dest="coupling", action="store_true", default=None,
                        help="Force Eq. (16) coupling on, storing M_ij in the cache")
    parser.add_argument("--no-coupling", dest="coupling", action="store_false")
    parser.add_argument("--calibration-size", type=int, default=80)
    parser.add_argument("--calibration-retention", type=float, default=0.5)
    parser.add_argument("--calibration-objective", choices=("accuracy", "nll"),
                        default="accuracy",
                        help="accuracy maximises receiver VQA accuracy; nll minimises the "
                             "transmitter answer NLL of Eq. (21)-(23)")
    parser.add_argument("--lambda-step", type=float, default=0.25)
    parser.add_argument("--include-legacy-weights", action="store_true", default=True)
    parser.add_argument("--output-dir", default="outputs/cv")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    visible = str(config["experiment"].get("cuda_visible_devices", "0"))
    if "," in visible:
        raise SystemExit("This driver is restricted to exactly one visible GPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import pandas as pd
    import torch
    from transformers import (AutoImageProcessor, AutoModelForCausalLM, AutoProcessor,
                              AutoTokenizer, Qwen3VLForConditionalGeneration)
    from src.channel.ideal import IdealChannel
    from src.experiments.prompts import (TRANSMITTER_REQUEST, build_choice_block,
                                         detect_meta_leak)
    from src.experiments.scoring import PragmaticScorer, SampleScores, combine_scores, load_image
    from src.metrics.communication import token_retention_ratio
    from src.metrics.statistics import mean_std_ci95
    from src.selection.random_selector import RandomTokenSelector
    from src.selection.utility_selector import FullPromptSelector, UtilityTokenSelector
    from src.utils.io import atomic_json, read_jsonl, write_jsonl
    from src.utils.logging import configure_logging, create_run_directory
    from src.utils.seed import set_global_seed

    folds = json.loads(Path(args.folds).read_text(encoding="utf-8"))
    processed = {str(row["row_idx"]): row for row in read_jsonl(args.processed)}
    for row in processed.values():
        row["_request"] = TRANSMITTER_REQUEST.format(question=row["question"],
                                                     choices=build_choice_block(row["choices"]))
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    run = create_run_directory("outputs/runs", "scienceqa_qwen3_5fold_cv", config)
    logger = configure_logging(run / "run.log")

    def manifest_ids(path):
        return [str(entry["sample_id"]) for entry in read_jsonl(path)]

    image_cache: dict[str, object] = {}

    def get_image(sample_id: str):
        """Decode each image once; calibration revisits every sample per grid point."""
        if sample_id not in image_cache:
            image_cache[sample_id] = load_image(processed[sample_id].get("image"))
        return image_cache[sample_id]

    needed = sorted({sid for fold in folds["folds"]
                     for sid in manifest_ids(fold["train_manifest"]) + manifest_ids(fold["test_manifest"])})
    logger.info("cross-validation over %d folds, %d distinct samples", len(folds["folds"]), len(needed))

    # ---------------------------------------------------------------- models
    tx_path, rx_path = config["model"]["transmitter_path"], config["model"]["receiver_path"]
    processor = AutoProcessor.from_pretrained(tx_path, local_files_only=True)
    processor.image_processor = AutoImageProcessor.from_pretrained(
        tx_path, min_pixels=config["model"]["image_min_pixels"],
        max_pixels=config["model"]["image_max_pixels"], local_files_only=True)
    tokenizer = processor.tokenizer
    receiver_tokenizer = AutoTokenizer.from_pretrained(rx_path, local_files_only=True)
    receiver = AutoModelForCausalLM.from_pretrained(
        rx_path, dtype=torch.bfloat16, attn_implementation="eager",
        local_files_only=True).to("cuda:0").eval()

    # ------------------------------------------------------------ 1. scoring
    cache_path = Path(args.score_cache)
    cache: dict[str, SampleScores] = {}
    if cache_path.exists():
        for row in read_jsonl(cache_path):
            cache[row["sample_id"]] = SampleScores.from_dict(row)
        logger.info("loaded %d cached score rows", len(cache))
    scorer = None
    if args.coupling is not None:
        config.setdefault("pragmatic", {})["coupling_enabled"] = bool(args.coupling)

    pending = [sid for sid in needed if sid not in cache]
    if pending:
        # The transmitter is only resident while the cache is being built; the
        # calibration and evaluation phases need the receiver alone.
        transmitter = Qwen3VLForConditionalGeneration.from_pretrained(
            tx_path, dtype=torch.bfloat16, attn_implementation="eager",
            local_files_only=True).to("cuda:0").eval()
        scorer = PragmaticScorer(transmitter, processor, config)
        logger.info("scoring %d new samples (%s coupling)", len(pending),
                    "with" if config["pragmatic"].get("coupling_enabled") else "without")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        for position, sample_id in enumerate(pending, 1):
            row = processed[sample_id]
            scores = scorer.score(row, get_image(sample_id))
            scores.prompt_meta_leak = detect_meta_leak(scores.semantic_prompt)
            cache[sample_id] = scores
            logger.info("scored %d/%d id=%s tokens=%d nll=%.3f", position, len(pending),
                        sample_id, scores.num_tokens, scores.answer_nll)
            if position % 10 == 0 or position == len(pending):
                write_jsonl(cache_path, [cache[sid].to_dict() for sid in sorted(cache)])
        write_jsonl(cache_path, [cache[sid].to_dict() for sid in sorted(cache)])

    if args.score_only:
        logger.info("score-only mode: cache has %d/%d samples", len(cache), len(needed))
        return

    # --------------------------------------------------------- 2. calibration
    weights_grid = simplex_grid(args.lambda_step)
    if args.include_legacy_weights:
        legacy = tuple(float(config["pragmatic"][key]) for key in
                       ("lambda_attention", "lambda_semantic", "lambda_gradient"))
        if abs(sum(legacy) - 1.0) < 1e-9 and legacy not in weights_grid:
            weights_grid.append(legacy)
    logger.info("calibration grid: %d weight triples", len(weights_grid))

    model_seed = int(config["experiment"]["model_seed"])
    retention_ratios = [float(r) for r in config["selection"]["retention_ratios"]]
    channel = IdealChannel()
    fold_rows, calibration_rows = [], []

    for fold in folds["folds"]:
        fold_index = int(fold["fold"])
        train_ids = manifest_ids(fold["train_manifest"])
        test_ids = manifest_ids(fold["test_manifest"])
        rng = __import__("random").Random(model_seed + fold_index)
        calibration_ids = sorted(rng.sample(train_ids, min(args.calibration_size, len(train_ids))))

        logger.info("fold %d: calibrating on %d training samples by %s",
                    fold_index, len(calibration_ids), args.calibration_objective)
        best_weights, best_score = None, None
        for weights in weights_grid:
            losses, hits = [], []
            for sample_id in calibration_ids:
                scores = cache[sample_id]
                utilities = combine_scores(scores.attention, scores.semantic,
                                           scores.gradient, weights)
                selection = UtilityTokenSelector().select(
                    scores.token_ids, retention_ratio=args.calibration_retention,
                    utilities=[float(u) for u in utilities])
                kept = [scores.token_ids[i] for i in selection.indices]
                pruned = tokenizer.decode(kept, skip_special_tokens=True).strip()
                if not pruned:
                    losses.append(float("inf")); hits.append(0.0)
                    continue
                # The NLL column is diagnostic: it needs the transmitter, which is
                # only resident while the cache is being built.
                losses.append(scorer.pruned_prompt_nll(processed[sample_id],
                                                       get_image(sample_id), pruned)
                              if scorer is not None else float("nan"))
                hits.append(float(answer_with_receiver(
                    receiver, receiver_tokenizer, processed[sample_id], pruned,
                    config["receiver"]["max_new_tokens"])))
            mean_loss = sum(losses) / len(losses)
            mean_accuracy = sum(hits) / len(hits)
            calibration_rows.append({"fold": fold_index, "weights": list(weights),
                                     "mean_answer_nll": mean_loss,
                                     "calibration_accuracy": mean_accuracy})
            objective = (mean_accuracy if args.calibration_objective == "accuracy"
                         else -mean_loss)
            if best_score is None or objective > best_score:
                best_weights, best_score = weights, objective
        logger.info("fold %d: selected weights %s (objective %.4f)",
                    fold_index, best_weights, best_score)

        # ------------------------------------------------------ 3. evaluation
        logger.info("fold %d: evaluating %d held-out samples", fold_index, len(test_ids))
        for sample_id in test_ids:
            scores = cache[sample_id]
            row = processed[sample_id]
            ground_truth = str(row["choices"][row["answer"]])
            utilities = [float(u) for u in combine_scores(scores.attention, scores.semantic,
                                                          scores.gradient, best_weights)]
            conditions = [("full", 1.0)] + [(method, ratio) for method in ("random", "proposed")
                                            for ratio in retention_ratios]
            for method, ratio in conditions:
                if method == "full":
                    selection = FullPromptSelector().select(scores.token_ids)
                elif method == "random":
                    selection = RandomTokenSelector(model_seed + int(sample_id)).select(
                        scores.token_ids, retention_ratio=ratio)
                else:
                    selection = UtilityTokenSelector().select(
                        scores.token_ids, retention_ratio=ratio, utilities=utilities)
                kept = [scores.token_ids[i] for i in selection.indices]
                received = channel.transmit(tokenizer.decode(kept, skip_special_tokens=True))
                correct = float(answer_with_receiver(
                    receiver, receiver_tokenizer, row, received,
                    config["receiver"]["max_new_tokens"]))
                fold_rows.append({
                    "fold": fold_index, "sample_id": sample_id, "method": method,
                    "retention_ratio": ratio,
                    "original_tokens": scores.num_tokens, "selected_tokens": len(kept),
                    "TRR": token_retention_ratio(len(kept), scores.num_tokens),
                    "correct": bool(correct),
                    "ground_truth": ground_truth,
                    "selected_text": received, "weights": list(best_weights)})
            gc.collect(); torch.cuda.empty_cache()
        frame = pd.DataFrame(fold_rows)
        frame.to_csv(output_dir / "fold_results.partial.csv", index=False)

    # ------------------------------------------------------------ aggregation
    frame = pd.DataFrame(fold_rows)
    frame.to_csv(output_dir / "fold_results.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(output_dir / "calibration.csv", index=False)

    per_fold = (frame.groupby(["fold", "method", "retention_ratio"], sort=True)["correct"]
                .mean().reset_index(name="accuracy"))
    summary_rows = []
    for (method, ratio), group in per_fold.groupby(["method", "retention_ratio"], sort=True):
        stats = mean_std_ci95(group["accuracy"].astype(float).tolist())
        summary_rows.append({"method": method, "retention_ratio": float(ratio),
                             "folds": len(group), "accuracy": stats["mean"],
                             "fold_std": stats["std"], "ci95": stats["ci95"],
                             "per_fold": [round(float(v), 4) for v in group["accuracy"]]})
    summary = {"num_folds": len(folds["folds"]),
               "samples": int(frame["sample_id"].nunique()),
               "dataset_seed": config["experiment"]["dataset_seed"],
               "model_seed": model_seed,
               "calibration_retention": args.calibration_retention,
               "calibration_size": args.calibration_size,
               "weights_grid_size": len(weights_grid),
               "channel": "ideal",
               "gradient_loss": "Eq. (2) answer NLL given (I, Q, P)",
               "conditions": summary_rows}
    atomic_json(output_dir / "summary.json", summary)
    atomic_json(run / "summary.json", summary)
    frame.to_csv(run / "fold_results.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
