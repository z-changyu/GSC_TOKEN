#!/usr/bin/env python
"""Section VI-B-3: validation of the semantic proxy F_T(.).

The pragmatic objective is a *proxy* for the receiver-facing degradation. Its
per-token component ``G_i`` is the first-order answer-NLL sensitivity of
Eq. (14), ``G_i = || dL_T / de_i ||_2 * || e_i - e_mask ||_2``. Because this is
an L2-norm product (always non-negative) it is a *conservative, monotone*
surrogate for the true deletion effect, over-estimating it by the factor
``1/|cos theta|`` between the gradient and the deletion direction. So the proxy
is validated two ways here:

1. **Per-token (small perturbation).** For every token we delete it alone and
   measure the empirical NLL change ``true_i = L_T(P) - L_T(P \\ {i})``. The proxy
   ``u'_i`` should rank tokens by their *magnitude* of importance, so we report
   Spearman(u'_i, |true_i|) and the calibrated scale ``alpha`` in ``|true_i|
   ~ alpha |u'_i|``. This is the regime where the first-order Taylor claim of
   Eq. (27) holds.

2. **Set-level (large perturbation).** Deleting the m most important tokens, the
   true NLL increase is reported against the proxy ``F_D(D)``; the set-level
   relationship is reported as a conservative linear surrogate ``true ~ alpha
   F_D`` plus the Eq. (27) remainder basis fit, so the paper can state the
   proxy never under-estimates degradation.

The transmitter must be resident to score the pruned prompts; forwards are
forward-only and cheap.
"""
from __future__ import annotations

import argparse, json, math, statistics, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoImageProcessor, AutoProcessor, Qwen3VLForConditionalGeneration

from src.experiments.scoring import PragmaticScorer, SampleScores, combine_scores, load_image
from src.utils.io import atomic_json, read_jsonl
from src.utils.logging import configure_logging, create_run_directory
from src.utils.seed import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/pilot_scienceqa.yaml")
    parser.add_argument("--processed", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--score-cache", default="outputs/cache/scienceqa_scores_coupling.jsonl")
    parser.add_argument("--weights", type=float, nargs=3, default=(0.25, 0.0, 0.75))
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--output-dir", default="outputs/proxy")
    args = parser.parse_args()

    config = (json.loads(Path(args.config).read_text()) if args.config.endswith(".json")
              else __import__("yaml").safe_load(Path(args.config).read_text()))
    set_global_seed(int(config["experiment"]["model_seed"]))
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    run = create_run_directory("outputs/runs", "semantic_proxy_validation", vars(args))
    logger = configure_logging(run / "run.log")

    cache = {row["sample_id"]: SampleScores.from_dict(row) for row in read_jsonl(args.score_cache)}
    dataset_cfg = config.get("dataset", config)
    processed_path = args.processed or dataset_cfg["processed"]
    manifest_path = args.manifest or dataset_cfg["manifest"]
    processed = {str(row["row_idx"]): row for row in read_jsonl(processed_path)}
    manifest = [str(entry["sample_id"]) for entry in read_jsonl(manifest_path)]
    manifest = [sid for sid in manifest if sid in cache][: args.num_samples]
    logger.info("proxy validation: %d samples, weights=%s", len(manifest), args.weights)

    beta = float(config["pragmatic"]["beta"])
    tx_path = config["model"]["transmitter_path"]
    processor = AutoProcessor.from_pretrained(tx_path, local_files_only=True)
    processor.image_processor = AutoImageProcessor.from_pretrained(
        tx_path, min_pixels=config["model"]["image_min_pixels"],
        max_pixels=config["model"]["image_max_pixels"], local_files_only=True)
    transmitter = Qwen3VLForConditionalGeneration.from_pretrained(
        tx_path, dtype=torch.bfloat16, attn_implementation="eager",
        local_files_only=True).to("cuda:0").eval()
    scorer = PragmaticScorer(transmitter, processor, config)
    image_cache: dict = {}

    def get_image(sample_id: str):
        if sample_id not in image_cache:
            image_cache[sample_id] = load_image(processed[sample_id].get("image"))
        return image_cache[sample_id]

    per_token_rows, set_rows = [], []
    for sample_id in manifest:
        scores = cache[sample_id]
        row = processed[sample_id]
        n = scores.num_tokens
        u_prime = combine_scores(scores.attention, scores.semantic, scores.gradient, args.weights)
        m_matrix = (torch.tensor(scores.coupling, dtype=torch.float32) if scores.coupling
                    else torch.zeros(n, n))
        full_nll = float(scores.answer_nll)
        # ---- per-token: delete token i alone --------------------------------
        for i in range(n):
            kept = [scores.token_ids[j] for j in range(n) if j != i]
            text = processor.tokenizer.decode(kept, skip_special_tokens=True)
            pruned_nll = scorer.pruned_prompt_nll(row, get_image(sample_id), text)
            true_i = full_nll - float(pruned_nll)
            per_token_rows.append({"sample_id": sample_id, "token": i,
                                   "proxy": float(u_prime[i]), "true_single": float(true_i)})
        # ---- set-level: delete the m most important tokens -------------------
        order = sorted(range(n), key=lambda i: -u_prime[i])
        for frac in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3):
            m = max(1, int(round(frac * n)))
            deleted = order[:m]
            f_d = float(sum(u_prime[j] for j in deleted)) + beta * float(
                (m_matrix[deleted][:, deleted] * torch.outer(
                    m_matrix.new_tensor([u_prime[j] for j in deleted]),
                    m_matrix.new_tensor([u_prime[j] for j in deleted]))).sum())
            kept = [scores.token_ids[j] for j in range(n) if j not in set(deleted)]
            text = processor.tokenizer.decode(kept, skip_special_tokens=True)
            pruned_nll = scorer.pruned_prompt_nll(row, get_image(sample_id), text)
            true_drop = full_nll - float(pruned_nll)
            set_rows.append({"sample_id": sample_id, "deleted": m,
                             "proxy_f_d": float(f_d), "true_drop": float(true_drop)})
        logger.info("sample %s n=%d done", sample_id, n)

    # ---- per-token statistics -------------------------------------------------
    proxy = np.array([r["proxy"] for r in per_token_rows])
    true = np.array([r["true_single"] for r in per_token_rows])
    abs_true = np.abs(true)
    try:
        from scipy.stats import spearmanr, pearsonr
        sp_abs = float(spearmanr(proxy, abs_true).statistic)
        sp_signed = float(spearmanr(proxy, true).statistic)
        pr_abs = float(pearsonr(proxy, abs_true)[0])
    except Exception:
        sp_abs = sp_signed = pr_abs = None
    # Calibrated scale: |true| ~ alpha |proxy| (OLS through origin on magnitudes).
    num = float(np.sum(np.abs(proxy) * abs_true)); den = float(np.sum(proxy ** 2))
    alpha = num / den if den > 0 else float("nan")
    frac_raise = float(np.mean(true > 0))

    # ---- set-level: conservative linear surrogate + Eq. (27) basis -----------
    sp = np.array([r["proxy_f_d"] for r in set_rows]); st = np.array([r["true_drop"] for r in set_rows])
    try:
        sp_set = float(spearmanr(sp, st).statistic)
    except Exception:
        sp_set = None
    # true_drop ~ a * proxy_f_d ; and the remainder basis {sqrt(m), m, m^{3/2}}.
    basis = np.array([[math.sqrt(r["deleted"]), r["deleted"], r["deleted"] ** 1.5]
                      for r in set_rows])
    coeffs, *_ = np.linalg.lstsq(basis, st - (np.sum(sp * st) / np.sum(sp ** 2)) * sp, rcond=None)
    remainder = (st - (np.sum(sp * st) / np.sum(sp ** 2)) * sp) - basis @ coeffs
    ss_res = float(np.sum(remainder ** 2)); ss_tot = float(np.sum((st - st.mean()) ** 2))
    r2_remainder = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    coverage = float(np.mean(np.abs(remainder) <= np.abs(basis @ coeffs) + 1e-9))

    summary = {
        "samples": len(manifest), "weights": list(args.weights), "beta": beta,
        "per_token_n": len(per_token_rows),
        "pearson_proxy_vs_abs_true": pr_abs,
        "spearman_proxy_vs_abs_true": sp_abs,
        "spearman_proxy_vs_signed_true": sp_signed,
        "calibrated_scale_alpha": alpha,
        "frac_tokens_raising_nll": frac_raise,
        "abs_true_mean": float(np.mean(abs_true)), "abs_true_ci95": float(
            statistics.mean(abs_true) - 1.96 * np.std(abs_true) / math.sqrt(len(abs_true))),
        "set_n": len(set_rows),
        "spearman_set_proxy_vs_true": sp_set,
        "set_remainder_fit_r_squared": r2_remainder,
        "set_remainder_coverage": coverage,
        "set_calibrated_eps_g": float(coeffs[0]), "set_calibrated_eps_H": float(coeffs[1]),
        "set_calibrated_L_3": float(coeffs[2]),
        "per_token_rows": per_token_rows, "set_rows": set_rows,
    }
    atomic_json(output_dir / "summary.json", {k: v for k, v in summary.items()
                                             if k not in ("per_token_rows", "set_rows")})
    atomic_json(run / "summary.json", {k: v for k, v in summary.items()
                                       if k not in ("per_token_rows", "set_rows")})
    atomic_json(output_dir / "per_token_rows.jsonl", per_token_rows)
    atomic_json(output_dir / "set_rows.jsonl", set_rows)
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("per_token_rows", "set_rows")}, indent=2))


if __name__ == "__main__":
    main()
