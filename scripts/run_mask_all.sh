#!/usr/bin/env bash
# 14-method main table with the body mask applied to every method (except full).
#
# Pairs with outputs/main_8b8b_full14 (same config, same manifest, same seeds,
# mask off), so every method can be compared under two information conditions:
# whether or not it is allowed to spend symbols re-transmitting the question and
# options that the receiver already holds.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1090
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate GSC_Token
export PYTHONPATH="$PWD"

CUDA_VISIBLE_DEVICES=0 python -u scripts/run_channel_experiment.py \
  --config configs/experiments/v2x_edge_edge_8b_rx8b.yaml \
  --manifest data/manifests/v2xqa_test_filtered_8b.jsonl \
  --score-cache outputs/cache/v2xqa_scores_8b_pred_cpl.jsonl \
  --benchmark-scores outputs/cache/v2xqa_scores_8b_pred_cpl_surprisal.jsonl \
  --answer-scores outputs/cache/v2xqa_scores_8b_pred_cpl_answer_aware.jsonl \
  --coupling-variant B --lp-utility context --beta 0.13 \
  --receiver-protocol v6_norestore --body-mask all \
  --methods full adaptive_sc answer_loo answer_loo_uep answer_attn tokcom_uep \
            llmlingua2 efpc tokcom llmlingua longllmlingua selective_context \
            plugin_capacity proposed \
  --channels configs/channels/rayleigh.yaml \
            configs/channels/rician.yaml \
            configs/channels/frequency_selective.yaml \
  --snr-db 0 15 --channel-seeds 0 1 2 \
  --output-dir outputs/main_8b8b_maskall

echo "MASK_ALL_DONE"
