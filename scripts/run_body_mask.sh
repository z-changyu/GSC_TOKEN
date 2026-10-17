#!/usr/bin/env bash
# Does excluding the duplicated question/option prefix from the candidate set help?
#
# ``proposed_body`` is the same scheduler on the same budget as ``proposed``; the
# only difference is that the leading span -- which the receiver already receives
# via build_question_block -- leaves the candidate set. The freed symbols are
# re-optimised over the report body.
#
# SNR 0/5/10 dB: the reliable budget binds at 0-5 dB (726-1506 symbols against a
# ~130-token payload), and saturates by 10 dB. That bracket is where a
# reallocation can matter and where it must stop mattering.
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
  --receiver-protocol v6_norestore \
  --methods full proposed proposed_body \
  --channels configs/channels/rayleigh.yaml \
            configs/channels/rician.yaml \
            configs/channels/frequency_selective.yaml \
  --snr-db 0 5 10 --channel-seeds 0 1 2 \
  --output-dir outputs/body_mask_8b8b_v2

echo "BODY_MASK_DONE"
