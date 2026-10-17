#!/usr/bin/env bash
# Fill the four SNR cells that the interrupted sweep never finished.
# Existing partial covers rayleigh{5,10,20} and rician{5,10}; missing are
# rician/20 and frequency_selective{5,10,20}.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1090
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate GSC_Token
export PYTHONPATH="$PWD"

CFG=configs/experiments/v2x_edge_edge_8b_rx8b_smoke.yaml
METHODS=(full adaptive_sc answer_loo answer_loo_uep answer_attn tokcom_uep llmlingua2
         efpc tokcom llmlingua longllmlingua selective_context plugin_capacity proposed)
BENCH=outputs/cache/v2xqa_scores_8b_pred_cpl_surprisal.jsonl
ANSWER=outputs/cache/v2xqa_scores_8b_pred_cpl_answer_aware.jsonl
COMMON=(--config "$CFG" --methods "${METHODS[@]}"
        --benchmark-scores "$BENCH" --answer-scores "$ANSWER"
        --coupling-variant B --lp-utility context --beta 0.13
        --receiver-protocol v6_norestore)

echo "########## rician / 20 dB"
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_channel_experiment.py "${COMMON[@]}" \
  --channels configs/channels/rician.yaml --snr-db 20 \
  --output-dir outputs/main_8b8b_snr_ric20

echo "########## frequency-selective / 5,10,20 dB"
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_channel_experiment.py "${COMMON[@]}" \
  --channels configs/channels/frequency_selective.yaml --snr-db 5 10 20 \
  --output-dir outputs/main_8b8b_snr_fs

echo "SNR_FILL_DONE"
