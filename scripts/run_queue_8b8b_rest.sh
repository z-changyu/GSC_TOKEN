#!/usr/bin/env bash
# Serialised queue for the remaining 8B->8B experiments (single GPU).
#   1. waits for the SNR sweep already running
#   2. Group-2 ablation (5 arms)
#   3. equal-cost Pareto sweep with the revised coupling definition
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1090
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate GSC_Token
export PYTHONPATH="$PWD"

CFG=configs/experiments/v2x_edge_edge_8b_rx8b_smoke.yaml
CACHE=outputs/cache/v2xqa_scores_8b_pred_cpl.jsonl
BENCH=outputs/cache/v2xqa_scores_8b_pred_cpl_surprisal.jsonl
ANSWER=outputs/cache/v2xqa_scores_8b_pred_cpl_answer_aware.jsonl

echo "[queue] waiting for the SNR sweep to finish..."
while pgrep -f "run_channel_experiment.py.*main_8b8b_snr" >/dev/null 2>&1; do sleep 60; done
echo "[queue] SNR sweep done"

echo "[queue] === group 2 ablation ==="
bash scripts/run_ablation_group2.sh

echo "[queue] === equal-cost pareto ==="
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_pareto.py \
  --score-cache "$CACHE" \
  --manifest data/manifests/v2xqa_test_filtered_8b.jsonl \
  --processed data/processed/v2xqa_test.jsonl \
  --receiver models/Qwen3-VL-8B-Instruct \
  --baseline-csv outputs/main_8b8b_full14/channel_results.csv \
  --snr-db 15 --seeds 0 1 2 \
  --coupling-variant B --lp-utility context --beta 0.13 \
  --receiver-protocol v6_norestore \
  --output-dir outputs/pareto_8b8b

echo "QUEUE_8B8B_REST_DONE"
