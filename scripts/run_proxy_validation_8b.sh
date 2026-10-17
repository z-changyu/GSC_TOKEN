#!/usr/bin/env bash
# VI-B-3 semantic proxy validation, current implementation, both target protocols.
#
# The 2026-08 run measured the true deletion effect on the GOLD answer while the
# proxy was built on the PREDICTED answer, so the two quantities described
# different things. Running both targets with everything else fixed isolates how
# much of the reported -0.178 set-level correlation was that mismatch.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1090
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate GSC_Token
export PYTHONPATH="$PWD"

COMMON=(--config configs/experiments/v2x_edge_edge_8b_rx8b.yaml
        --manifest data/manifests/v2xqa_test_filtered_8b.jsonl
        --processed data/processed/v2xqa_test.jsonl
        --score-cache outputs/cache/v2xqa_scores_8b_pred_cpl.jsonl
        --weights 0.25 0.30 0.45
        --num-samples 20)

echo "########## target=predicted"
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_proxy_validation.py "${COMMON[@]}" \
  --score-target predicted --output-dir outputs/proxy_8b_pred

echo "########## target=gold (legacy protocol)"
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_proxy_validation.py "${COMMON[@]}" \
  --score-target gold --output-dir outputs/proxy_8b_gold

echo "PROXY_VALIDATION_DONE"
