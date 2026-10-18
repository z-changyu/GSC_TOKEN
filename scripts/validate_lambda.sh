#!/usr/bin/env bash
# Validate the calibrated weights against the legacy triple on the full manifest.
#
# The 30-sample calibration could not separate the candidates (60 observations,
# ~12pp CI), so the top two and the legacy triple are re-run on all 60 samples at
# 0 dB -- the SNR where the budget binds and the weights actually decide which
# tokens survive. Caveat: half of these samples took part in the selection step,
# so the absolute numbers carry a mild optimism; the paired ordering is what this
# run is for.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1090
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate GSC_Token
export PYTHONPATH="$PWD"

for spec in "0.25 0.30 0.45" "0.10 0.20 0.70" "0.15 0.15 0.70"; do
  read -r la ls lg <<< "$spec"
  tag="${la}_${ls}_${lg}"
  echo "########## weights ${tag}"
  CUDA_VISIBLE_DEVICES=0 python -u scripts/run_channel_experiment.py \
    --config configs/experiments/v2x_edge_edge_8b_rx8b.yaml \
    --manifest data/manifests/v2xqa_test_filtered_8b.jsonl \
    --score-cache outputs/cache/v2xqa_scores_8b_pred_cpl.jsonl \
    --coupling-variant B --lp-utility context --beta 0.13 \
    --receiver-protocol v6_norestore --body-mask all \
    --weights "$la" "$ls" "$lg" \
    --methods proposed \
    --channels configs/channels/rayleigh.yaml \
              configs/channels/rician.yaml \
              configs/channels/frequency_selective.yaml \
    --snr-db 0 --channel-seeds 0 1 2 \
    --output-dir "outputs/lambda_validate/$tag"
done

echo "LAMBDA_VALIDATE_DONE"
