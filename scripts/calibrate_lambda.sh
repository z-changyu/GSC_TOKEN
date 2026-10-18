#!/usr/bin/env bash
# Recalibrate the pragmatic weights on a held-out split, target = receiver accuracy.
#
# The legacy (0.25, 0.30, 0.45) was inherited from the paper. The project notes
# record that in-fold calibration against receiver accuracy lands near
# lambda_G ~ 0.75-1.0, but the previous attempt ran on the unfiltered pool and was
# discarded. This runs on the filtered manifest (full = 100%), splits it 30/30,
# calibrates at 0 dB -- where the reliable budget actually binds and the weights
# decide which tokens survive -- and reports the winner on the untouched half.
#
# Never calibrate against transmitter-side NLL: that target drives lambda_A to 0
# and destroys the method (docs, section 11.7).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1090
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate GSC_Token
export PYTHONPATH="$PWD"

CANDIDATES=("0.25 0.30 0.45" "0.20 0.20 0.60" "0.15 0.15 0.70" "0.10 0.20 0.70"
            "0.05 0.15 0.80" "0.10 0.10 0.80" "0.00 0.20 0.80" "0.00 0.00 1.00")

for spec in "${CANDIDATES[@]}"; do
  read -r la ls lg <<< "$spec"
  tag="${la}_${ls}_${lg}"
  echo "########## weights ${tag}"
  CUDA_VISIBLE_DEVICES=0 python -u scripts/run_channel_experiment.py \
    --config configs/experiments/v2x_edge_edge_8b_rx8b.yaml \
    --manifest data/manifests/v2xqa_calib30.jsonl \
    --score-cache outputs/cache/v2xqa_scores_8b_pred_cpl.jsonl \
    --coupling-variant B --lp-utility context --beta 0.13 \
    --receiver-protocol v6_norestore --body-mask all \
    --weights "$la" "$ls" "$lg" \
    --methods proposed \
    --channels configs/channels/rayleigh.yaml \
    --snr-db 0 --channel-seeds 0 1 \
    --output-dir "outputs/lambda_calib/$tag"
done

echo "LAMBDA_CALIB_DONE"
