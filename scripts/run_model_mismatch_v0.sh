#!/usr/bin/env bash
# Valid v0 ladder for the model-mismatch study.
#
# v0's receiver prompt is built from the received text alone -- the question and the
# options travel inside the payload. Under --body-mask all those tokens are removed
# (which is correct for v6, whose prompt supplies them out of band), so the v0 arm
# of the previous sweep could not answer at all: its near-zero accuracy measures the
# broken task, not the receiver. This re-runs v0 with the mask off, which is the
# only self-consistent way to use that protocol, and gives a ladder that can be
# compared against the v6 + mask-all one.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1090
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate GSC_Token
export PYTHONPATH="$PWD"

for rx in 1b7 4b 8b; do
  if [ "$rx" = "8b" ]; then cfg=configs/experiments/v2x_edge_edge_8b_rx8b.yaml
  else cfg=configs/experiments/v2x_edge_edge_8b_rx${rx}.yaml; fi
  echo "########## receiver=${rx} protocol=v0 mask=off"
  CUDA_VISIBLE_DEVICES=0 python -u scripts/run_channel_experiment.py \
    --config "$cfg" \
    --manifest data/manifests/v2xqa_test_filtered_8b.jsonl \
    --score-cache outputs/cache/v2xqa_scores_8b_pred_cpl.jsonl \
    --coupling-variant B --lp-utility context --beta 0.13 \
    --receiver-protocol v0 --body-mask off \
    --methods full proposed \
    --channels configs/channels/rayleigh.yaml \
    --snr-db 0 15 --channel-seeds 0 1 \
    --output-dir "outputs/model_mismatch_v0/${rx}_v0"
done

echo "MODEL_MISMATCH_V0_DONE"
