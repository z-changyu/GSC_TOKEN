#!/usr/bin/env bash
# VI-B-4 model mismatch: calibrate the schedule for one receiver, deploy on others.
#
# The transmitter and the schedule are identical in every run (same 8B score cache,
# same methods, same seeds, same channel realisations). Only the receiver changes,
# 1.7B -> 4B -> 8B, and the answer prompt is shifted as well (v0 description-only
# vs v6_norestore task-intent + letter output). Whatever accuracy moves is therefore
# attributable to the receiver, not to the compressor: that gap is epsilon_model.
#
# Reported per (receiver, prompt, method, SNR) on the same 60 samples.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1090
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate GSC_Token
export PYTHONPATH="$PWD"

for rx in 1b7 4b 8b; do
  if [ "$rx" = "8b" ]; then cfg=configs/experiments/v2x_edge_edge_8b_rx8b.yaml
  else cfg=configs/experiments/v2x_edge_edge_8b_rx${rx}.yaml; fi
  for proto in v0 v6_norestore; do
    echo "########## receiver=${rx} protocol=${proto}"
    CUDA_VISIBLE_DEVICES=0 python -u scripts/run_channel_experiment.py \
      --config "$cfg" \
      --manifest data/manifests/v2xqa_test_filtered_8b.jsonl \
      --score-cache outputs/cache/v2xqa_scores_8b_pred_cpl.jsonl \
      --coupling-variant B --lp-utility context --beta 0.13 \
      --receiver-protocol "$proto" --body-mask all \
      --methods full proposed \
      --channels configs/channels/rayleigh.yaml \
      --snr-db 0 15 --channel-seeds 0 1 \
      --output-dir "outputs/model_mismatch/${rx}_${proto}"
  done
done

echo "MODEL_MISMATCH_DONE"
