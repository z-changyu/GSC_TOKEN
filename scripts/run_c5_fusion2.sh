#!/usr/bin/env bash
# C5: cross-view fusion at the main line, with an explicit dual-device override.
# The first attempt OOMed because the config pins CUDA_VISIBLE_DEVICES=0, so the
# driver's own dual-device placement could not take effect. Runs after C4.
set -u
cd /mnt/4t/code_space/zcy/GSC_Token

PY=/home/sc/miniconda3/envs/GSC_Token/bin/python
MAIN=configs/experiments/v2x_edge_edge_main_4b8b.yaml
LOG=outputs/queue_c5.log

while pgrep -f "run_(channel_experiment|build_score_cache|v2x_co_fusion|multiterminal)\.py|run_c4_multiterminal" > /dev/null; do sleep 60; done

echo "[$(date '+%F %H:%M')] START v2x_co_fusion_4b8b (CUDA_VISIBLE_DEVICES=0,1)" >> "$LOG"
CUDA_VISIBLE_DEVICES=0,1 "$PY" scripts/run_v2x_co_fusion.py --config "$MAIN" \
  --output-dir outputs/v2x_co_fusion_4b8b >> outputs/v2x_co_fusion_4b8b.log 2>&1
rc=$?
echo "[$(date '+%F %H:%M')] DONE  v2x_co_fusion_4b8b (exit $rc)" >> "$LOG"
echo "[$(date '+%F %H:%M')] QUEUE_C5 COMPLETE" >> "$LOG"
