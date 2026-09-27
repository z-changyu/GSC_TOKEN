#!/usr/bin/env bash
# C3: cross-view report fusion at the main line (4B terminal -> 8B coordinator).
# Runs after C2 so the two jobs never contend for the same device.
set -u
cd /mnt/4t/code_space/zcy/GSC_Token

PY=/home/sc/miniconda3/envs/GSC_Token/bin/python
MAIN=configs/experiments/v2x_edge_edge_main_4b8b.yaml
LOG=outputs/queue_c3.log

while pgrep -f "run_(channel_experiment|build_score_cache)\.py|run_c2_queue" > /dev/null; do sleep 60; done

echo "[$(date '+%F %H:%M')] START v2x_co_fusion_4b8b" >> "$LOG"
"$PY" scripts/run_v2x_co_fusion.py --config "$MAIN" \
  --output-dir outputs/v2x_co_fusion_4b8b >> outputs/v2x_co_fusion_4b8b.log 2>&1
echo "[$(date '+%F %H:%M')] DONE  v2x_co_fusion_4b8b (exit $?)" >> "$LOG"
echo "[$(date '+%F %H:%M')] QUEUE_C3 COMPLETE" >> "$LOG"
