#!/usr/bin/env bash
# C4: full-scale multi-terminal sweep on the main line.
# The original run estimated every (K, theta, SNR) cell from a single channel
# realisation of 60 samples, i.e. 60 units -- 6 slots at K=10 and only 3 at K=20 --
# so accuracies were quantised to 1/60 (e.g. 20/60 = 33.3%). Every channel seed
# now re-evaluates the same slot partition, giving 3 x 60 = 180 units per cell
# (18 slots at K=10, 9 at K=20) on an identical question pool and protocol.
set -u
cd /mnt/4t/code_space/zcy/GSC_Token

PY=/home/sc/miniconda3/envs/GSC_Token/bin/python
MAIN=configs/experiments/v2x_edge_edge_main_4b8b.yaml
LOG=outputs/queue_c4.log

while pgrep -f "run_(channel_experiment|build_score_cache|v2x_co_fusion)\.py|run_c2_queue|run_c3_fusion" > /dev/null; do sleep 60; done

echo "[$(date '+%F %H:%M')] START multiterminal_4b8b_s3" >> "$LOG"
"$PY" scripts/run_multiterminal.py --config "$MAIN" \
  --output-dir outputs/multiterminal_4b8b_s3 >> outputs/multiterminal_4b8b_s3.log 2>&1
echo "[$(date '+%F %H:%M')] DONE  multiterminal_4b8b_s3 (exit $?)" >> "$LOG"
echo "[$(date '+%F %H:%M')] QUEUE_C4 COMPLETE" >> "$LOG"
