#!/usr/bin/env bash
# C-phase queue: the two remaining paper-visible items that still carried
# reduced-configuration numbers.
#   1. cross-domain reproduction (ScienceQA) at the main line 4B -> 8B
#   2. cross-view report fusion at the main line 4B -> 8B
set -u
cd /mnt/4t/code_space/zcy/GSC_Token

PY=/home/sc/miniconda3/envs/GSC_Token/bin/python
MAIN=configs/experiments/v2x_edge_edge_main_4b8b.yaml
SQA=configs/experiments/channel_scienceqa_4b8b.yaml
LOG=outputs/queue_c.log

while pgrep -f "run_(channel_experiment|gamma_screen|span_screen|multiterminal|pareto|v2x_co_fusion|build_score_cache)\.py" > /dev/null; do sleep 30; done

run () {   # $1 tag   rest: command
  tag="$1"; shift
  echo "[$(date '+%F %H:%M')] START $tag" >> "$LOG"
  "$@" >> "outputs/$tag.log" 2>&1
  rc=$?
  echo "[$(date '+%F %H:%M')] DONE  $tag (exit $rc)" >> "$LOG"
}

# ---- 1. cross-domain (ScienceQA) at the main line ---------------------------
run cache_sqa_4b "$PY" scripts/build_score_cache.py --config "$SQA"

run channel_scienceqa_4b8b "$PY" scripts/run_channel_experiment.py \
  --config "$SQA" --weights 0.25 0.0 0.75 \
  --receiver-protocol v6_norestore --output-dir outputs/channel_scienceqa_4b8b

# ---- 2. cross-view fusion at the main line ---------------------------------
# The fusion driver scores both viewpoints on the fly with the transmitter in the
# config, so pointing it at the main-line config is enough (4B transmitter,
# 8B coordinator); it sweeps v0 and v6-norestore by default.
run v2x_co_fusion_4b8b "$PY" scripts/run_v2x_co_fusion.py \
  --config "$MAIN" --output-dir outputs/v2x_co_fusion_4b8b

echo "[$(date '+%F %H:%M')] QUEUE_C COMPLETE" >> "$LOG"
