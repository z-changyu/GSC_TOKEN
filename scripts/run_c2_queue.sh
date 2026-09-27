#!/usr/bin/env bash
# C2: cross-domain (ScienceQA) reproduction at the main line 4B -> 8B.
# (The fusion item of queue C hit a dual-model OOM and is handled separately.)
set -u
cd /mnt/4t/code_space/zcy/GSC_Token

PY=/home/sc/miniconda3/envs/GSC_Token/bin/python
SQA=configs/experiments/channel_scienceqa_4b8b.yaml
LOG=outputs/queue_c2.log

while pgrep -f "run_(channel_experiment|build_score_cache)\.py" > /dev/null; do sleep 30; done

run () {
  tag="$1"; shift
  echo "[$(date '+%F %H:%M')] START $tag" >> "$LOG"
  "$@" >> "outputs/$tag.log" 2>&1
  rc=$?
  echo "[$(date '+%F %H:%M')] DONE  $tag (exit $rc)" >> "$LOG"
}

run cache_sqa_4b "$PY" scripts/build_score_cache.py --config "$SQA"

run channel_scienceqa_4b8b "$PY" scripts/run_channel_experiment.py \
  --config "$SQA" --weights 0.25 0.0 0.75 \
  --receiver-protocol v6_norestore --output-dir outputs/channel_scienceqa_4b8b

echo "[$(date '+%F %H:%M')] QUEUE_C2 COMPLETE" >> "$LOG"
