#!/usr/bin/env bash
# B-phase: main-line robustness run on the *unfiltered* 150-question pool.
# Both caches are rebuilt with the 4B terminal transmitter, then the experiment
# runs with the fair receiver protocol.
set -u
cd /mnt/4t/code_space/zcy/GSC_Token

PY=/home/sc/miniconda3/envs/GSC_Token/bin/python
CFG=configs/experiments/v2x_edge_edge_main_4b8b_pool150.yaml
LOG=outputs/queue_150p.log

while pgrep -f "run_(channel_experiment|gamma_screen|span_screen|multiterminal|pareto)\.py|run_b_ablations_queue" > /dev/null; do sleep 60; done

run () {   # $1 tag   rest: command
  tag="$1"; shift
  echo "[$(date '+%F %H:%M')] START $tag" >> "$LOG"
  "$@" >> "outputs/$tag.log" 2>&1
  rc=$?
  echo "[$(date '+%F %H:%M')] DONE  $tag (exit $rc)" >> "$LOG"
}

run cache_4b_pool150 "$PY" scripts/build_score_cache.py \
  --config "$CFG"

run bench_4b_pool150 "$PY" scripts/build_benchmark_scores.py \
  --config "$CFG" --output outputs/cache/v2xqa_scores_4b_pool150_surprisal.jsonl

run channel_main_4b8b_pool150 "$PY" scripts/run_channel_experiment.py \
  --config "$CFG" --weights 0.25 0.30 0.45 \
  --receiver-protocol v6_norestore --output-dir outputs/channel_main_4b8b_pool150

echo "[$(date '+%F %H:%M')] QUEUE150 COMPLETE" >> "$LOG"
