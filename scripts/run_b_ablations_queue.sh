#!/usr/bin/env bash
# B-phase queue: main-line (terminal 4B -> edge 8B) versions of the ablation
# screens whose paper numbers were still taken from the reduced 2B configuration.
# All runs reuse the existing 4B score cache (60-sample filtered pool) and the
# 8B edge receiver, with damage-triggered restoration disabled so the receiver
# protocol matches the fair main line (v6-norestore).
set -u
cd /mnt/4t/code_space/zcy/GSC_Token

PY=/home/sc/miniconda3/envs/GSC_Token/bin/python
CACHE=outputs/cache/v2xqa_scores_4b.jsonl
RX=models/Qwen3-VL-8B-Instruct
FADE="configs/channels/rayleigh.yaml configs/channels/rician.yaml configs/channels/frequency_selective.yaml"
LOG=outputs/queue_b.log

while pgrep -f "run_(channel_experiment|gamma_screen|span_screen|multiterminal|pareto)\.py" > /dev/null; do sleep 30; done

run () {   # $1 tag   rest: command
  tag="$1"; shift
  echo "[$(date '+%F %H:%M')] START $tag" >> "$LOG"
  "$@" >> "outputs/$tag.log" 2>&1
  rc=$?
  echo "[$(date '+%F %H:%M')] DONE  $tag (exit $rc)" >> "$LOG"
}

# ---- 1. concave-utility screen (gamma) -------------------------------------
run gamma_screen_4b8b "$PY" scripts/run_gamma_screen.py \
  --score-cache "$CACHE" --receiver "$RX" \
  --channels $FADE --seeds 0 --restore-threshold 0.0 \
  --output-dir outputs/gamma_screen_4b8b

# ---- 2. span-granularity screen -------------------------------------------
run span_screen_4b8b "$PY" scripts/run_span_screen.py \
  --score-cache "$CACHE" --receiver "$RX" \
  --channels $FADE --seeds 0 --restore-threshold 0.0 \
  --output-dir outputs/span_screen_4b8b

# ---- 3. multi-terminal proportional fairness ------------------------------
run multiterminal_4b8b "$PY" scripts/run_multiterminal.py \
  --config configs/experiments/v2x_edge_edge_main_4b8b.yaml \
  --output-dir outputs/multiterminal_4b8b

echo "[$(date '+%F %H:%M')] QUEUE_B COMPLETE" >> "$LOG"
