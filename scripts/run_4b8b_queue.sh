#!/usr/bin/env bash
# Serial queue for the 4B->8B configuration (single GPU only).
# Waits for any running channel experiment to finish, then runs the remaining
# items one after another.  Progress is appended to outputs/queue_4b8b.log.
set -u
cd /mnt/4t/code_space/zcy/GSC_Token

PY=/home/sc/miniconda3/envs/GSC_Token/bin/python
CFG=configs/experiments/v2x_edge_edge_main_4b8b.yaml
W="--weights 0.25 0.30 0.45"
LOG=outputs/queue_4b8b.log
mkdir -p outputs

while pgrep -f run_channel_experiment.py > /dev/null; do sleep 60; done

step () {
  tag="$1"; cfg="$2"; shift 2
  echo "[$(date '+%F %H:%M')] START $tag" >> "$LOG"
  "$PY" scripts/run_channel_experiment.py --config "$cfg" $W \
        --output-dir "outputs/$tag" "$@" >> "outputs/$tag.log" 2>&1
  echo "[$(date '+%F %H:%M')] DONE  $tag (exit $?)" >> "$LOG"
}

# 1. protocol arm with completion enabled (third column of tab:protocol)
step channel_main_4b8b_v6      "$CFG" --receiver-protocol v6

# 2. bandwidth sweep (main run already covers 1 MHz)
step channel_main_4b8b_b025    configs/experiments/v2x_edge_edge_main_4b8b_b025.yaml
step channel_main_4b8b_b050    configs/experiments/v2x_edge_edge_main_4b8b_b050.yaml
step channel_main_4b8b_b200    configs/experiments/v2x_edge_edge_main_4b8b_b200.yaml

# 3. CSI-error sweep (main run covers sigma_e^2 = 0.01)
step channel_main_4b8b_csise005 "$CFG" \
  --channels configs/channels/rayleigh_se005.yaml \
             configs/channels/rician_se005.yaml \
             configs/channels/frequency_selective_se005.yaml
step channel_main_4b8b_csise010 "$CFG" \
  --channels configs/channels/rayleigh_se010.yaml \
             configs/channels/rician_se010.yaml \
             configs/channels/frequency_selective_se010.yaml

# 4. iso-cost budget sweep (feeding the Pareto figure)
echo "[$(date '+%F %H:%M')] START pareto_main_4b8b" >> "$LOG"
"$PY" scripts/run_pareto.py \
  --score-cache outputs/cache/v2xqa_scores_4b.jsonl \
  --baseline-csv outputs/channel_main_4b8b_fair/channel_results.csv \
  --budgets 400 600 800 1200 1600 2000 2600 3200 4000 \
  --output-dir outputs/pareto_main_4b8b >> outputs/pareto_main_4b8b.log 2>&1
echo "[$(date '+%F %H:%M')] DONE  pareto_main_4b8b" >> "$LOG"

echo "[$(date '+%F %H:%M')] QUEUE COMPLETE" >> "$LOG"
