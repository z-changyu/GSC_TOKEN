#!/usr/bin/env bash
# Corrected serial queue: the bandwidth and CSI sweeps must use the fair
# receiver protocol (v6_norestore) exactly like the main run, otherwise they
# are v0 curves and not comparable with tab:main.
set -u
cd /mnt/4t/code_space/zcy/GSC_Token

PY=/home/sc/miniconda3/envs/GSC_Token/bin/python
CFG=configs/experiments/v2x_edge_edge_main_4b8b.yaml
W="--weights 0.25 0.30 0.45"
PROTO="--receiver-protocol v6_norestore"
LOG=outputs/queue_4b8b2.log

while pgrep -f run_channel_experiment.py > /dev/null; do sleep 30; done

step () {           # $1 tag  $2 config  rest args
  tag="$1"; cfg="$2"; shift 2
  echo "[$(date '+%F %H:%M')] START $tag" >> "$LOG"
  "$PY" scripts/run_channel_experiment.py --config "$cfg" $W $PROTO \
        --output-dir "outputs/$tag" "$@" >> "outputs/$tag.log" 2>&1
  rc=$?
  proto=$("$PY" - "$tag" <<'PYEOF'
import sys, pandas as pd
tag = sys.argv[1]
try:
    d = pd.read_csv(f"outputs/{tag}/channel_results.csv")
    print(",".join(sorted(d.receiver_protocol.unique())))
except Exception:
    print("MISSING")
PYEOF
)
  echo "[$(date '+%F %H:%M')] DONE  $tag (exit $rc, protocol=$proto)" >> "$LOG"
}

# ---- bandwidth sweep (1 MHz already covered by the main run) ----
step channel_main_4b8b_b025 configs/experiments/v2x_edge_edge_main_4b8b_b025.yaml
step channel_main_4b8b_b050 configs/experiments/v2x_edge_edge_main_4b8b_b050.yaml
step channel_main_4b8b_b200 configs/experiments/v2x_edge_edge_main_4b8b_b200.yaml

# ---- CSI-error sweep (sigma_e^2 = 0.01 already covered) ----
step channel_main_4b8b_csise005 "$CFG" \
  --channels configs/channels/rayleigh_se005.yaml \
             configs/channels/rician_se005.yaml \
             configs/channels/frequency_selective_se005.yaml
step channel_main_4b8b_csise010 "$CFG" \
  --channels configs/channels/rayleigh_se010.yaml \
             configs/channels/rician_se010.yaml \
             configs/channels/frequency_selective_se010.yaml

# ---- iso-cost budget sweep on the main-line edge model + fair protocol ----
echo "[$(date '+%F %H:%M')] START pareto_main_4b8b" >> "$LOG"
"$PY" scripts/run_pareto.py \
  --score-cache outputs/cache/v2xqa_scores_4b.jsonl \
  --baseline-csv outputs/channel_main_4b8b_fair/channel_results.csv \
  --receiver models/Qwen3-VL-8B-Instruct \
  --receiver-protocol v6_norestore \
  --budgets 400 600 800 1200 1600 2000 2600 3200 4000 \
  --output-dir outputs/pareto_main_4b8b >> outputs/pareto_main_4b8b.log 2>&1
echo "[$(date '+%F %H:%M')] DONE  pareto_main_4b8b" >> "$LOG"

echo "[$(date '+%F %H:%M')] QUEUE2 COMPLETE" >> "$LOG"
