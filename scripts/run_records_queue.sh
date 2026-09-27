#!/usr/bin/env bash
# Two same-condition passes: the loose 15 dB instance (the budget rarely binds)
# and the binding 0 dB instance (where the schedulers actually differ).
set -u
cd /mnt/4t/code_space/zcy/GSC_Token

PY=/home/sc/miniconda3/envs/GSC_Token/bin/python
LOG=outputs/queue_records.log

while pgrep -f "run_(channel_experiment|build_score_cache)\.py|measure_task_records" > /dev/null; do sleep 60; done

echo "[$(date '+%F %H:%M')] START loose(15 dB)" >> "$LOG"
"$PY" scripts/measure_task_records.py --samples 30 --repeats 3 --warmup 5 \
  --snr-db 15 --instance-tag loose --output-dir outputs/task_records \
  >> outputs/task_records.log 2>&1
echo "[$(date '+%F %H:%M')] DONE loose (exit $?)" >> "$LOG"

echo "[$(date '+%F %H:%M')] START binding(0 dB)" >> "$LOG"
"$PY" scripts/measure_task_records.py --samples 30 --repeats 3 --warmup 5 \
  --snr-db 0 --instance-tag binding --output-dir outputs/task_records_binding \
  >> outputs/task_records_binding.log 2>&1
echo "[$(date '+%F %H:%M')] DONE binding (exit $?)" >> "$LOG"
echo "[$(date '+%F %H:%M')] QUEUE_RECORDS COMPLETE" >> "$LOG"
