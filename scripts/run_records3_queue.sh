#!/usr/bin/env bash
# Re-measure the same-condition task records with the revised accounting:
#   * balanced readout roster (every method, not just the randomised first one);
#   * main metric boundary without the on-board trigger and the answer downlink;
#   * explicit information-bit / coded-bit / channel-use / window mapping;
#   * TokCom external signal acquisition timed (offline cost still unmeasured);
#   * serialisation split from the allocator release, batched transfer as the
#     official path with the per-element path kept as a diagnostic.
set -u
cd "$(dirname "$0")/.." || exit 1
P=/home/sc/miniconda3/envs/GSC_Token/bin/python
LOG=outputs/queue_records3.log
: > "$LOG"
run() {  # $1 = instance tag, $2 = snr dB, $3 = output dir
    echo "[$(date +%m-%d_%H:%M)] START $1 ($2 dB)" | tee -a "$LOG"
    "$P" scripts/measure_task_records.py --snr-db "$2" --instance-tag "$1" \
         --output-dir "$3" >> "$LOG" 2>&1
    echo "[$(date +%m-%d_%H:%M)] DONE $1 exit=$?" | tee -a "$LOG"
}
run loose 15 outputs/task_records
run binding 0 outputs/task_records_binding
echo "[$(date +%m-%d_%H:%M)] QUEUE_RECORDS3 COMPLETE" | tee -a "$LOG"
