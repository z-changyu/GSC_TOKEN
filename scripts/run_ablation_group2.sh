#!/usr/bin/env bash
# Group 2 -- pairwise ablation of the set-dependent scheduler.
#
# Five arms, each a *separate single-method run* so that the per-(sample,condition)
# random-number trajectory is identical across arms (the driver consumes rng
# sequentially across methods, so batching them would make the arms incomparable).
#
#   A1  independent score + joint selection/protection
#       coupling applied only in the local refinement (marginal LP)  -> "no set-aware LP"
#   A2  set-aware score + fixed protection
#       Eq. (19) ranking but every transmitted token at the strongest level
#   A3  set-aware score + joint selection/protection  (the complete method)
#   A4  shuffled coupling + joint selection/protection
#       symmetric permutation: same coefficients, wrong partners -> isolates structure
#   A5  task compression + ordinary unequal error protection
#       the 2025 TokCom signal driving the same scheduler, without set awareness
#
# All arms: 60 samples x 3 channels x SNR {0,15} x 3 seeds, protocol v6_norestore.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1090
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate GSC_Token
export PYTHONPATH="$PWD"

CFG=configs/experiments/v2x_edge_edge_8b_rx8b_smoke.yaml
BENCH=outputs/cache/v2xqa_scores_8b_pred_cpl_surprisal.jsonl
ANSWER=outputs/cache/v2xqa_scores_8b_pred_cpl_answer_aware.jsonl
COMMON=(--config "$CFG" --methods proposed --snr-db 0 15
        --receiver-protocol v6_norestore --beta 0.13)

run () {  # run <tag> <extra args...>
  local tag="$1"; shift
  echo "########## $tag"
  CUDA_VISIBLE_DEVICES=0 python -u scripts/run_channel_experiment.py "${COMMON[@]}" "$@" \
    --output-dir "outputs/ablation2/$tag"
}

run A1_marginal_lp  --coupling-variant B --lp-utility marginal
run A2_fixed_prot   --coupling-variant B --lp-utility context --protection-mode fixed
run A3_full         --coupling-variant B --lp-utility context
run A4_shuffled     --coupling-variant B --lp-utility context --coupling-shuffle

echo "########## A5 tokcom_uep (task compression + ordinary UEP)"
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_channel_experiment.py \
  --config "$CFG" --methods tokcom_uep --snr-db 0 15 --receiver-protocol v6_norestore \
  --beta 0.13 --coupling-variant raw --lp-utility marginal \
  --benchmark-scores "$BENCH" --answer-scores "$ANSWER" \
  --output-dir outputs/ablation2/A5_tokcom_uep

echo "ABLATION_GROUP2_DONE"
