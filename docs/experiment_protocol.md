# Experiment protocol

1. Freeze sample IDs in a manifest with dataset seed 2026. Never resample per run.
2. Inspect dataset distributions before selecting a maximum prompt length. The old `N <= 64` is not inherited.
3. Validate image preprocessing, chat template, token alignment, hidden states, genuine attention, embedding gradients, dtype, memory, and determinism on one sample.
4. Gate the pilot on those checks. Use an ideal channel only.
5. Run the same fixed 150 ScienceQA validation samples for full, seeded random, and minimal additive utility selection at TRR 0.3, 0.5, 0.7, and 1.0.
6. Do not tune on a test split. Save resolved config, results, summary, log, and environment snapshot in a unique run directory.
7. Stop before scale-up if any core capability fails, or if NaN/Inf, token misalignment, implausible full-prompt accuracy, or nondeterminism is observed.

Hardware constraint: expose exactly one RTX 3090 (`CUDA_VISIBLE_DEVICES=0`). Multi-GPU sharding and automatic placement across both installed GPUs are forbidden for this stage.

The weights (0.25, 0.30, 0.45) and beta 0.25 are **legacy paper parameters**, not optimized values for Qwen3.
