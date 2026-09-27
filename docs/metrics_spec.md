# Metrics specification

- QA/VQA accuracy follows the dataset's official protocol when available. Exact match uses lowercase, punctuation/article removal, and whitespace normalization.
- A-OKVQA/VQA soft consensus is `min(number of matching annotators / 3, 1)`; the downloaded schema must be verified before use.
- Token retention ratio is `selected_tokens / original_tokens`, with tokenizer boundaries recorded.
- Transmitted symbols are `sum(x_i * r_i)`. Token count and symbol budget are distinct axes.
- AUBC is the trapezoidal integral of accuracy over budget divided by the evaluated budget span.
- Target-accuracy saving is `1 - R_proposed(A_t) / R_baseline(A_t)` after documented curve interpolation.
- Repeated experiments report mean, sample standard deviation, and normal-approximation 95% CI; all channel seeds are saved.

Every sample result must include original/selected token counts, TRR, transmitted symbols, prediction, ground truth, correctness, runtime, and peak GPU memory.

