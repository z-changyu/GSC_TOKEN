# Open questions and method-critical blockers

## Gradient sensitivity \(G_i\)

Problem: Equation (3) defines \(G_i=\|\partial\mathcal L/\partial e_i\|_2\), but the paper only calls \(\mathcal L\) the transmitter VLM autoregressive NLL.

Cause: The target sequence, teacher-forcing policy, exact embedding tensor/layer, prompt/output masking, sequence reduction, and whether NLL covers the full output are unspecified.

Evidence: Section III-A(3), Eq. (3), provides no further operational definition.

Possible solutions: full generated-sequence NLL; ground-truth semantic-prompt NLL; QA-target NLL; token-local loss. These are not theoretically equivalent.

Recommended option: decide the theory before formal experiments. The code exposes `GradientScorer(loss_type, target_mode)` and requires the caller to construct the loss explicitly. No default scientific definition is hidden.

**Need final theoretical decision on the exact definition of L in G_i.**

## Attention and coupling under Qwen3-VL

Problem: \(A_i\) and \(M_{ij}\) require genuine aligned self-attention weights. Modern SDPA paths may return no weights, and multimodal visual tokens complicate alignment with textual prompt tokens.

Cause: Backbone and Transformers attention implementations differ; the old paper used BLIP-2, while this pilot proposes Qwen3-VL.

Evidence: GPU/model capability validation is currently blocked because `nvidia-smi` cannot communicate with the driver. The exact Transformers behavior has not been claimed or approximated.

Possible solutions: test eager attention first, then SDPA only if it returns identical, correctly aligned tensors; define an explicit text-token slice after inspecting the processor template.

Recommended option: require a PASS from `scripts/test_model_loading.py` using eager attention and inspect tensor/token alignment. Never use hidden-state cosine similarity as fake \(M_{ij}\).

## Local swapping ambiguity

Problem: The insertion gain is shown, but the paper says the removal loss is “computed analogously” without an explicit equation. It also does not fully specify recomputation after each accepted swap.

Recommended option: keep local swapping disabled in the minimal pilot until the authors specify removal loss and update semantics.

## Dataset protocols

ScienceQA uses answer-choice indices; GQA evaluation commonly normalizes direct answers; A-OKVQA contains multiple direct answers and rationales and may require official consensus evaluation. Repository schemas and split names must be inspected after download. Metadata preservation is mandatory.

## Old-paper versus new-pilot comparability

The PDF reports BLIP-2, Flan-T5-Large, 5,000 image-containing ScienceQA samples, five-fold cross-validation, and \(N\le64\). The new Qwen3 configuration and 100–200 fixed validation samples are a pipeline pilot, not a reproduction of reported performance.
