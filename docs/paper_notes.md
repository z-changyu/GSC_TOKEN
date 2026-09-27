# Paper method and experiment audit

Source: `Pragmatics-Aware Token Selection and Scheduling for Generative Semantic Communication over Time-Varying Channels.pdf` (6 pages, created 2026-07-27).

## System model

A VLM maps multimodal input to prompt tokens \(P=\{w_i\}_{i=1}^N\), with embeddings \(e_i\) and binary transmission decisions \(x_i\). Priority-dependent coding costs are \(r_i\in\{r_H,r_M,r_L\}\), so \(R=\sum_i x_i r_i\). Under estimated CSI \(\hat h_\tau=h_\tau+e_\tau\), capacity is \(C_\tau=B\log_2(1+|\hat h_\tau|^2\bar\gamma)\), subject to \(R\le C_\tau\). Packet loss is abstracted at whole-token granularity after decoding.

## Token pragmatic utility

- Structural importance, Eq. (1): \(A_i=H^{-1}\sum_h[N^{-1}\sum_j\alpha^{(h)}_{j,i}]\). It is incoming attention averaged over heads and source/query tokens.
- Semantic direction, Eq. (2): \(g=\sum_i A_i e_i/\sum_i A_i\), then \(S_i=e_i^Tg/(\|e_i\|_2\|g\|_2)\).
- Gradient sensitivity, Eq. (3): \(G_i=\|\partial\mathcal L/\partial e_i\|_2\), with \(\mathcal L\) described only as transmitter-VLM autoregressive NLL.
- Base utility, Eq. (4): \(u'_i=\lambda_1\tilde A_i+\lambda_2\tilde S_i+\lambda_3\tilde G_i\), weights nonnegative and summing to one.
- Coupling, Eq. (5): \(M_{ij}=H^{-1}\sum_h(\alpha^{(h)}_{i,j}+\alpha^{(h)}_{j,i})/2\).
- Context-aware utility, Eq. (6): \(u_i=u'_i+\beta\sum_{j\ne i}M_{ij}u'_j\), with the stated dominance bound on beta. It is precomputed on the full token set, which the paper acknowledges causes selection-set overestimation bias.

## Selection and scheduling

The additive objective is a capacity-constrained 0–1 knapsack. Lagrangian relaxation yields density \(\rho_i=u_i/r_i\) and threshold decision \(x_i=1\) when \(\rho_i>\mu\). Stage 1 greedily accumulates feasible tokens in descending density. Stage 2 proposes coupling-aware swaps using the stated insertion gain \(\Delta_j^+=u'_j+2\beta\sum_{m\in T_0}M_{jm}u'_m\), but removal loss is not explicitly defined. Online per-terminal dual variables track capacity. The coordinator allocates bandwidth as \(B_k=B[p_k|\hat h_k|]/\sum_j[p_j|\hat h_j|]\), where \(p_k=N_k^{-1}\sum_i u_i^{(k)}/r_i^{(k)}\).

## Existing experiment and limitations

The paper reports 5,000 image-containing ScienceQA samples with five-fold cross-validation, BLIP-2 transmitter, Flan-T5-Large receiver, CLIP ViT-L/14, prompt truncation at 64, coding rates 48/32/16, legacy weights 0.25/0.30/0.45, beta 0.25, eight swap rounds, and dual step 0.1. It compares four channels at 0/15 dB and several baselines.

Reproducibility gaps include: no exact model repository/revision or software environment; no prompt/chat templates; no split IDs or fold manifests; no precise NLL target for gradients; no attention layer aggregation policy; no normalization definition; no exact removal-loss equation; no decoding parameters; no random/channel seeds; no official QA normalization protocol; no confidence intervals; no raw per-sample outputs; and no derivation of token coding-symbol costs. Truncating to 64 precedes a reported token-length study. Reported cross-validation and hyperparameter grid search do not state how test leakage is prevented. These are audit findings, not changes to the method.

