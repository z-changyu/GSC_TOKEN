"""Method dispatch for the reliability-aware schedule.

``select_level`` returns the per-token protection level, with ``0`` meaning "not
transmitted". The four scheduling behaviours differ only in how much of
Algorithm 1 they use, which is exactly the ladder the ablation study needs:

``full``
    whole prompt at the strongest level, budget deliberately ignored, so the
    symbol cost of an unprotected reference stays visible;
``random``
    uniformly random tokens at the strongest level until the budget is gone;
``fixed_H``
    utility-greedy at a single protection level — the ablation that removes
    differentiated protection;
``proposed``
    Lagrangian LP (Eq. 53), certified extreme-point rounding (Eq. 54) and
    coupling-aware local refinement (Eq. 56-59).

``truncate`` / ``selective_context`` / ``llmlingua``
    external prompt-compression baselines, ranked by token position, by LM
    self-information (Li et al., EMNLP 2023) and by coarse-to-fine LM perplexity
    (Jiang et al., ACL 2023) respectively. All three transmit at the strongest
    level only and use the same budget as ``fixed_H``, so the comparison isolates
    the *selection criterion*.

The internal plug-in-capacity baseline of Section VI-A is *not* a separate
scheduler: it is ``proposed`` driven by the budget ``T_s B log2(1+|h_hat|^2 g)``
instead of Eq. (6), so the caller passes a different budget rather than a
different method.
"""
from __future__ import annotations

import math
import random

import torch

from .problem import SchedulingProblem

SCHEDULING_METHODS = ("full", "random", "truncate", "selective_context", "llmlingua",
                      "longllmlingua", "llmlingua2", "efpc", "tokcom", "answer_loo",
                      "answer_attn", "adaptive_sc", "fixed_H", "proposed")


def _fill_strongest(order, level: torch.Tensor, rates: torch.Tensor,
                    budget: float) -> torch.Tensor:
    """Transmit tokens in ``order`` at the strongest level until the budget is gone.

    Every selection-only method shares this template, so the comparison isolates
    the *selection criterion* and not the protection policy.
    """
    strongest_cost = float(rates[0])
    if not math.isfinite(budget):
        # The ideal channel has an unbounded reliable budget, so every token fits.
        # Keeping this explicit avoids int(inf) and NaN comparisons downstream.
        for token in order:
            level[token] = 1
        return level
    spent = 0.0
    for token in order:
        if spent + strongest_cost <= budget:
            level[token] = 1
            spent += strongest_cost
    return level


def _llmlingua_order(score: torch.Tensor, target_keep: int, segment_size: int = 16,
                     coarse_headroom: float = 1.3) -> list[int]:
    """Coarse-to-fine perplexity pruning, the core of LLMLingua (Jiang et al. 2023).

    Stage 1 (coarse): rank fixed-size segments by mean self-information and keep
    the most informative ones until a deliberately generous token budget is met.
    Stage 2 (fine): among the surviving tokens drop the least informative until
    the exact token budget is met.
    """
    n = int(score.numel())
    target_keep = max(0, min(int(target_keep), n))
    if target_keep >= n:
        return list(range(n))
    segments = [(s, min(s + segment_size, n)) for s in range(0, n, segment_size)]
    seg_mean = [float(score[s:e].mean()) for s, e in segments]
    seg_order = sorted(range(len(segments)), key=lambda k: -seg_mean[k])
    coarse_budget = min(n, int(math.ceil(target_keep * coarse_headroom)))
    kept: list[int] = []
    used = 0
    for k in seg_order:
        if used >= coarse_budget:
            break
        s, e = segments[k]
        kept.extend(range(s, e))
        used += e - s
    kept.sort(key=lambda i: -float(score[i]))
    return kept[:target_keep]


def _select_level(method: str, utility: torch.Tensor, coupling: torch.Tensor, beta: float,
                 q: torch.Tensor, rates: torch.Tensor, budget: float, t_max: int = 8,
                 rng: random.Random | None = None,
                 alt_score: torch.Tensor | None = None,
                 joint_q: torch.Tensor | None = None,
                 selection_utility: torch.Tensor | None = None) -> torch.Tensor:
    """Per-token protection level; ``0`` means the token is not transmitted."""
    if method not in SCHEDULING_METHODS:
        raise ValueError(f"unknown method {method!r}; available: {SCHEDULING_METHODS}")
    utility = torch.as_tensor(utility, dtype=torch.float32).reshape(-1)
    rates = torch.as_tensor(rates, dtype=torch.float32).reshape(-1)
    num_tokens = int(utility.numel())
    level = torch.zeros(num_tokens, dtype=torch.long)
    strongest_cost = float(rates[0])          # codebook is ordered r_H > r_M > r_L

    if method == "full":
        level[:] = 1
        return level

    if method == "random":
        order = list(range(num_tokens))
        (rng or random.Random(0)).shuffle(order)
        return _fill_strongest(order, level, rates, budget)

    if method == "truncate":
        # Naive leading-token truncation -- the degenerate lexical baseline that
        # the LLMLingua paper reports alongside its own method.
        return _fill_strongest(list(range(num_tokens)), level, rates, budget)

    if method in ("selective_context", "llmlingua", "longllmlingua", "tokcom",
                  "llmlingua2", "efpc", "answer_loo", "answer_attn"):
        if alt_score is None:
            raise ValueError(f"{method!r} needs the LM self-information scores "
                             f"(pass alt_score)")
        score = torch.as_tensor(alt_score, dtype=torch.float32).reshape(-1)
        if score.numel() != num_tokens:
            raise ValueError(f"{method!r}: self-information has {score.numel()} entries "
                             f"but the prompt has {num_tokens} tokens")
        if method in ("llmlingua", "longllmlingua"):
            # Jiang et al.: coarse-to-fine pruning under a budget controller. The
            # symbol budget is converted to a token count first. ``llmlingua``
            # ranks by plain PPL; ``longllmlingua`` by the question-aware
            # contrastive relevance I_context(x) - I_question(x), which the caller
            # has already folded into ``alt_score``.
            if not math.isfinite(budget) or strongest_cost <= 0:
                target = num_tokens
            else:
                target = int(budget // strongest_cost)
            order = _llmlingua_order(score, target)
        else:
            # selective_context / tokcom / llmlingua2 / answer_loo / answer_attn:
            # keep the top-scored tokens. ``tokcom`` ranks by the transmitter's
            # cross-modal context attention (Qiao et al., 2025); ``llmlingua2`` by
            # a trained token classifier (Xiao et al., 2024); ``answer_loo`` by
            # the exact leave-one-out answer-NLL importance (the gold standard);
            # ``answer_attn`` by the receiver's answer-grounded attention.
            order = sorted(range(num_tokens), key=lambda i: -float(score[i]))
        return _fill_strongest(order, level, rates, budget)

    if method == "adaptive_sc":
        # Adaptive-SC: every transmitted token gets the *same* protection level,
        # but both the level and the token count adapt to the channel -- classical
        # rate adaptation. The level maximises the expected delivered utility
        # q_ell * sum(top-n utilities), with n the count the budget affords there.
        if not math.isfinite(budget):
            return _fill_strongest(sorted(range(num_tokens),
                                          key=lambda i: -float(utility[i])),
                                   level, rates, budget)
        probs = torch.as_tensor(q, dtype=torch.float32).reshape(-1)
        ranked = sorted(range(num_tokens), key=lambda i: -float(utility[i]))
        best = None
        for ell in range(int(rates.numel())):
            cost = float(rates[ell])
            if cost <= 0:
                continue
            affordable = int(budget // cost)
            if affordable <= 0:
                continue
            prob = float(probs[ell]) if ell < int(probs.numel()) else 0.0
            top = ranked[:affordable]
            expected = prob * float(sum(float(utility[i]) for i in top))
            if best is None or expected > best[0]:
                best = (expected, ell, top)
        if best is None:
            return level
        for token in best[2]:
            level[token] = best[1] + 1          # level 0 means "not transmitted"
        return level

    if method == "fixed_H":
        # Ablation arm "set-aware score, fixed protection": the ranking may come from
        # the Eq. (19) context-aware utility while every transmitted token still gets
        # the strongest level, which isolates the value of *differentiated* protection
        # from the value of the set-aware ranking itself.
        ranking = utility if selection_utility is None else selection_utility
        order = sorted(range(num_tokens), key=lambda i: -float(ranking[i]))
        return _fill_strongest(order, level, rates, budget)

    from .lp_rounding import certified_rounding
    from .local_refinement import refine

    problem = SchedulingProblem(base_utility=utility, coupling=coupling, beta=beta,
                                q=q, rates=rates, budget=float(budget), joint_q=joint_q)
    # The base LP and the certified rounding are additive by construction (Eq. 51-55),
    # so on their own they can only express u'_i q_{i,ell}: the set dependence enters
    # exclusively through the local refinement, which is why it moves so few
    # decisions. Passing ``selection_utility`` (the context-aware utility u_i of
    # Eq. 19) makes the LP itself solve the *linearised* set objective, so the
    # coupling also steers which tokens are selected. Feasibility is unchanged
    # because both problems share the same budget, rates and q.
    lp_problem = problem if selection_utility is None else SchedulingProblem(
        base_utility=selection_utility, coupling=coupling, beta=beta, q=q, rates=rates,
        budget=float(budget), joint_q=joint_q)
    rounded = certified_rounding(lp_problem)
    if t_max <= 0:
        return rounded.level
    # Refinement always improves the *exact* set objective of Eq. (34).
    return refine(problem, rounded.level, t_max=t_max).level


def select_level(method: str, utility: torch.Tensor, coupling: torch.Tensor, beta: float,
                 q: torch.Tensor, rates: torch.Tensor, budget: float, t_max: int = 8,
                 rng: random.Random | None = None,
                 alt_score: torch.Tensor | None = None,
                 joint_q: torch.Tensor | None = None,
                 selection_utility: torch.Tensor | None = None,
                 candidate_mask: torch.Tensor | None = None) -> torch.Tensor:
    """``_select_level`` restricted to ``candidate_mask``.

    ``candidate_mask`` marks the tokens a method is allowed to transmit. The
    body-mask experiment uses it to drop the duplicated question/option prefix,
    which the receiver already gets through ``build_question_block``.

    Restricting the candidate set has to happen *inside* the selection, not by
    voiding the result afterwards: a method that spent budget on an excluded token
    and had it deleted afterwards would silently lose that budget, which would
    penalise exactly the baselines the comparison is supposed to be fair to. So
    every ranking sees the restriction, and the order-based methods (random,
    truncate) shuffle/truncate only over the allowed tokens.
    """
    if candidate_mask is None:
        return _select_level(method, utility, coupling, beta, q, rates, budget, t_max,
                             rng, alt_score, joint_q, selection_utility)
    mask = torch.as_tensor(candidate_mask, dtype=torch.bool).reshape(-1)
    utility = torch.as_tensor(utility, dtype=torch.float32).reshape(-1)
    if mask.numel() != utility.numel():
        raise ValueError(f"candidate_mask has {mask.numel()} entries but the prompt has "
                         f"{utility.numel()} tokens")
    allowed = [index for index in range(utility.numel()) if bool(mask[index])]
    if not allowed:
        raise ValueError("candidate_mask excludes every token")
    rates_t = torch.as_tensor(rates, dtype=torch.float32).reshape(-1)
    num_tokens = utility.numel()

    if method == "random":
        order = list(allowed)
        (rng or random.Random(0)).shuffle(order)
        return _fill_strongest(order, torch.zeros(num_tokens, dtype=torch.long),
                               rates_t, budget)
    if method == "truncate":
        return _fill_strongest(allowed, torch.zeros(num_tokens, dtype=torch.long),
                               rates_t, budget)

    masked_utility = utility.clone()
    masked_utility[~mask] = 0.0
    masked_selection = selection_utility
    if selection_utility is not None:
        masked_selection = torch.as_tensor(
            selection_utility, dtype=torch.float32).reshape(-1).clone()
        masked_selection[~mask] = 0.0
    masked_alt = alt_score
    if alt_score is not None:
        masked_alt = torch.as_tensor(alt_score, dtype=torch.float32).reshape(-1).clone()
        # Rank excluded tokens last instead of at -inf, so percentile-based
        # controllers (LLMLingua's budget search) still receive finite values.
        masked_alt[~mask] = float(masked_alt.min()) - 1.0

    level = _select_level(method, masked_utility, coupling, beta, q, rates, budget,
                          t_max, rng, masked_alt, joint_q, masked_selection)
    level = level.clone()
    level[~mask] = 0
    return level
