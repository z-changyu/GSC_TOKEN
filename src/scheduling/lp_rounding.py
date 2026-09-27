"""Base multiple-choice LP and the certified extreme-point rounding, Eq. (53)-(55).

The additive relaxation of Eq. (34) is the one-resource LP (53). It is *not*
solved by filling modes in descending density ``a_{i,ell}/r_ell``: that greedy is
optimal for a plain knapsack but wrong here, because at the optimal multiplier a
token may prefer a costlier mode with a larger absolute contribution over a
cheaper one with a higher density. The paper's route is used instead — search the
breakpoints of the dual multiplier in (52) and recover the primal at the optimal
breakpoint.

Writing ``g(mu) = mu C + sum_i max(0, max_ell (a_{i,ell} - mu r_ell))``, ``g`` is
convex and piecewise linear, so its minimum is attained at a breakpoint. At that
point the tokens with strictly positive reduced cost are taken at their optimal
mode, and the tokens whose reduced value is exactly zero form the boundary set.
Every boundary mode has density exactly ``mu*``, so filling the residual capacity
from that set in any order yields the extreme optimum whose single fractional
group the rounding argument of Appendix B needs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .problem import ScheduleResult, SchedulingProblem


@dataclass
class LPSolution:
    y: torch.Tensor
    fractional_token: int | None
    lp_value: float
    resource_use: float
    multiplier: float = 0.0


def _feasible_columns(rates: torch.Tensor, budget: float) -> torch.Tensor:
    """Appendix B: discard modes with ``r_ell > C`` before solving."""
    return rates <= float(budget)


def _breakpoints(a: torch.Tensor, r: torch.Tensor) -> list[float]:
    points = set()
    for ell in range(int(r.numel())):
        points.update((a[:, ell] / r[ell]).tolist())
        for other in range(ell + 1, int(r.numel())):
            if float(r[ell]) == float(r[other]):
                continue
            points.update(((a[:, ell] - a[:, other]) / (r[ell] - r[other])).tolist())
    return sorted(point for point in points if point > 0 and math.isfinite(point))


def solve_base_lp(problem: SchedulingProblem, tolerance: float = 1e-6) -> LPSolution:
    """Eq. (53): extreme optimum of the base LP via dual breakpoint search."""
    rates = problem.rates
    budget = float(problem.budget)
    feasible = _feasible_columns(rates, budget)
    y = torch.zeros_like(problem.mode_value)
    if not bool(feasible.any()):
        return LPSolution(y=y, fractional_token=None, lp_value=0.0, resource_use=0.0)

    columns = torch.nonzero(feasible).reshape(-1)
    a = problem.mode_value[:, feasible]
    r = rates[feasible]
    # Decoding probability of every feasible mode. Tokens sharing a slot see the
    # same link, so the row is constant across tokens.
    qf = problem.q[0][feasible]

    # The candidate set is every breakpoint of g(mu) plus the two ends, i.e.
    # O(N|R|^2) points (407 at N=135, |R|=3). Evaluating g one candidate at a time
    # costs one tensor pass per breakpoint and dominates the scheduler: measured
    # 12.6 ms P50 / 29.7 ms P95 per call, against 0.6 ms for the refinement. All
    # candidates are therefore reduced in one batched pass. The per-candidate
    # arithmetic is kept identical -- float32 for the reduced values, float64 for
    # the final sum, exactly as the scalar version did -- so the minimiser, and
    # hence every scheduling decision, is unchanged; the equivalence is verified
    # by ``scripts/check_scheduler_equivalence.py``.
    candidates = [0.0] + _breakpoints(a, r) + [float((a / r).max()) + 1.0]
    mu_vec = torch.tensor(candidates, dtype=a.dtype)
    chunk = max(1, int(4_000_000 // max(1, a.numel())))
    pieces = []
    for start in range(0, int(mu_vec.numel()), chunk):
        block = mu_vec[start:start + chunk]
        reduced_block = a.unsqueeze(0) - block.reshape(-1, 1, 1) * r.reshape(1, 1, -1)
        pieces.append(reduced_block.max(dim=-1).values.clamp(min=0).sum(dim=-1).double()
                      + block.double() * float(budget))
    dual_values = torch.cat(pieces)
    # ``min(candidates, key=dual)`` takes the first minimiser in candidate order;
    # ``argmin`` is free to break ties differently, so pick the first exact one.
    best_dual = dual_values.min()
    mu_star = float(mu_vec[(dual_values == best_dual).nonzero()[0, 0]])

    # Primal recovery at the optimal breakpoint. A token whose reduced value is
    # uniquely maximised gets an integral decision. A token whose modes *tie* at
    # mu* is the fractional group: just below the breakpoint it prefers the
    # costlier tied mode and just above the cheaper one, so splitting between
    # them tunes its resource use continuously while the reduced value stays
    # constant. Starting every tied token at its cheapest tied mode reproduces
    # the right-hand limit D_right <= C, and shifting mass towards the costliest
    # tied mode walks the resource use up to C. Appendix B guarantees only one
    # such group, so at most one token is left fractional.
    reduced = a - mu_star * r
    best_value, best_index = reduced.max(dim=-1)
    tie_tolerance = 1e-6 * max(1.0, float(best_value.abs().max()))
    optimal = reduced >= (best_value - tie_tolerance).unsqueeze(-1)
    tied = optimal.sum(dim=-1) > 1
    positive = best_value > tie_tolerance
    integral = (~tied) & positive

    # Primal recovery, vectorised: the three loops below used to cost one
    # ``torch.nonzero`` plus a scalar tensor write per token (139 calls at
    # N=135), which is what kept the solver above the 1 ms slot. Every quantity
    # they need is a per-token reduction of ``optimal`` / ``best_index``, so all
    # tokens are handled in one pass; the only sequential parts that remain are
    # the tied-mode walk and the residual fill, which the theory bounds to a
    # single fractional token.
    used = 0.0
    fractional_token: int | None = None

    integral_tokens = torch.nonzero(integral).reshape(-1)
    if integral_tokens.numel():
        integral_levels = best_index[integral_tokens]
        y[integral_tokens, columns[integral_levels]] = 1.0
        # ``used`` must be accumulated in the same order as the scalar loop did:
        # the residual fill below is decided by comparisons that sit exactly on a
        # cost boundary (the LP optimum is tight), so a different summation order
        # shifts which token ends up fractional. A ~1e2-iteration Python sum costs
        # microseconds -- the expensive part of the old loop was the per-token
        # ``torch.nonzero`` and scalar tensor write, not the float addition.
        for cost in r[integral_levels].tolist():
            used += cost

    # Only walk the resource use of a tied token towards its costlier tied mode
    # when that mode actually buys reliability. Tied modes share the same reduced
    # value, so when they also share the same decoding probability -- a good
    # channel where every mode decodes equally well -- spending the slack on the
    # costlier mode leaves Eq. (34) unchanged while burning the very resource the
    # schedule is meant to economise. Skipping the shift keeps the same LP value
    # (the tied modes have identical ``a``), so the 1/2 guarantee of Eq. (55) is
    # untouched.
    adjustable: list[tuple[int, int, int, float, float]] = []
    tied_tokens = torch.nonzero(tied & positive).reshape(-1)
    if tied_tokens.numel():
        # Cheapest / costliest *tied* mode. The costly one is found as the
        # minimiser of the negated cost: an ``argmax`` over a mask padded with
        # ``+inf`` would return the padding instead of the costliest real mode.
        masked_cost = torch.where(optimal, r.unsqueeze(0),
                                  torch.full_like(a, math.inf))
        cheap_levels = masked_cost.argmin(dim=-1)[tied_tokens]
        costly_levels = torch.where(optimal, -r.unsqueeze(0),
                                    torch.full_like(a, math.inf)).argmin(
                                        dim=-1)[tied_tokens]
        y[tied_tokens, columns[cheap_levels]] = 1.0
        for cost in r[cheap_levels].tolist():      # same order as the scalar loop
            used += cost
        shiftable = (r[costly_levels] > r[cheap_levels]) & (qf[costly_levels] > qf[cheap_levels])
        adjustable = [(int(token), int(cheap), int(costly),
                       float(r[cheap]), float(r[costly]))
                      for token, cheap, costly in zip(tied_tokens[shiftable].tolist(),
                                                      cheap_levels[shiftable].tolist(),
                                                      costly_levels[shiftable].tolist())]

    remaining = budget - used
    for token, cheap, costly, cheap_rate, costly_rate in adjustable:
        if remaining <= tolerance:
            break
        span = costly_rate - cheap_rate
        shift = min(remaining, span)
        theta = shift / span
        y[token, int(columns[cheap])] = 1.0 - theta
        y[token, int(columns[costly])] = theta
        used += shift
        remaining -= shift
        if theta < 1.0 - 1e-9:
            fractional_token = int(token)
            break

    # Tokens whose best reduced value is zero are value-neutral; they only ever
    # matter when the budget is not already exhausted. Among the modes attaining
    # the best value take the cheapest: they are worth the same to Eq. (34), so
    # the extra symbols buy nothing.
    if remaining > tolerance:
        zero_tokens = torch.nonzero(~positive).reshape(-1)
        if zero_tokens.numel():
            row_best = a.max(dim=-1).values
            # ``float(row.max()) - 1e-12`` was evaluated in float64 and then cast,
            # so the threshold is built the same way to keep the mode sets equal.
            threshold = (row_best.double() - 1e-12).float()
            top_modes = a >= threshold.unsqueeze(-1)
            cheapest = torch.where(top_modes, r.unsqueeze(0),
                                   torch.full_like(a, math.inf)).argmin(dim=-1)
            zero_levels = cheapest[zero_tokens]
            # The fill itself stays sequential -- it decides which token, if any,
            # is left fractional, and the running ``used`` is compared against the
            # budget exactly as the scalar loop did. Only the tensor writes are
            # batched, which is where the time went.
            costs = r[zero_levels].tolist()
            filling = 0
            fractional_index: int | None = None
            for index, cost in enumerate(costs):
                if remaining <= tolerance:
                    break
                if used + cost <= budget + tolerance:
                    used += cost
                    remaining -= cost
                    filling += 1
                else:
                    fractional_index = index
                    break
            if filling:
                y[zero_tokens[:filling], columns[zero_levels[:filling]]] = 1.0
            if fractional_index is not None:
                cost = costs[fractional_index]
                level = int(zero_levels[fractional_index])
                y[int(zero_tokens[fractional_index]), int(columns[level])] = remaining / cost
                fractional_token = int(zero_tokens[fractional_index])

    return LPSolution(y=y, fractional_token=fractional_token,
                      lp_value=float(problem.base_expected_utility(y)),
                      resource_use=float(problem.resource_use(y)),
                      multiplier=mu_star)


def best_single_mode(problem: SchedulingProblem) -> tuple[torch.Tensor, float]:
    """``y^S``: the best single token-mode pair that fits the budget on its own."""
    y = torch.zeros_like(problem.mode_value)
    feasible = _feasible_columns(problem.rates, problem.budget)
    if not bool(feasible.any()):
        return y, 0.0
    values = torch.where(feasible.reshape(1, -1), problem.mode_value,
                         torch.full_like(problem.mode_value, -math.inf))
    flat = int(values.argmax())
    if not math.isfinite(float(values.reshape(-1)[flat])):
        return y, 0.0
    token, level = divmod(flat, problem.num_levels)
    y[token, level] = 1.0
    return y, float(problem.base_expected_utility(y))


def certified_rounding(problem: SchedulingProblem,
                       lp: LPSolution | None = None) -> ScheduleResult:
    """Eq. (54): ``y_hat = argmax{F_base(y^I), F_base(y^S)}``.

    Guarantees Eq. (55), ``F_base(y_hat) >= F_base(y*) / 2``, because the
    fractional group's LP value is no larger than that of its best individually
    feasible mode.
    """
    lp = lp if lp is not None else solve_base_lp(problem)
    y_integral = torch.where(lp.y >= 1.0 - 1e-6, torch.ones_like(lp.y), torch.zeros_like(lp.y))
    value_integral = float(problem.base_expected_utility(y_integral))
    y_single, value_single = best_single_mode(problem)

    if value_integral >= value_single:
        chosen, value, origin = y_integral, value_integral, "integral_groups"
    else:
        chosen, value, origin = y_single, value_single, "best_single_mode"

    level = torch.zeros(problem.num_tokens, dtype=torch.long)
    rows = chosen.sum(dim=-1) > 0.5
    if bool(rows.any()):
        level[rows] = chosen[rows].argmax(dim=-1) + 1
    return ScheduleResult(
        y=chosen, level=level, utility=value,
        metadata={"lp_value": lp.lp_value, "lp_resource_use": lp.resource_use,
                  "lp_multiplier": lp.multiplier,
                  "fractional_token": lp.fractional_token,
                  "value_integral": value_integral, "value_single": value_single,
                  "rounding_origin": origin,
                  "approximation_ratio": (value / lp.lp_value) if lp.lp_value > 0 else None,
                  "utility_coupled": float(problem.expected_utility(chosen))})
