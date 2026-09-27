"""Lagrangian relaxation of the base problem, Eq. (51), (52), (60), (61).

For a fixed multiplier ``mu`` the additive problem separates across tokens, so
the mode decision is the exact maximiser of Eq. (52). The demand that drives the
online price update is deliberately the *untruncated* one of Eq. (60): the LP
rounded set is what gets transmitted, but only the untruncated demand preserves
the correct subgradient of the dual function.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .problem import SchedulingProblem


@dataclass
class LagrangianSolution:
    level: torch.Tensor        # 0 = not transmitted, else 1..|R|
    reduced_value: torch.Tensor
    demand: float
    dual_value: float


def mode_reduced_value(problem: SchedulingProblem, mu: float) -> torch.Tensor:
    """``q_{i,ell} u'_i - mu r_ell`` for every token and protection mode."""
    return problem.mode_value - float(mu) * problem.rates.reshape(1, -1)


def lagrangian_mode(problem: SchedulingProblem, mu: float) -> LagrangianSolution:
    """Eq. (52): the exact mode decision of the Lagrangian subproblem at ``mu``.

    Level 0 encodes the "not transmitted" option with ``q_{i0} = r_0 = 0``.
    """
    values = mode_reduced_value(problem, mu)
    best_value, best_level = values.max(dim=-1)
    transmit = best_value > 0
    level = torch.where(transmit, best_level + 1, torch.zeros_like(best_level))
    demand = float(problem.rates[(level - 1).clamp(min=0)][transmit].sum())
    dual = float(best_value.clamp(min=0).sum()) + float(mu) * float(problem.budget)
    return LagrangianSolution(level=level, reduced_value=best_value,
                              demand=demand, dual_value=dual)


def untruncated_demand(problem: SchedulingProblem, mu: float) -> float:
    """Eq. (60): ``D(mu) = sum_i r_{i, ell*_i(mu)}`` before any rounding."""
    return lagrangian_mode(problem, mu).demand


def dual_value(problem: SchedulingProblem, mu: float) -> float:
    """The dual function ``g(mu)``, convex and piecewise linear in ``mu``."""
    return lagrangian_mode(problem, mu).dual_value


def update_price(mu: float, demand: float, budget: float, step: float,
                 mu_max: float | None = None) -> float:
    """Eq. (61): projected subgradient step on the untruncated demand."""
    updated = float(mu) + float(step) * (float(demand) - float(budget))
    updated = max(0.0, updated)
    if mu_max is not None:
        updated = min(updated, float(mu_max))
    return updated


def breakpoints(problem: SchedulingProblem) -> list[float]:
    """Every ``mu`` at which the argmax of Eq. (52) can change.

    Modes tie when ``a_{i,ell} - mu r_ell = a_{i,m} - mu r_m`` and drop out when
    ``a_{i,ell} - mu r_ell = 0``. The Lagrangian is linear between consecutive
    breakpoints, so evaluating only these points is enough to minimise the dual.
    """
    a = problem.mode_value
    r = problem.rates
    points = {0.0}
    for ell in range(problem.num_levels):
        points.update((a[:, ell] / r[ell]).tolist())
        for other in range(ell + 1, problem.num_levels):
            if float(r[ell]) == float(r[other]):
                continue
            points.update(((a[:, ell] - a[:, other]) / (r[ell] - r[other])).tolist())
    return sorted(value for value in points if value > 0 and value == value)


def solve_dual(problem: SchedulingProblem) -> tuple[float, dict]:
    """Minimise the dual over breakpoints and return the optimal multiplier."""
    candidates = [0.0] + breakpoints(problem)
    best_mu, best_value = candidates[0], dual_value(problem, candidates[0])
    for mu in candidates[1:]:
        value = dual_value(problem, mu)
        if value < best_value:
            best_mu, best_value = mu, value
    return best_mu, {"dual_value": best_value, "num_breakpoints": len(candidates)}
