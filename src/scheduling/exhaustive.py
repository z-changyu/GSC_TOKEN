"""Exhaustive optimum of the quadratic multiple-choice problem.

Proposition 1 (Eq. 64) promises ``F(y_alg) >= F(y*) / (2 (1 + kappa))`` with
``kappa = beta d_M / 2``. Verifying that bound needs the true optimum, and the
problem is NP-hard in general, so the search is only tractable for small token
counts: a token has ``|R| + 1`` choices (off, or one of the protection modes),
hence ``(|R|+1)^N`` assignments. With ``|R| = 3`` that is about one million at
``N = 10`` and seventeen million at ``N = 12``, which is enumerated in row
blocks to keep the memory footprint flat.
"""
from __future__ import annotations

import numpy as np
import torch

from .problem import SchedulingProblem


def enumerate_choices(num_tokens: int, num_levels: int, block: int = 1 << 21):
    """Yield ``[block, N]`` int8 arrays of per-token choices, 0 = not transmitted."""
    base = num_levels + 1
    total = base ** num_tokens
    powers = base ** np.arange(num_tokens, dtype=np.int64)
    for start in range(0, total, block):
        codes = np.arange(start, min(start + block, total), dtype=np.int64)[:, None]
        yield ((codes // powers[None, :]) % base).astype(np.int8)


def exhaustive_optimum(problem: SchedulingProblem, block: int = 1 << 21) -> dict:
    """Brute-force the exact optimum of Eq. (34) over every feasible assignment."""
    num_tokens, num_levels = problem.num_tokens, problem.num_levels
    utility = problem.base_utility.numpy().astype(np.float64)
    rates = problem.rates.numpy().astype(np.float64)
    q = problem.q.numpy().astype(np.float64)
    coupling = problem.coupling.numpy().astype(np.float64)
    beta, budget = float(problem.beta), float(problem.budget)

    best_value, best_assignment = -np.inf, None
    for choices in enumerate_choices(num_tokens, num_levels, block):
        mode = choices.astype(np.int64)
        transmitted = mode > 0
        z = np.where(transmitted, q[np.arange(num_tokens)[None, :],
                                    np.clip(mode - 1, 0, None)], 0.0)
        resource = np.where(transmitted, rates[np.clip(mode - 1, 0, None)], 0.0).sum(axis=1)
        weighted = z * utility[None, :]
        # Coupling term with the *joint* pair probability: for a common channel the
        # pair survives with Q[lvl_i, lvl_j], not with the product of marginals.
        # Q defaults to the product when the caller did not supply a joint matrix,
        # in which case this is numerically the previous reference objective.
        joint = problem.joint_q.numpy().astype(np.float64)
        pair_index = np.clip(mode - 1, 0, None)
        pair = joint[pair_index[:, :, None], pair_index[:, None, :]]          # [b,N,N]
        pair = np.where(transmitted[:, :, None] & transmitted[:, None, :], pair, 0.0)
        # M is strictly upper triangular, so the double sum is over i < j only.
        values = weighted.sum(axis=1) + beta * np.einsum(
            "bij,ij->b", pair, coupling * np.outer(utility, utility), optimize=True)
        feasible = resource <= budget + 1e-9
        if not feasible.any():
            continue
        candidate = float(values[feasible].max())
        if candidate > best_value:
            best_value = candidate
            best_assignment = mode[feasible][int(np.argmax(values[feasible]))].copy()

    if best_assignment is None:
        return {"value": 0.0, "assignment": np.zeros(num_tokens, dtype=np.int64)}
    return {"value": best_value, "assignment": best_assignment}


def assignment_to_level(assignment) -> torch.Tensor:
    """Convert a per-token mode array (0 = off) into the level convention of the solvers."""
    array = np.asarray(assignment, dtype=np.int64)
    return torch.as_tensor(array, dtype=torch.long)


def schedule_value(problem: SchedulingProblem, level: torch.Tensor) -> float:
    y = torch.zeros_like(problem.mode_value)
    selected = level > 0
    if bool(selected.any()):
        y[selected, level[selected] - 1] = 1.0
    return float(problem.expected_utility(y))
