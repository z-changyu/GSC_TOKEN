"""Scheduler tests.

The LP greedy of Eq. (53) is validated against an independent reference solver,
because the 1/2 guarantee of Eq. (55) is only meaningful if the extreme point it
rounds really is the LP optimum.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.scheduling import (best_single_mode, breakpoints, certified_rounding, lagrangian_mode,
                            solve_base_lp, solve_dual, untruncated_demand, update_price)
from src.scheduling.problem import SchedulingProblem


def scipy_lp_optimum(problem: SchedulingProblem) -> float:
    """Independent reference: the same LP solved by HiGHS."""
    from scipy.optimize import linprog

    num_tokens, num_levels = problem.num_tokens, problem.num_levels
    objective = -problem.mode_value.reshape(-1).numpy().astype(float)
    rates = problem.rates.numpy().astype(float)
    rows = [np.tile(rates, num_tokens)]
    bounds = [float(problem.budget)]
    for token in range(num_tokens):
        row = np.zeros(num_tokens * num_levels)
        row[token * num_levels:(token + 1) * num_levels] = 1.0
        rows.append(row)
        bounds.append(1.0)
    result = linprog(objective, A_ub=np.array(rows), b_ub=np.array(bounds),
                     bounds=(0, 1), method="highs")
    assert result.success
    return float(-result.fun)


def make_problem(seed: int, num_tokens: int = 6, num_levels: int = 3,
                 budget: float | None = None, beta: float = 0.25) -> SchedulingProblem:
    generator = torch.Generator().manual_seed(seed)
    utility = torch.rand(num_tokens, generator=generator)
    coupling = torch.rand(num_tokens, num_tokens, generator=generator).triu(1)
    q = 0.4 + 0.6 * torch.rand(num_levels, generator=generator)
    rates = torch.tensor([48.0, 32.0, 16.0][:num_levels])
    if budget is None:
        budget = float(rates.sum()) * num_tokens * 0.35
    return SchedulingProblem(base_utility=utility, coupling=coupling, beta=beta,
                             q=q, rates=rates, budget=budget)


def test_problem_validates_shapes_and_ranges():
    problem = make_problem(0)
    assert problem.num_tokens == 6 and problem.num_levels == 3
    with pytest.raises(ValueError):
        SchedulingProblem(base_utility=problem.base_utility, coupling=problem.coupling,
                          beta=0.0, q=torch.ones(2), rates=problem.rates, budget=1.0)
    with pytest.raises(ValueError):
        SchedulingProblem(base_utility=problem.base_utility, coupling=problem.coupling,
                          beta=0.0, q=problem.q * 2, rates=problem.rates, budget=1.0)


def test_expected_utility_matches_equation_34():
    problem = make_problem(1, num_tokens=3, num_levels=2)
    y = torch.zeros(3, 2)
    y[0, 0] = 1.0
    y[2, 1] = 0.5
    z = (y * problem.q).sum(-1)
    expected = float((problem.base_utility * z).sum())
    expected += float(problem.beta) * float(
        (problem.coupling * torch.outer(problem.base_utility * z,
                                        problem.base_utility * z)).sum())
    assert float(problem.expected_utility(y)) == pytest.approx(expected)


def test_lagrangian_mode_is_the_argmax_of_equation_52():
    problem = make_problem(2)
    for mu in (0.0, 0.005, 0.02, 0.5):
        solution = lagrangian_mode(problem, mu)
        values = problem.mode_value - mu * problem.rates
        for token in range(problem.num_tokens):
            level = int(solution.level[token])
            if level == 0:
                assert float(values[token].max()) <= 0
            else:
                assert float(values[token][level - 1]) == pytest.approx(
                    float(values[token].max()))


def test_untruncated_demand_is_monotone_in_mu():
    problem = make_problem(3, num_tokens=40)
    previous = float("inf")
    for mu in [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        demand = untruncated_demand(problem, mu)
        assert demand <= previous + 1e-9
        previous = demand
    assert untruncated_demand(problem, 1e6) == 0.0


def test_dual_minimum_agrees_with_a_dense_scan():
    problem = make_problem(4, num_tokens=20)
    best_mu, _ = solve_dual(problem)
    grid = torch.logspace(-7, 1, 400)
    values = [problem.mode_value - float(mu) * problem.rates for mu in grid]
    dual = [float(torch.clamp(v.max(dim=-1).values, min=0).sum()) + float(mu) * float(problem.budget)
            for mu, v in zip(grid, values)]
    assert min(dual) >= problem_best_dual(problem, best_mu) - 1e-4


def problem_best_dual(problem: SchedulingProblem, mu: float) -> float:
    values = problem.mode_value - mu * problem.rates
    return float(torch.clamp(values.max(dim=-1).values, min=0).sum()) + mu * float(problem.budget)


def test_breakpoints_contain_every_mode_change():
    problem = make_problem(5, num_tokens=8)
    points = [0.0] + breakpoints(problem)
    for mu in points:
        values = problem.mode_value - mu * problem.rates
        assert values.ndim == 2


def test_price_update_follows_equation_61():
    # Over-demand raises the price, under-demand lowers it, and the result is
    # projected onto [0, mu_max].
    assert update_price(0.4, 100.0, 60.0, 0.1) == pytest.approx(0.4 + 0.1 * 40.0)
    assert update_price(0.4, 20.0, 60.0, 0.005) == pytest.approx(0.4 - 0.005 * 40.0)
    assert update_price(0.01, 20.0, 60.0, 0.1) == 0.0
    assert update_price(0.4, 100.0, 60.0, 0.1, mu_max=1.0) == 1.0


@pytest.mark.parametrize("seed", range(12))
def test_base_lp_matches_the_reference_solver(seed):
    problem = make_problem(100 + seed, num_tokens=7)
    solution = solve_base_lp(problem)
    assert solution.lp_value == pytest.approx(scipy_lp_optimum(problem), rel=1e-5, abs=1e-6)


@pytest.mark.parametrize("seed", range(12))
def test_lp_solution_is_feasible_and_has_at_most_one_fractional_group(seed):
    problem = make_problem(200 + seed, num_tokens=9)
    solution = solve_base_lp(problem)
    assert problem.is_feasible(solution.y)
    fractional_rows = ((solution.y > 1e-6) & (solution.y < 1 - 1e-6)).sum(dim=-1)
    assert int((fractional_rows > 0).sum()) <= 1


@pytest.mark.parametrize("seed", range(20))
def test_certified_rounding_achieves_the_half_guarantee(seed):
    problem = make_problem(300 + seed, num_tokens=8)
    lp = solve_base_lp(problem)
    reference = scipy_lp_optimum(problem)
    assert lp.lp_value == pytest.approx(reference, rel=1e-5, abs=1e-6)

    rounded = certified_rounding(problem, lp)
    assert problem.is_feasible(rounded.y)
    assert rounded.utility >= lp.lp_value / 2.0 - 1e-6
    assert rounded.metadata["approximation_ratio"] >= 0.5 - 1e-6


def test_rounding_drops_modes_that_do_not_fit_individually():
    problem = make_problem(6, num_tokens=5)
    tiny = SchedulingProblem(base_utility=problem.base_utility, coupling=problem.coupling,
                             beta=problem.beta, q=problem.q, rates=problem.rates,
                             budget=8.0)   # below every protection cost
    rounded = certified_rounding(tiny)
    assert float(rounded.y.sum()) == 0.0
    assert rounded.utility == 0.0


def test_best_single_mode_is_feasible():
    problem = make_problem(7, num_tokens=6)
    y, value = best_single_mode(problem)
    assert problem.is_feasible(y)
    assert float(y.sum()) <= 1.0 + 1e-6
    assert value == pytest.approx(float(problem.base_expected_utility(y)))


def test_reduction_single_level_unit_q_is_capacity_constrained_selection():
    # Appendix C: q = 1 and |R| = 1 reduce the problem to picking the tokens with
    # the largest utility under a token-count budget.
    generator = torch.Generator().manual_seed(8)
    utility = torch.rand(10, generator=generator)
    problem = SchedulingProblem(base_utility=utility, coupling=torch.zeros(10, 10), beta=0.0,
                                q=torch.ones(1), rates=torch.ones(1), budget=4.0)
    rounded = certified_rounding(problem)
    assert int(rounded.selected.sum()) == 4
    top = torch.topk(utility, 4).indices.tolist()
    assert sorted(rounded.selected.nonzero().reshape(-1).tolist()) == sorted(top)


def test_reduction_beta_zero_is_the_additive_knapsack():
    # Appendix C: beta = 0 removes the coupling term, so the objective is additive
    # and the rounded utility equals the sum of the retained mode values.
    problem = make_problem(9, num_tokens=7, beta=0.0)
    rounded = certified_rounding(problem)
    assert rounded.metadata["utility_coupled"] == pytest.approx(rounded.utility)
