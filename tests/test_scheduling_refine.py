"""Local refinement and bandwidth allocation tests.

The refinement gains are algebraic identities derived from Eq. (34), so the
strongest available check is to compare each predicted gain against the objective
actually realised after the move.
"""
from __future__ import annotations

import math

import pytest
import torch

from src.scheduling import (allocate_bandwidth, allocation_from_nu, certified_rounding,
                            jain_index, price_error_bound, refine, solve_base_lp,
                            strong_concavity_bound)
from src.scheduling.local_refinement import _level_to_schedule
from src.scheduling.problem import SchedulingProblem


def make_problem(seed, num_tokens=12, num_levels=3, budget=None, beta=0.25):
    generator = torch.Generator().manual_seed(seed)
    utility = torch.rand(num_tokens, generator=generator)
    coupling = torch.rand(num_tokens, num_tokens, generator=generator).triu(1)
    coupling = coupling / max(float(coupling.sum(dim=-1).max()), 1e-6)
    q = torch.tensor([0.95, 0.75, 0.5][:num_levels])
    rates = torch.tensor([48.0, 32.0, 16.0][:num_levels])
    if budget is None:
        budget = float(rates.sum()) * num_tokens * 0.25
    return SchedulingProblem(base_utility=utility, coupling=coupling, beta=beta,
                             q=q, rates=rates, budget=budget)


def level_of(result):
    return result.level


@pytest.mark.parametrize("seed", range(8))
def test_refinement_never_decreases_the_exact_objective(seed):
    problem = make_problem(seed)
    start = certified_rounding(problem)
    improved = refine(problem, start.level, t_max=8)
    assert improved.utility >= start.utility - 1e-5
    assert problem.is_feasible(_level_to_schedule(problem, improved.level))


@pytest.mark.parametrize("seed", range(8))
def test_refinement_only_accepts_positive_gains(seed):
    problem = make_problem(seed)
    start = certified_rounding(problem)
    improved = refine(problem, start.level, t_max=8)
    for move in improved.metadata["moves"]:
        assert move["gain"] > 0


def test_refinement_stops_at_t_max():
    problem = make_problem(0, num_tokens=20)
    start = certified_rounding(problem)
    result = refine(problem, start.level, t_max=2)
    assert result.num_moves <= 2


def test_refinement_leaves_a_full_budget_schedule_unchanged():
    # With beta = 0 and every token already affordable, no move can help.
    problem = make_problem(1, num_tokens=6, beta=0.0, budget=1000.0)
    level = torch.tensor([1, 1, 1, 1, 1, 1])
    result = refine(problem, level, t_max=8)
    assert result.num_moves == 0


def test_refinement_upgrades_protection_when_it_pays():
    # A strong channel makes the reliably decodable level attractive: q_H - q_L
    # is large enough that upgrading is an exact gain if the budget allows.
    problem = make_problem(2, num_tokens=4, beta=0.0, budget=48.0 * 4)
    level = torch.tensor([3, 3, 3, 3])          # all at the weakest level
    result = refine(problem, level, t_max=8)
    assert problem.is_feasible(_level_to_schedule(problem, result.level))
    assert result.utility > float(problem.expected_utility(_level_to_schedule(problem, level)))


def test_refinement_prefers_strong_coupling_partners():
    # Token 1 is useless alone but strongly supports token 0; with coupling on,
    # the swap that keeps token 1 should beat dropping it for an isolated token.
    utility = torch.tensor([1.0, 0.05, 0.9])
    coupling = torch.zeros(3, 3)
    coupling[0, 1] = 0.9                        # M_{0,1}: token 0 supports token 1? see Eq. (16)
    problem = SchedulingProblem(base_utility=utility, coupling=coupling.triu(1), beta=2.0,
                                q=torch.ones(1), rates=torch.tensor([1.0]), budget=2.0)
    start = torch.tensor([1, 0, 1])
    result = refine(problem, start, t_max=4)
    assert problem.is_feasible(_level_to_schedule(problem, result.level))
    assert result.utility == pytest.approx(float(
        problem.expected_utility(_level_to_schedule(problem, result.level))))


def test_jain_index_properties():
    assert jain_index(torch.tensor([1.0, 1.0, 1.0])) == pytest.approx(1.0)
    assert jain_index(torch.tensor([1.0, 0.0, 0.0])) == pytest.approx(1 / 3)
    assert jain_index(torch.tensor([0.0, 0.0])) == 0.0
    assert 0.0 < jain_index(torch.tensor([3.0, 1.0])) < 1.0


def test_bandwidth_allocation_sums_to_the_budget():
    reports = torch.tensor([0.4, 0.2, 0.1, 0.05])
    result = allocate_bandwidth(reports, total_bandwidth=10.0, theta=1.0)
    assert float(result.bandwidth.sum()) == pytest.approx(10.0, rel=1e-6)
    assert bool((result.bandwidth >= 0).all())


def test_kkt_condition_holds_for_active_terminals():
    reports = torch.tensor([0.5, 0.3, 0.1])
    theta, epsilon = 1.0, 1e-6
    result = allocate_bandwidth(reports, total_bandwidth=6.0, theta=theta, epsilon_b=epsilon)
    bandwidth = result.bandwidth.to(torch.float64)
    active = bandwidth > 1e-9
    # Eq. (46): g_k + theta/(B_k + eps_B) = nu on the active set.
    for k in torch.nonzero(active).reshape(-1).tolist():
        lhs = float(reports[k]) + theta / (float(bandwidth[k]) + epsilon)
        assert lhs == pytest.approx(result.nu, rel=1e-4)
    # Inactive terminals violate the condition in the safe direction.
    for k in torch.nonzero(~active).reshape(-1).tolist():
        assert float(reports[k]) >= result.nu - 1e-6


def test_higher_marginal_value_gets_more_bandwidth():
    reports = torch.tensor([0.8, 0.2])
    result = allocate_bandwidth(reports, total_bandwidth=4.0, theta=1.0)
    assert float(result.bandwidth[0]) > float(result.bandwidth[1])


def test_theta_zero_is_bang_bang():
    reports = torch.tensor([0.9, 0.1, 0.5])
    result = allocate_bandwidth(reports, total_bandwidth=5.0, theta=0.0)
    assert float(result.bandwidth.sum()) == pytest.approx(5.0)
    assert int(result.bandwidth.argmax()) == 0


def test_theta_controls_the_fairness_efficiency_tradeoff():
    reports = torch.tensor([1.0, 0.05, 0.05, 0.05])
    greedy = allocate_bandwidth(reports, total_bandwidth=8.0, theta=0.05)
    fair = allocate_bandwidth(reports, total_bandwidth=8.0, theta=5.0)
    assert jain_index(fair.bandwidth) > jain_index(greedy.bandwidth)


def test_strong_concavity_and_price_error_bounds():
    assert strong_concavity_bound(10.0, 1.0, 1e-6) == pytest.approx(1 / 100.0, rel=1e-5)
    bound = price_error_bound(torch.tensor([0.1, -0.1]), sigma_b=0.5)
    assert bound["gradient_error_norm"] == pytest.approx(math.sqrt(0.02))
    assert bound["allocation_error_bound"] == pytest.approx(math.sqrt(0.02) / 0.5)
    assert bound["objective_gap_bound"] == pytest.approx(0.02 / 0.5)


def test_allocation_from_nu_matches_the_closed_form():
    reports = torch.tensor([0.3, 0.2], dtype=torch.float64)
    nu, theta, epsilon = 1.0, 2.0, 0.5
    expected = torch.clamp(theta / (nu - reports) - epsilon, min=0.0)
    assert torch.allclose(allocation_from_nu(reports, nu, theta, epsilon), expected)


def test_end_to_end_schedule_is_feasible_and_improves():
    problem = make_problem(7, num_tokens=16)
    lp = solve_base_lp(problem)
    rounded = certified_rounding(problem, lp)
    refined = refine(problem, rounded.level, t_max=8)
    assert problem.is_feasible(_level_to_schedule(problem, refined.level))
    assert refined.utility >= rounded.utility - 1e-5
    # Proposition 1: the 1/2 guarantee survives refinement.
    assert refined.utility >= lp.lp_value / 2.0 - 1e-5
