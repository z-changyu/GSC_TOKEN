"""Method dispatch tests, including the ladder used by the ablation study."""
from __future__ import annotations

import random

import pytest
import torch

from src.scheduling import SCHEDULING_METHODS, select_level
from src.scheduling.problem import SchedulingProblem

RATES = torch.tensor([48.0, 32.0, 16.0])


def make_inputs(seed: int = 0, num_tokens: int = 12):
    generator = torch.Generator().manual_seed(seed)
    utility = torch.rand(num_tokens, generator=generator)
    coupling = torch.zeros(num_tokens, num_tokens)
    q = torch.tensor([0.9, 0.7, 0.4])
    return utility, coupling, q


def symbol_cost(level: torch.Tensor) -> float:
    return float(RATES[(level - 1).clamp(min=0)][level > 0].sum())


def test_unknown_method_is_rejected():
    utility, coupling, q = make_inputs()
    with pytest.raises(ValueError):
        select_level("not_a_method", utility, coupling, 0.25, q, RATES, 500.0)


def test_full_transmits_every_token_and_ignores_the_budget():
    utility, coupling, q = make_inputs()
    level = select_level("full", utility, coupling, 0.25, q, RATES, 0.0)
    assert int((level > 0).sum()) == utility.numel()


@pytest.mark.parametrize("budget", [48.0 * 3, 48.0 * 5 + 1, 1000.0])
def test_greedy_methods_respect_the_budget(budget):
    utility, coupling, q = make_inputs()
    for method in ("random", "fixed_H"):
        level = select_level(method, utility, coupling, 0.25, q, RATES, budget)
        assert symbol_cost(level) <= budget + 1e-6
        assert int((level > 0).sum()) == min(utility.numel(), int(budget // 48.0))


def test_fixed_H_keeps_the_highest_utility_tokens():
    utility, coupling, q = make_inputs()
    level = select_level("fixed_H", utility, coupling, 0.25, q, RATES, 48.0 * 4)
    kept = sorted(level.nonzero().reshape(-1).tolist())
    expected = sorted(torch.topk(utility, 4).indices.tolist())
    assert kept == expected


def test_random_is_seed_dependent_but_reproducible():
    utility, coupling, q = make_inputs()
    first = select_level("random", utility, coupling, 0.25, q, RATES, 48.0 * 4,
                         rng=random.Random(7))
    again = select_level("random", utility, coupling, 0.25, q, RATES, 48.0 * 4,
                         rng=random.Random(7))
    other = select_level("random", utility, coupling, 0.25, q, RATES, 48.0 * 4,
                         rng=random.Random(8))
    assert torch.equal(first, again)
    assert int((first > 0).sum()) == int((other > 0).sum())


@pytest.mark.parametrize("budget", [48.0 * 2, 48.0 * 4, 16.0 * 12 + 32.0 * 6])
def test_proposed_is_feasible_and_no_worse_than_fixed_H(budget):
    utility, coupling, q = make_inputs()
    level = select_level("proposed", utility, coupling, 0.25, q, RATES, budget)
    assert symbol_cost(level) <= budget + 1e-4

    def value(chosen):
        problem = SchedulingProblem(base_utility=utility, coupling=coupling, beta=0.25,
                                    q=q, rates=RATES, budget=budget)
        y = torch.zeros(utility.numel(), RATES.numel())
        rows = chosen > 0
        if bool(rows.any()):
            y[rows, chosen[rows] - 1] = 1.0
        return float(problem.expected_utility(y))

    assert value(level) >= value(select_level("fixed_H", utility, coupling, 0.25, q,
                                              RATES, budget)) - 1e-6


def test_zero_budget_transmits_nothing():
    utility, coupling, q = make_inputs()
    for method in ("random", "fixed_H", "proposed"):
        level = select_level(method, utility, coupling, 0.25, q, RATES, 0.0)
        assert int((level > 0).sum()) == 0


def test_differentiated_protection_appears_only_in_proposed():
    # A tight budget forces the scheduler to spend fewer equivalent bits on
    # low-utility tokens instead of dropping them, which is exactly what
    # differentiated protection buys.
    utility, coupling, q = make_inputs(num_tokens=20)
    budget = 48.0 * 5
    level = select_level("proposed", utility, coupling, 0.0, q, RATES, budget)
    assert symbol_cost(level) <= budget + 1e-4
    assert int((level > 0).sum()) > 5      # more tokens than the strongest level affords


def test_all_declared_methods_are_callable():
    utility, coupling, q = make_inputs()
    alt = torch.rand(utility.numel())          # stand-in LM self-information
    for method in SCHEDULING_METHODS:
        level = select_level(method, utility, coupling, 0.25, q, RATES, 400.0,
                             alt_score=alt)
        assert level.shape == utility.shape
        assert int(level.min()) >= 0
        assert int(level.max()) <= RATES.numel()


@pytest.mark.parametrize("budget", [48.0 * 3, 48.0 * 5 + 1, 1000.0])
def test_selection_baselines_respect_the_budget(budget):
    utility, coupling, q = make_inputs()
    alt = torch.rand(utility.numel())
    for method in ("truncate", "selective_context", "llmlingua", "longllmlingua"):
        level = select_level(method, utility, coupling, 0.25, q, RATES, budget,
                             alt_score=alt)
        assert symbol_cost(level) <= budget + 1e-6


def test_selective_context_keeps_the_most_surprising_tokens():
    utility, coupling, q = make_inputs()
    alt = torch.rand(utility.numel())
    level = select_level("selective_context", utility, coupling, 0.25, q, RATES,
                         48.0 * 4, alt_score=alt)
    kept = sorted(level.nonzero().reshape(-1).tolist())
    expected = sorted(torch.topk(alt, 4).indices.tolist())
    assert kept == expected


def test_lm_baselines_require_the_self_information_score():
    utility, coupling, q = make_inputs()
    for method in ("selective_context", "llmlingua", "longllmlingua"):
        with pytest.raises(ValueError):
            select_level(method, utility, coupling, 0.25, q, RATES, 400.0)


def test_adaptive_sc_uses_one_protection_level_and_fits_the_budget():
    utility, coupling, q = make_inputs()
    level = select_level("adaptive_sc", utility, coupling, 0.25, q, RATES, 400.0)
    chosen = level[level > 0]
    assert int(chosen.numel()) > 0
    assert int(chosen.unique().numel()) == 1          # equal protection for all
    assert symbol_cost(level) <= 400.0 + 1e-6


def test_proposed_does_not_burn_budget_on_uninformative_upgrades():
    # When every mode decodes equally well, moving a token to a costlier mode
    # cannot improve Eq. (34); the schedule must keep the cheapest tied mode
    # instead of walking the resource use up to the budget.
    utility, coupling, _ = make_inputs(num_tokens=12)
    uniform_q = torch.tensor([0.9, 0.9, 0.9])
    budget = 16.0 * 12 + 200.0
    level = select_level("proposed", utility, coupling, 0.25, uniform_q, RATES, budget)
    assert int((level > 0).sum()) == 12
    assert symbol_cost(level) <= 16.0 * 12 + 1e-4
