"""Joint-reception semantics.

The whole report shares one instantaneous channel, so a *pair* of tokens survives
with probability ``Q[lvl_i, lvl_j]`` -- both decode iff the realised rate clears the
weaker of the two levels -- and NOT with the product of the marginals ``q_i q_j``,
which would price two independent fading draws. These tests pin the corrected
arithmetic, and pin the fact that omitting the joint matrix reproduces the historical
product model bit for bit.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.channel.link import (ProtectionCodebook, decoding_probability,
                              joint_decoding_probability)
from src.pragmatic.coupling import incident_coupling
from src.scheduling.local_refinement import refine
from src.scheduling.problem import SchedulingProblem

CODEBOOK = ProtectionCodebook(levels=(48.0, 32.0, 16.0), information_bits_per_token=16.0)


def make_state(snr_db: float = 15.0, csi_error_variance: float = 0.01,
               packet_loss_rate: float = 0.05) -> SimpleNamespace:
    return SimpleNamespace(h_hat=torch.tensor([1.0 + 0.0j]),
                           snr_linear=10.0 ** (snr_db / 10.0),
                           csi_error_variance=csi_error_variance,
                           packet_loss_rate=packet_loss_rate)


def make_problem(seed: int = 0, num_tokens: int = 5, beta: float = 0.3,
                 joint: bool = True, budget: float | None = None) -> SchedulingProblem:
    generator = torch.Generator().manual_seed(seed)
    utility = torch.rand(num_tokens, generator=generator)
    coupling = torch.rand(num_tokens, num_tokens, generator=generator).triu(1)
    q = 0.5 + 0.5 * torch.rand(CODEBOOK.num_levels, generator=generator)
    rates = torch.tensor([48.0, 32.0, 16.0])
    joint_q = None
    if joint:
        # min-structure, exactly what a common channel implies: Q_ij = min(a_i, a_j).
        joint_q = torch.minimum(q[:, None], q[None, :])
    if budget is None:
        budget = float(rates.sum()) * num_tokens * 0.35
    return SchedulingProblem(base_utility=utility, coupling=coupling, beta=beta, q=q,
                             rates=rates, budget=budget, joint_q=joint_q)


def schedule_of(problem: SchedulingProblem, level: torch.Tensor) -> torch.Tensor:
    """One-hot [N, |R|] schedule from a per-token level (0 = not transmitted)."""
    y = torch.zeros(problem.num_tokens, problem.num_levels)
    selected = level > 0
    y[torch.arange(problem.num_tokens)[selected], level[selected] - 1] = 1.0
    return y


# --------------------------------------------------------------------- link level
def test_joint_probability_is_the_weaker_level_times_independent_losses():
    levels = CODEBOOK.num_levels
    state = make_state()
    lossless = make_state(packet_loss_rate=0.0)
    geometric = decoding_probability(lossless, CODEBOOK).reshape(-1)
    joint = joint_decoding_probability(state, CODEBOOK).reshape(levels, levels)
    expected = torch.minimum(geometric[:, None], geometric[None, :]) * (1 - 0.05) ** 2
    assert torch.allclose(joint, expected, atol=1e-6)


def test_joint_probability_never_undercuts_the_independent_product():
    for snr_db in (0.0, 15.0):
        for packet_loss in (0.0, 0.1):
            state = make_state(snr_db=snr_db, packet_loss_rate=packet_loss)
            q = decoding_probability(state, CODEBOOK).reshape(-1)
            joint = joint_decoding_probability(state, CODEBOOK)
            # A shared fading draw makes the pair *at least* as reliable as two
            # independent draws; q_i q_j is therefore a strict under-estimate.
            assert bool((joint >= torch.outer(q, q) - 1e-7).all())


# ---------------------------------------------------------------- problem level
def test_missing_joint_matrix_reproduces_the_product_model():
    problem = make_problem(joint=False)
    assert problem.joint_model == "independent"
    y = torch.zeros(problem.num_tokens, problem.num_levels)
    y[:, 1] = 1.0
    z = problem.reception_degree(y)
    weighted = problem.base_utility * z
    historical = (weighted.sum()
                  + problem.beta * (problem.coupling * torch.outer(weighted, weighted)).sum())
    assert float(problem.expected_utility(y)) == pytest.approx(float(historical), rel=1e-6)


def test_joint_matrix_switches_the_model_and_raises_the_coupling_term():
    independent = make_problem(joint=False)
    joint = make_problem(joint=True)
    assert joint.joint_model == "common_channel"
    y = torch.zeros(joint.num_tokens, joint.num_levels)
    y[:, 0] = 1.0
    assert float(joint.coupling_expected_value(y)) > float(
        independent.coupling_expected_value(y))


def test_coupling_affinity_matches_the_brute_force_marginal():
    problem = make_problem(seed=3, num_tokens=4, joint=True)
    incident = incident_coupling(problem.coupling)
    level = torch.tensor([1, 0, 3, 2])                    # token 1 is not transmitted

    base = float(problem.expected_utility(schedule_of(problem, level)))
    add = problem.base_utility.reshape(-1, 1) * problem.q
    affinity = problem.coupling_affinity(level, incident)
    for token in (1,):
        for lvl in range(problem.num_levels):
            trial = level.clone()
            trial[token] = lvl + 1
            gain = float(problem.expected_utility(schedule_of(problem, trial))) - base
            assert gain == pytest.approx(float(add[token, lvl] + affinity[token, lvl]),
                                         abs=1e-5)


def test_refinement_never_decreases_the_joint_objective():
    problem = make_problem(seed=5, num_tokens=5, joint=True)
    level = torch.zeros(problem.num_tokens, dtype=torch.long)
    level[:2] = 3
    before = float(problem.expected_utility(schedule_of(problem, level)))
    result = refine(problem, level, t_max=6)
    assert result.utility >= before - 1e-6
    assert problem.is_feasible(schedule_of(problem, result.level))
