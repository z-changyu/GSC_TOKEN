"""Unit-row coupling normalisation.

Raw Eq. (16) coefficients are diffuse: a row sums to 1.5-2.9 while a single
coefficient is ~0.01, so Eq. (62)'s ``beta < 2 / d_M`` pins the reachable pair term
three orders of magnitude below any measurable effect. Unit-row scaling sets
``d_M = 1`` and frees the whole ``beta in [0, 2)`` range. These tests pin the two
properties the rest of the code relies on: the output is *symmetric* (because
``incident_coupling`` rebuilds ``M^{<->}`` from it) and its maximum incident row sum
is exactly one.
"""
from __future__ import annotations

import torch

from src.pragmatic.coupling import (NORMALIZATIONS, incident_coupling,
                                    normalize_coupling)
from src.pragmatic.utility import coupling_regime, set_utility


def make_coupling(seed: int = 0, n: int = 40) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    raw = torch.rand(n, n, generator=generator) ** 3          # diffuse, uneven rows
    return raw.triu(1)


def test_none_is_the_identity():
    coupling = make_coupling()
    assert torch.equal(normalize_coupling(coupling, "none"), coupling)
    assert set(NORMALIZATIONS) == {"none", "row"}


def test_row_normalisation_gives_unit_row_sums_and_stays_symmetric():
    for seed in range(4):
        coupling = make_coupling(seed)
        normalized = normalize_coupling(coupling, "row")
        assert torch.allclose(normalized, normalized.triu(1)), "must stay strictly upper"
        incident = incident_coupling(normalized)
        assert torch.allclose(incident, incident.t(), atol=1e-6), "incident must be symmetric"
        assert float(incident.sum(dim=1).max()) == 1.0 or abs(
            float(incident.sum(dim=1).max()) - 1.0) < 1e-4
        assert coupling_regime(normalized, 0.5)["d_M"] == 1.0 or abs(
            coupling_regime(normalized, 0.5)["d_M"] - 1.0) < 1e-4


def test_row_normalisation_keeps_beta_two_inside_the_weak_coupling_regime():
    coupling = make_coupling(1)
    normalized = normalize_coupling(coupling, "row")
    assert coupling_regime(normalized, 1.0)["weak_coupling"]
    assert coupling_regime(normalized, 1.9)["weak_coupling"]
    assert coupling_regime(normalized, 2.1)["weak_coupling"] is False


def test_unit_rows_make_the_coupling_mass_a_weighted_average_of_the_utilities():
    """The point of the normalisation: the coupling mass is bounded by max(u').

    Raw rows sum to 1.5-2.9, so the mass scales with an unbounded row sum; with unit
    rows it is a convex combination of the per-token utilities, which is exactly the
    scale at which the additive term lives.
    """
    n = 40
    coupling = normalize_coupling(make_coupling(2, n), "row")
    incident = incident_coupling(coupling)
    uniform = torch.ones(n)
    assert abs(float((incident @ uniform).median()) - 1.0) < 1e-4

    base = torch.rand(n, generator=torch.Generator().manual_seed(9))
    mass = incident @ base
    assert float(mass.min()) >= float(base.min()) - 1e-6
    assert float(mass.max()) <= float(base.max()) + 1e-6
    assert set_utility(torch.arange(5), base, coupling, 1.0) > 0


def test_set_utility_rejects_beta_above_the_normalised_cap():
    coupling = normalize_coupling(make_coupling(3), "row")
    base = torch.rand(coupling.shape[0], generator=torch.Generator().manual_seed(4))
    import pytest

    with pytest.raises(ValueError):
        set_utility(torch.arange(5), base, coupling, 2.5)
