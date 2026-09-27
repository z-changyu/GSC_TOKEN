"""The reliability-aware joint selection and scheduling problem of Section IV.

Decision variables are ``y_{i,ell,tau}``: token ``i`` is transmitted with
protection mode ``ell``, at most one mode per token (Eq. 9). Everything the
scheduler needs is collected in :class:`SchedulingProblem`; the solvers in the
sibling modules are pure functions of it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class SchedulingProblem:
    """One terminal, one slot."""

    base_utility: torch.Tensor        # u'_i,                 [N]
    coupling: torch.Tensor            # M_ij, strictly upper, [N, N]
    beta: float                       # coupling strength
    q: torch.Tensor                   # q_{i,ell},            [N, |R|]
    rates: torch.Tensor               # r_ell,                [|R|]
    budget: float                     # C_tau

    def __post_init__(self) -> None:
        self.base_utility = torch.as_tensor(self.base_utility, dtype=torch.float32).reshape(-1)
        self.rates = torch.as_tensor(self.rates, dtype=torch.float32).reshape(-1)
        q = torch.as_tensor(self.q, dtype=torch.float32)
        if q.dim() == 1:
            # A mode's success probability depends on the token only through the
            # mode, because tokens sharing a slot see the same link.
            q = q.reshape(1, -1).expand(self.num_tokens, -1)
        self.q = q
        if self.q.shape != (self.num_tokens, self.num_levels):
            raise ValueError(f"q must be [N, |R|], got {tuple(self.q.shape)}")
        if self.coupling.shape != (self.num_tokens, self.num_tokens):
            raise ValueError("coupling must be [N, N]")
        if self.beta < 0:
            raise ValueError("beta must be nonnegative")
        if float(self.budget) < 0:
            raise ValueError("budget must be nonnegative")
        if bool((self.rates <= 0).any()):
            raise ValueError("protection costs must be positive")
        if bool((self.q < 0).any() or (self.q > 1).any()):
            raise ValueError("decoding probabilities must lie in [0, 1]")

    @property
    def num_tokens(self) -> int:
        return int(self.base_utility.numel())

    @property
    def num_levels(self) -> int:
        return int(self.rates.numel())

    @property
    def mode_value(self) -> torch.Tensor:
        """``a_{i,ell} = u'_i q_{i,ell}``, the additive weight of Eq. (34)."""
        return self.base_utility[:, None] * self.q

    def reception_degree(self, y: torch.Tensor) -> torch.Tensor:
        """Eq. (33): ``z_{i,tau} = sum_ell y_{i,ell,tau} q_{i,ell,tau}``."""
        return (y * self.q).sum(dim=-1)

    def resource_use(self, y: torch.Tensor) -> torch.Tensor:
        """Eq. (35): ``sum_{i,ell} y_{i,ell} r_ell``.

        Accumulated in float64: with costs of order 10^2 and budgets of order
        10^3, a float32 sum drifts by roughly 10^-5 and would report a feasible
        schedule as infeasible.
        """
        return (y * self.rates).sum(dtype=torch.float64).to(torch.float32)

    def base_expected_utility(self, y: torch.Tensor) -> torch.Tensor:
        """``F_base`` of Eq. (63): the coupling-free part of Eq. (34)."""
        return (self.mode_value * y).sum(dtype=torch.float64).to(torch.float32)

    def expected_utility(self, y: torch.Tensor) -> torch.Tensor:
        """Eq. (34): expected semantic utility of a (fractional) schedule."""
        z = self.reception_degree(y)
        weighted = self.base_utility * z
        coupling = self.beta * (self.coupling * torch.outer(weighted, weighted)).sum()
        return (self.base_utility * z).sum() + coupling

    def is_feasible(self, y: torch.Tensor, tolerance: float = 1e-5) -> bool:
        if float(self.resource_use(y)) > float(self.budget) + tolerance:
            return False
        if bool((y.sum(dim=-1) > 1.0 + tolerance).any()):
            return False
        return bool((y >= -tolerance).all())


@dataclass
class ScheduleResult:
    """A feasible integral schedule plus the audit trail of how it was built."""

    y: torch.Tensor
    level: torch.Tensor              # 0 = not transmitted, else 1..|R|
    utility: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def selected(self) -> torch.Tensor:
        return self.level > 0

    def as_dict(self) -> dict[str, Any]:
        return {"utility": self.utility, **self.metadata}
