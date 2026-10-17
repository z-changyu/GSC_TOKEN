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
    # P(both tokens received) for a pair of protection levels, [|R|, |R|]. The whole
    # report shares one instantaneous channel, so the joint probability is NOT the
    # product of the marginals (see channel.link.joint_decoding_probability). When the
    # caller does not supply it we fall back to the independent-pair product, and
    # ``joint_model`` records which of the two the schedule was optimised under.
    joint_q: torch.Tensor | None = None

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
        explicit_joint = self.joint_q is not None
        if not explicit_joint:
            marginal = self.q[0]
            self.joint_q = marginal[:, None] * marginal[None, :]
        else:
            self.joint_q = torch.as_tensor(self.joint_q, dtype=torch.float32)
            if self.joint_q.shape != (self.num_levels, self.num_levels):
                raise ValueError(f"joint_q must be [|R|, |R|], got {tuple(self.joint_q.shape)}")
            if bool((self.joint_q < 0).any() or (self.joint_q > 1).any()):
                raise ValueError("joint decoding probabilities must lie in [0, 1]")
        self.joint_model = "common_channel" if explicit_joint else "independent"

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

    def coupling_expected_value(self, y: torch.Tensor) -> torch.Tensor:
        """Eq. (34) coupling term: ``beta * sum_{i<j} M_ij u'_i u'_j P(both received)``.

        ``P(both) = sum_{ell,m} y_{i,ell} y_{j,m} Q_{ell,m}`` with ``Q`` the joint
        decoding probability. For an integral (or fractional) ``y`` this is
        ``y Q y^T``; when ``Q`` is the independent product it collapses to
        ``z_i z_j``, which is exactly the historical behaviour.
        """
        pair = y @ self.joint_q @ y.t()
        weight = self.coupling * torch.outer(self.base_utility, self.base_utility)
        return self.beta * (weight * pair).sum()

    def coupling_affinity(self, level: torch.Tensor,
                          incident: torch.Tensor) -> torch.Tensor:
        """``[N, |R|]``: ``beta * u'_i * sum_{j in T} M^<->_ij u'_j Q_{ell, lvl_j}``.

        This is the exact marginal of the coupling term with respect to *adding*
        token ``i`` at level ``ell``, given the currently selected set ``T`` (an
        integral schedule). Evaluating it at ``(i, lvl_i)`` also gives the coupling
        part of removing ``i``, which keeps insertion / removal / mode-change gains
        on one formula.
        """
        selected = level > 0
        index = (level - 1).clamp(min=0)
        partner = self.base_utility * selected.float()
        rows = incident * partner.reshape(1, -1)              # M^<->_ij u'_j
        return (self.beta * (rows @ self.joint_q[:, index].t())
                * self.base_utility.reshape(-1, 1))

    def expected_utility(self, y: torch.Tensor) -> torch.Tensor:
        """Eq. (34): expected semantic utility of a (fractional) schedule."""
        z = self.reception_degree(y)
        return (self.base_utility * z).sum() + self.coupling_expected_value(y)

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
