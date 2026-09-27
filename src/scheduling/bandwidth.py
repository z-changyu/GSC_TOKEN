"""Multi-terminal bandwidth allocation, Eq. (43)-(50), and the Jain index.

Each terminal reports one scalar, the marginal semantic value of bandwidth
``g_hat_k = omega_k T_s mu_k c_eps(h_hat_k)`` (Eq. 47). The coordinator keeps the
concave time-sharing closure of every terminal value function, so the KKT
conditions of the proportional-fair outer problem are sufficient rather than
merely heuristic, and the active terminals satisfy Eq. (46)::

    g_k + theta / (B_k + epsilon_B) = nu

with a closed form (Eq. 48) once ``nu`` is fixed. Because the left-hand side is
monotone in ``B_k`` and the allocations sum to ``B``, a one-dimensional
bisection on ``nu`` solves the slot.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class AllocationResult:
    bandwidth: torch.Tensor
    nu: float
    active: torch.Tensor
    iterations: int
    strong_concavity: float


def jain_index(values: torch.Tensor) -> float:
    """Jain fairness index of the per-terminal utilities.

    The paper uses it only as an evaluation metric; a Pareto claim requires
    sweeping ``theta`` rather than reporting one operating point.
    """
    values = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    if values.numel() == 0 or float(values.sum()) == 0.0:
        return 0.0
    return float(values.sum().pow(2) / (values.numel() * values.pow(2).sum()))


def allocation_from_nu(reports: torch.Tensor, nu: float, theta: float,
                       epsilon_b: float) -> torch.Tensor:
    """Eq. (48): closed form of the active allocation at a fixed multiplier."""
    reports = torch.as_tensor(reports, dtype=torch.float64)
    if theta <= 0:
        raise ValueError("Eq. (48) requires theta > 0")
    active = nu > reports
    share = torch.zeros_like(reports)
    share[active] = theta / (nu - reports[active]) - epsilon_b
    return torch.clamp(share, min=0.0)


def allocate_bandwidth(reports, total_bandwidth: float, theta: float = 1.0,
                       epsilon_b: float = 1e-6, iterations: int = 200,
                       tolerance: float = 1e-12) -> AllocationResult:
    """Solve Eq. (46) by bisection on ``nu`` so that the allocations sum to ``B``."""
    reports = torch.as_tensor(reports, dtype=torch.float64).reshape(-1)
    total = float(total_bandwidth)
    if total <= 0 or reports.numel() == 0:
        return AllocationResult(torch.zeros_like(reports), float("inf"),
                                torch.zeros_like(reports, dtype=torch.bool), 0, 0.0)

    if theta <= 0:
        # Without the proportional-fair term the objective is linear in B, so the
        # optimum is bang-bang: everything goes to the strongest report.
        bandwidth = torch.zeros_like(reports)
        bandwidth[int(reports.argmax())] = total
        return AllocationResult(bandwidth, float(reports.max()), bandwidth > 0, 0, 0.0)

    def total_at(nu: float) -> float:
        return float(allocation_from_nu(reports, nu, theta, epsilon_b).sum())

    lower = float(reports.max()) + theta / (total + reports.numel() * epsilon_b + 1.0)
    upper = float(reports.max()) + theta / epsilon_b
    while total_at(upper) > total and upper < 1e18:
        upper *= 2.0
    step = 0
    for step in range(1, max(1, int(iterations)) + 1):
        middle = 0.5 * (lower + upper)
        if total_at(middle) > total:
            lower = middle
        else:
            upper = middle
        if upper - lower <= max(tolerance, 1e-15 * max(1.0, abs(upper))):
            break
    nu = 0.5 * (lower + upper)
    bandwidth = allocation_from_nu(reports, nu, theta, epsilon_b)

    # The bisection hits the sum only up to the interval width, so rescale the
    # active terminals to consume the budget exactly.
    assigned = float(bandwidth.sum())
    if assigned > 0 and abs(assigned - total) > 1e-9:
        bandwidth = bandwidth * (total / assigned)

    return AllocationResult(bandwidth=bandwidth.to(torch.float32), nu=nu,
                            active=bandwidth > 0, iterations=step,
                            strong_concavity=theta / (total + epsilon_b) ** 2)


def strong_concavity_bound(total_bandwidth: float, theta: float,
                           epsilon_b: float = 1e-6) -> float:
    """Eq. (49): ``sigma_B >= theta / (B + epsilon_B)^2``."""
    return theta / (float(total_bandwidth) + float(epsilon_b)) ** 2


def price_error_bound(report_error: torch.Tensor, sigma_b: float) -> dict[str, float]:
    """Eq. (50): allocation and objective gaps induced by an inexact price report."""
    error = torch.as_tensor(report_error, dtype=torch.float64).reshape(-1)
    norm = float(error.norm())
    return {"gradient_error_norm": norm, "strong_concavity": float(sigma_b),
            "allocation_error_bound": (norm / sigma_b) if sigma_b > 0 else math.inf,
            "objective_gap_bound": (norm ** 2 / sigma_b) if sigma_b > 0 else math.inf}
