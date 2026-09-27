"""Reliable spectral efficiency and reliable budget, Eq. (5)-(8).

Section II-B refuses to insert the CSI estimate straight into the perfect-CSI
Shannon expression, because that is optimistic under estimation error. Instead
the budget is the largest rate whose *conditional* outage stays below
``epsilon_out``::

    c_eps(h_hat) = sup{ c : Pr[ log2(1 + gamma_bar |h|^2) >= c | h_hat ] >= 1 - epsilon_out }

With ``h ~ CN(0, 1)``, ``h_hat = h + e`` and ``e ~ CN(0, sigma_e^2)``, the
posterior is ``h | h_hat ~ CN(h_hat/(1+sigma_e^2), sigma_e^2/(1+sigma_e^2))``,
so ``|h|`` is Rician and the required quantile is available in closed form.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.stats import rice

# Below this conditional variance the CSI is treated as perfect.
PERFECT_CSI_VARIANCE = 1e-12


def posterior_gain_parameters(h_hat: torch.Tensor,
                              csi_error_variance: float) -> tuple[torch.Tensor, float]:
    """MMSE posterior of ``h`` given ``h_hat``: ``h | h_hat ~ CN(mu, nu)``."""
    sigma = float(csi_error_variance)
    denominator = 1.0 + sigma
    mu = torch.as_tensor(h_hat, dtype=torch.complex64) / denominator
    nu = sigma / denominator
    return mu, nu


def gain_power_quantile(mu: torch.Tensor, nu: float, epsilon: float) -> torch.Tensor:
    """``epsilon``-quantile of ``|h|^2`` under ``h ~ CN(mu, nu)``.

    ``|h|`` is Rician with ``b = |mu| / sigma`` and ``sigma = sqrt(nu/2)``.
    """
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must lie in (0, 1)")
    if nu <= PERFECT_CSI_VARIANCE:
        return mu.abs().pow(2)
    scale = math.sqrt(nu / 2.0)
    mu_abs = np.asarray(mu.abs(), dtype=np.float64)
    levels = rice.ppf(epsilon, np.asarray(mu_abs / scale, dtype=np.float64), scale=scale)
    return torch.as_tensor(np.square(levels), dtype=torch.float32)


def gain_power_cdf(mu: torch.Tensor, nu: float, threshold) -> torch.Tensor:
    """``Pr[|h|^2 < threshold]`` under ``h ~ CN(mu, nu)``."""
    threshold = torch.as_tensor(threshold, dtype=torch.float32)
    if nu <= PERFECT_CSI_VARIANCE:
        return (mu.abs().pow(2) < threshold).to(torch.float32)
    scale = math.sqrt(nu / 2.0)
    mu_abs = np.asarray(mu.abs(), dtype=np.float64)
    root = np.sqrt(np.asarray(threshold, dtype=np.float64))
    values = rice.cdf(root, np.asarray(mu_abs / scale, dtype=np.float64), scale=scale)
    return torch.as_tensor(values, dtype=torch.float32)


def reliable_spectral_efficiency(state, epsilon_out: float) -> torch.Tensor:
    """Eq. (5): conditional reliable spectral efficiency, one value per terminal."""
    mu, nu = posterior_gain_parameters(state.h_hat, state.csi_error_variance)
    quantile = gain_power_quantile(mu, nu, epsilon_out)
    if math.isinf(state.snr_linear):
        return torch.full_like(quantile, float("inf"))
    return torch.log2(1.0 + float(state.snr_linear) * quantile)


def reliable_budget(state, epsilon_out: float, bandwidth: float = 1.0,
                    slot_duration: float = 1.0) -> torch.Tensor:
    """Eq. (6): ``C_tau = T_s B c_eps(h_hat)``, in equivalent bits."""
    return float(slot_duration) * float(bandwidth) * reliable_spectral_efficiency(state, epsilon_out)


def robust_reliable_spectral_efficiency(state, epsilon_out: float, num_calibration: int = 1024,
                                        delta_amb: float = 0.05,
                                        generator: torch.Generator | None = None) -> torch.Tensor:
    """Eq. (7)/(8): distributionally robust reliable spectral efficiency.

    The ambiguity set is handled through the statistic the paper suggests, an
    empirical lower-confidence quantile: with ``N`` draws from the conditional
    law, the one-sided ``(1 - delta_amb)`` lower confidence bound for the
    ``epsilon_out``-quantile is the order statistic of rank
    ``floor(N*eps - z*sqrt(N*eps*(1-eps)))``. As ``N`` grows the rank fraction
    tends to ``epsilon_out`` and this returns to Eq. (5), so the robust value is
    never more optimistic than the nominal one.
    """
    if num_calibration < 2:
        raise ValueError("num_calibration must be at least two")
    if not 0.0 < delta_amb < 1.0:
        raise ValueError("delta_amb must lie in (0, 1)")
    mu, nu = posterior_gain_parameters(state.h_hat, state.csi_error_variance)
    if nu <= PERFECT_CSI_VARIANCE:
        return reliable_spectral_efficiency(state, epsilon_out)

    scale = math.sqrt(nu / 2.0)
    mu_abs = mu.abs().reshape(-1)
    num = mu_abs.numel()
    shape = (num_calibration, num)
    real = torch.randn(shape, generator=generator, dtype=torch.float32) * scale
    imag = torch.randn(shape, generator=generator, dtype=torch.float32) * scale
    power = (real + mu_abs).pow(2) + imag.pow(2)

    z = _normal_quantile(1.0 - delta_amb)
    rank = int(math.floor(num_calibration * epsilon_out
                          - z * math.sqrt(num_calibration * epsilon_out * (1.0 - epsilon_out))))
    rank = max(0, min(num_calibration - 1, rank))
    ordered, _ = torch.sort(power, dim=0)
    quantile = ordered[rank]
    if math.isinf(state.snr_linear):
        return torch.full_like(quantile, float("inf"))
    return torch.log2(1.0 + float(state.snr_linear) * quantile)


def _normal_quantile(probability: float) -> float:
    """Standard normal quantile, used only for the order-statistic rank."""
    from scipy.stats import norm
    return float(norm.ppf(probability))
