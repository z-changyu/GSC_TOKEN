"""Protection codebook and conditional decoding probability.

The paper fixes the three equivalent-bit costs to ``r_H/r_M/r_L = 48/32/16``
per token and leaves the decoding probabilities to be "calibrated from the same
link-level channel and packet-loss simulator used for evaluation". This module
makes that calibration explicit.

The link model is:

* a token carries ``b_tok`` equivalent information bits;
* protection level ``ell`` transmits it at code rate ``rho_ell = b_tok / r_ell``
  bits per channel use, so the strongest level with cost 48 runs at rate 1/3 and
  the weakest with cost 16 at rate 1;
* the token decodes iff the instantaneous rate clears the code rate,
  ``log2(1 + gamma_bar |h|^2) >= rho_ell``;
* packet loss is an independent erasure on the decoded token.

Hence ``q_{i,ell,tau} = (1 - Pr[log2(1 + gamma_bar |h|^2) < rho_ell | h_hat]) (1 - p_loss)``.
It depends on the token index only through the protection level, because tokens
sharing a slot see the same link.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .reliable_capacity import gain_power_cdf, posterior_gain_parameters


@dataclass(frozen=True)
class ProtectionCodebook:
    """The finite set ``R`` of token-protection modes, Eq. (9)."""

    levels: tuple[float, ...] = (48.0, 32.0, 16.0)      # r_H, r_M, r_L
    information_bits_per_token: float = 16.0             # b_tok

    def __post_init__(self) -> None:
        if len(self.levels) == 0:
            raise ValueError("at least one protection level is required")
        if any(level <= 0 for level in self.levels):
            raise ValueError("protection costs must be positive")
        if self.information_bits_per_token <= 0:
            raise ValueError("information_bits_per_token must be positive")

    @property
    def num_levels(self) -> int:
        return len(self.levels)

    @property
    def code_rates(self) -> tuple[float, ...]:
        """``rho_ell = b_tok / r_ell`` bits per channel use."""
        return tuple(self.information_bits_per_token / level for level in self.levels)

    def rate_threshold(self, level_index: int, snr_linear: float) -> float:
        """``|h|^2`` needed for level ``level_index`` to decode."""
        rate = self.code_rates[level_index]
        if math.isinf(snr_linear):
            return 0.0
        return (2.0 ** rate - 1.0) / snr_linear


def decoding_probability(state, codebook: ProtectionCodebook) -> torch.Tensor:
    """``q_{i,ell,tau}`` for every protection level, shape ``[..., |R|]``."""
    mu, nu = posterior_gain_parameters(state.h_hat, state.csi_error_variance)
    levels = []
    for level_index in range(codebook.num_levels):
        threshold = codebook.rate_threshold(level_index, state.snr_linear)
        outage = gain_power_cdf(mu, nu, torch.as_tensor(threshold, dtype=torch.float32))
        levels.append(1.0 - outage)
    stacked = torch.stack(levels, dim=-1)
    return stacked * (1.0 - state.packet_loss_rate)


def reception_degree(selection_mask, level_index, q) -> torch.Tensor:
    """Eq. (33): ``z_{i,tau} = sum_ell y_{i,ell,tau} q_{i,ell,tau}``.

    ``selection_mask`` marks the tokens that carry a mode and ``level_index``
    gives the chosen mode per token; unselected tokens receive ``z = 0``.
    """
    selected = torch.as_tensor(selection_mask, dtype=torch.bool)
    index = torch.as_tensor(level_index, dtype=torch.long)
    per_token = torch.zeros(selected.shape, dtype=q.dtype)
    per_token[selected] = q[index[selected]]
    return per_token
