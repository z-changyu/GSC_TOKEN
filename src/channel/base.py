"""Physical channel models for the time-varying uplink of Section II-B.

A channel object owns three things:

* the stochastic law of the complex baseband gain ``h``;
* the imperfect-CSI model of Eq. (4), ``h_hat = h + e`` with ``e ~ CN(0, sigma_e^2)``;
* the packet-loss rate, which the paper folds into ``q_{i,ell,tau}`` rather than
  modelling at the transport layer.

Everything downstream (reliable capacity, conditional decoding probability)
is a pure function of the sampled :class:`ChannelState`, so the scheduler never
touches the fading law directly.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class ChannelState:
    """One channel realisation per terminal and slot."""

    h: torch.Tensor                     # complex true gain, E|h|^2 = 1
    h_hat: torch.Tensor                 # complex estimate of Eq. (4)
    snr_linear: float                   # gamma-bar
    csi_error_variance: float           # sigma_e^2
    packet_loss_rate: float
    channel: str = "abstract"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.h.shape)

    def flatten(self) -> "ChannelState":
        """Return a state whose gains are one-dimensional, for scalar utilities."""
        return ChannelState(self.h.reshape(-1), self.h_hat.reshape(-1), self.snr_linear,
                            self.csi_error_variance, self.packet_loss_rate, self.channel,
                            dict(self.metadata))


class Channel(ABC):
    """Stochastic physical channel.

    ``transmit`` is retained from the ideal-channel prototype so that a payload
    can still be pushed through the model end to end; for the fading channels it
    applies the token-level erasure pattern implied by ``q_{i,ell,tau}``.
    """

    name = "abstract"

    def __init__(self, *, snr_db: float, csi_error_variance: float = 0.0,
                 packet_loss_rate: float = 0.0, seed: int = 0,
                 csi_age_s: float = 0.0, doppler_hz: float = 0.0) -> None:
        if csi_error_variance < 0:
            raise ValueError("csi_error_variance must be nonnegative")
        if not 0 <= packet_loss_rate < 1:
            raise ValueError("packet_loss_rate must lie in [0, 1)")
        if csi_age_s < 0 or doppler_hz < 0:
            raise ValueError("csi_age_s and doppler_hz must be nonnegative")
        self.snr_db = float(snr_db)
        self.snr_linear = 10.0 ** (float(snr_db) / 10.0)
        self.csi_error_variance = float(csi_error_variance)
        self.packet_loss_rate = float(packet_loss_rate)
        self.seed = int(seed)
        # CSI aging: age [s] between the channel estimate the scheduler plans with
        # and the instant the payload is transmitted; doppler_hz sets how fast the
        # channel decorrelates over that age (Clarke/Jakes, Eq. (4) extension).
        self.csi_age_s = float(csi_age_s)
        self.doppler_hz = float(doppler_hz)
        self.generator = torch.Generator().manual_seed(self.seed)

    @abstractmethod
    def sample_gain(self, shape: tuple[int, ...]) -> torch.Tensor:
        """Complex baseband gain with unit average power, ``E|h|^2 = 1``."""

    def csi_correlation(self) -> float:
        """Correlation between the aged estimate and the channel at transmission.

        ``rho = J0(2 pi f_d tau)`` for the Clarke/Jakes autocorrelation, clipped to
        ``[0, 1]``. ``rho = 1`` (no age, or no mobility) leaves the historical
        perfect-CSI-in-time behaviour untouched.
        """
        if self.csi_age_s <= 0.0 or self.doppler_hz <= 0.0:
            return 1.0
        from scipy.special import j0
        return float(min(1.0, max(0.0, abs(j0(2.0 * math.pi * self.doppler_hz
                                               * self.csi_age_s)))))

    def sample(self, shape: tuple[int, ...] = (1,)) -> ChannelState:
        h = self.sample_gain(tuple(shape))
        h_hat = h
        rho = self.csi_correlation()
        if rho < 1.0:
            # The estimate describes an older realisation: keep the component of the
            # true gain that survived the age and refresh the rest, so that
            # E|h_hat|^2 = 1 and corr(h_hat, h) = rho.
            fresh = self._complex_normal(tuple(h.shape), 1.0)
            h_hat = rho * h + math.sqrt(max(0.0, 1.0 - rho * rho)) * fresh
        if self.csi_error_variance > 0:
            h_hat = h_hat + self._complex_normal(tuple(h.shape), self.csi_error_variance)
        return ChannelState(h=h, h_hat=h_hat, snr_linear=self.snr_linear,
                            csi_error_variance=self.csi_error_variance,
                            packet_loss_rate=self.packet_loss_rate, channel=self.name,
                            metadata={"csi_correlation": rho,
                                      "csi_age_s": self.csi_age_s,
                                      "doppler_hz": self.doppler_hz})

    def _complex_normal(self, shape: tuple[int, ...], variance: float) -> torch.Tensor:
        scale = math.sqrt(variance / 2.0)
        real = torch.randn(shape, generator=self.generator, dtype=torch.float32) * scale
        imag = torch.randn(shape, generator=self.generator, dtype=torch.float32) * scale
        return torch.complex(real, imag)

    def transmit(self, payload, **kwargs):
        del kwargs
        return payload

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(snr_db={self.snr_db}, "
                f"sigma_e^2={self.csi_error_variance}, p_loss={self.packet_loss_rate}, "
                f"seed={self.seed})")
