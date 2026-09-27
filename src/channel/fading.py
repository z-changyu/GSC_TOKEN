"""Rayleigh, Rician and frequency-selective fading, Section II-B."""
from __future__ import annotations

import math
from typing import Any

import torch

from .base import Channel


class RayleighChannel(Channel):
    """h ~ CN(0, 1): no line-of-sight component."""

    name = "rayleigh"

    def sample_gain(self, shape: tuple[int, ...]) -> torch.Tensor:
        real = torch.randn(shape, generator=self.generator, dtype=torch.float32)
        imag = torch.randn(shape, generator=self.generator, dtype=torch.float32)
        return torch.complex(real, imag) / math.sqrt(2.0)


class RicianChannel(Channel):
    """Rician fading with K-factor ``k_factor`` and the LOS component on the real axis.

    h = sqrt(K/(K+1)) + sqrt(1/(K+1)) * CN(0, 1),  so E|h|^2 = 1.
    K = 0 recovers Rayleigh; K -> infinity gives an AWGN channel.
    """

    name = "rician"

    def __init__(self, *, k_factor: float = 4.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if k_factor < 0:
            raise ValueError("k_factor must be nonnegative")
        self.k_factor = float(k_factor)

    def sample_gain(self, shape: tuple[int, ...]) -> torch.Tensor:
        los = math.sqrt(self.k_factor / (1.0 + self.k_factor))
        scatter = math.sqrt(1.0 / (1.0 + self.k_factor) / 2.0)
        real = torch.randn(shape, generator=self.generator, dtype=torch.float32) * scatter
        imag = torch.randn(shape, generator=self.generator, dtype=torch.float32) * scatter
        return torch.complex(real + los, imag)

    def __repr__(self) -> str:
        return f"RicianChannel(K={self.k_factor}, snr_db={self.snr_db})"


class FrequencySelectiveChannel(Channel):
    """Exponential power-delay-profile multipath observed on one OFDM subcarrier.

    The ``num_taps`` path gains follow ``CN(0, sigma_l^2)`` with
    ``sigma_l^2 ~ exp(-l / delay_spread)`` normalised to unit total power, and
    the reported gain is the frequency response on a uniformly chosen
    subcarrier, ``H = sum_l g_l exp(-j 2 pi u l / fft_size)``. Different draws
    therefore land on different parts of the spectrum, which is what makes the
    model frequency selective rather than a scaled Rayleigh.
    """

    name = "frequency_selective"

    def __init__(self, *, num_taps: int = 4, delay_spread: float = 1.0,
                 fft_size: int = 64, subcarrier: str = "random", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if num_taps < 1:
            raise ValueError("num_taps must be at least one")
        if delay_spread <= 0:
            raise ValueError("delay_spread must be positive")
        if fft_size < 1:
            raise ValueError("fft_size must be at least one")
        if subcarrier not in ("random", "first"):
            raise ValueError("subcarrier must be 'random' or 'first'")
        self.num_taps = int(num_taps)
        self.delay_spread = float(delay_spread)
        self.fft_size = int(fft_size)
        self.subcarrier = subcarrier
        weights = torch.exp(-torch.arange(self.num_taps, dtype=torch.float32) / self.delay_spread)
        # Normalising the profile keeps E|H|^2 = 1 on every subcarrier.
        self._tap_std = torch.sqrt(weights / weights.sum())

    def sample_taps(self, shape: tuple[int, ...]) -> torch.Tensor:
        """Path gains ``[..., num_taps]`` with the exponential power profile."""
        taps = []
        for std in self._tap_std:
            scale = float(std) / math.sqrt(2.0)
            real = torch.randn(shape, generator=self.generator, dtype=torch.float32) * scale
            imag = torch.randn(shape, generator=self.generator, dtype=torch.float32) * scale
            taps.append(torch.complex(real, imag))
        return torch.stack(taps, dim=-1)

    def frequency_response(self, taps: torch.Tensor) -> torch.Tensor:
        """``[..., fft_size]`` complex response over the whole band."""
        index = torch.arange(self.fft_size, dtype=torch.float32)
        phase = -2.0 * math.pi * torch.outer(index, torch.arange(self.num_taps, dtype=torch.float32)) / self.fft_size
        kernel = torch.exp(torch.complex(torch.zeros_like(phase), phase))  # [fft_size, num_taps]
        return taps @ kernel.T

    def sample_gain(self, shape: tuple[int, ...]) -> torch.Tensor:
        taps = self.sample_taps(tuple(shape))
        response = self.frequency_response(taps)
        if self.subcarrier == "first":
            return response[..., 0]
        flat = response.reshape(-1, self.fft_size)
        pick = torch.randint(0, self.fft_size, (flat.shape[0],), generator=self.generator)
        return flat[torch.arange(flat.shape[0]), pick].reshape(shape)

    def __repr__(self) -> str:
        return (f"FrequencySelectiveChannel(taps={self.num_taps}, "
                f"delay_spread={self.delay_spread}, snr_db={self.snr_db})")
