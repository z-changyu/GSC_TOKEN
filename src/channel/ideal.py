from __future__ import annotations

import math

import torch

from .base import Channel, ChannelState


class IdealChannel(Channel):
    """Error-free reference link: unit gain, perfect CSI, no packet loss.

    It is the only channel under which the pragmatic scores can be validated in
    isolation, because it removes the outage channel from the end-to-end
    decomposition of Eq. (66).
    """

    name = "ideal"

    def __init__(self, *, seed: int = 0, **kwargs) -> None:
        super().__init__(snr_db=float("inf"), csi_error_variance=0.0,
                         packet_loss_rate=0.0, seed=seed)

    def sample_gain(self, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.ones(shape, dtype=torch.complex64)

    def sample(self, shape: tuple[int, ...] = (1,)) -> ChannelState:
        h = self.sample_gain(tuple(shape))
        return ChannelState(h=h, h_hat=h, snr_linear=float("inf"),
                            csi_error_variance=0.0, packet_loss_rate=0.0,
                            channel=self.name)

    def __repr__(self) -> str:
        return "IdealChannel()"
