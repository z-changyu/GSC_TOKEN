"""DeepMIMO ray-tracing channel, loaded from a local scenario export.

The paper evaluates DeepMIMO scenario I1 [15]. That dataset is a ray-tracing
export that is not part of this repository and cannot be fetched without
network access, so this module is a loader rather than a synthesised stand-in:
it consumes a ``.npy`` (or ``.npz`` with key ``"channels"``) array of complex
channel vectors, normalises the average gain to one, and samples rows.

Exporting a scenario is a single command once the data is available::

    import numpy as np
    np.save("data/channels/deepmimo_i1.npy", dataset[0]["user"]["channel"].astype(np.complex64))

Until then :func:`DeepMIMOChannel.is_available` returns ``False`` and the
experiment driver must report the channel as unavailable rather than
substituting a hand-written fading law under the DeepMIMO name.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .base import Channel


class DeepMIMOChannel(Channel):
    """Empirical ray-tracing channel sampled from a local export."""

    name = "deepmimo"

    def __init__(self, *, scenario: str = "I1", path: str | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.scenario = scenario
        self.path = Path(path) if path else None
        self._gains = None
        if self.path is not None:
            self._gains = self._load(self.path)

    @staticmethod
    def is_available(path: str | None) -> bool:
        return path is not None and Path(path).exists()

    @staticmethod
    def _load(path: Path) -> torch.Tensor:
        if not path.exists():
            raise FileNotFoundError(
                f"DeepMIMO export not found at {path}. Export the scenario to a .npy array "
                f"of complex channel vectors before evaluating this channel.")
        array = np.load(path, allow_pickle=False)
        if isinstance(array, np.lib.npyio.NpzFile):
            if "channels" not in array:
                raise KeyError(f"{path} must contain a 'channels' key")
            array = array["channels"]
        gains = torch.as_tensor(np.asarray(array), dtype=torch.complex64).reshape(-1)
        if gains.numel() == 0:
            raise ValueError(f"{path} contains no channel realisations")
        power = gains.abs().pow(2).mean()
        if not torch.isfinite(power) or power <= 0:
            raise ValueError(f"{path} has a degenerate average gain")
        return gains / torch.sqrt(power)

    def sample_gain(self, shape: tuple[int, ...]) -> torch.Tensor:
        if self._gains is None:
            raise FileNotFoundError(
                "DeepMIMOChannel requires a scenario export; pass path=<...npy>. "
                "No synthetic substitute is produced under the DeepMIMO name.")
        flat = int(torch.Size(shape).numel())
        index = torch.randint(0, self._gains.numel(), (flat,), generator=self.generator)
        return self._gains[index].reshape(shape)

    def __repr__(self) -> str:
        return (f"DeepMIMOChannel(scenario={self.scenario}, rows="
                f"{0 if self._gains is None else self._gains.numel()}, snr_db={self.snr_db})")
