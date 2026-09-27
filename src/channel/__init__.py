from .base import Channel, ChannelState
from .deepmimo import DeepMIMOChannel
from .fading import FrequencySelectiveChannel, RayleighChannel, RicianChannel
from .ideal import IdealChannel
from .link import ProtectionCodebook, decoding_probability, reception_degree
from .reliable_capacity import (gain_power_cdf, gain_power_quantile, posterior_gain_parameters,
                                reliable_budget, reliable_spectral_efficiency,
                                robust_reliable_spectral_efficiency)

CHANNELS = {
    "ideal": IdealChannel,
    "rayleigh": RayleighChannel,
    "rician": RicianChannel,
    "frequency_selective": FrequencySelectiveChannel,
    "deepmimo": DeepMIMOChannel,
}


def build_channel(spec: dict):
    """Build a channel from its configuration block.

    ``spec`` is the mapping stored under a ``configs/channels/*.yaml`` file; the
    ``name`` key selects the model and every remaining key is forwarded to the
    constructor.
    """
    spec = dict(spec or {})
    name = str(spec.pop("name", "ideal")).lower()
    if name not in CHANNELS:
        raise ValueError(f"unknown channel {name!r}; available: {sorted(CHANNELS)}")
    deepmimo_path = spec.get("path")
    if name == "deepmimo" and not DeepMIMOChannel.is_available(deepmimo_path):
        raise FileNotFoundError(
            f"channel 'deepmimo' requires a scenario export at {deepmimo_path!r}; "
            "no synthetic substitute is produced under this name")
    return CHANNELS[name](**spec)


__all__ = [
    "Channel", "ChannelState", "IdealChannel", "RayleighChannel", "RicianChannel",
    "FrequencySelectiveChannel", "DeepMIMOChannel", "CHANNELS", "build_channel",
    "ProtectionCodebook", "decoding_probability", "reception_degree",
    "gain_power_cdf", "gain_power_quantile", "posterior_gain_parameters",
    "reliable_budget", "reliable_spectral_efficiency",
    "robust_reliable_spectral_efficiency",
]
