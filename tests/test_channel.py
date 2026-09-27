import math

import pytest
import torch

from src.channel import (IdealChannel, RayleighChannel, RicianChannel, build_channel,
                         decoding_probability, gain_power_cdf, gain_power_quantile,
                         posterior_gain_parameters, reliable_budget,
                         reliable_spectral_efficiency, robust_reliable_spectral_efficiency)
from src.channel.fading import FrequencySelectiveChannel
from src.channel.link import ProtectionCodebook, reception_degree


def average_power(channel, shape=(200000,)):
    return float(channel.sample_gain(shape).abs().pow(2).mean())


def test_all_channels_are_power_normalised():
    for channel in (RayleighChannel(snr_db=10),
                    RicianChannel(snr_db=10, k_factor=4.0),
                    FrequencySelectiveChannel(snr_db=10),
                    RicianChannel(snr_db=10, k_factor=0.0)):
        assert average_power(channel) == pytest.approx(1.0, abs=0.02)


def test_rician_zero_k_matches_rayleigh_second_moment():
    assert average_power(RicianChannel(snr_db=10, k_factor=0.0)) == pytest.approx(1.0, abs=0.02)


def test_csi_error_has_the_configured_variance():
    channel = RayleighChannel(snr_db=10, csi_error_variance=0.05, seed=7)
    state = channel.sample((100000,))
    error = state.h_hat - state.h
    assert float(error.abs().pow(2).mean()) == pytest.approx(0.05, rel=0.05)


def test_perfect_csi_returns_the_instantaneous_rate():
    channel = RayleighChannel(snr_db=10, csi_error_variance=0.0, seed=3)
    state = channel.sample((5,))
    c = reliable_spectral_efficiency(state, epsilon_out=0.01)
    expected = torch.log2(1.0 + state.snr_linear * state.h_hat.abs().pow(2))
    assert torch.allclose(c, expected)
    # With perfect CSI the outage target cannot change the budget.
    assert torch.allclose(c, reliable_spectral_efficiency(state, epsilon_out=0.1))


def test_posterior_is_the_mmse_estimate():
    h_hat = torch.tensor([1.0 + 0.5j])
    mu, nu = posterior_gain_parameters(h_hat, 0.05)
    assert nu == pytest.approx(0.05 / 1.05)
    assert complex(mu[0]) == pytest.approx(complex(h_hat[0]) / 1.05)


def test_uninformative_csi_recovers_the_unconditional_rayleigh_budget():
    # As sigma_e^2 -> infinity the posterior reverts to the prior, so the
    # conditional reliable efficiency tends to the unconditional Rayleigh value
    # log2(1 + gamma_bar * (-ln(1 - eps))).
    channel = RayleighChannel(snr_db=15, csi_error_variance=1e6, seed=11)
    state = channel.sample((2,))
    for epsilon in (0.01, 0.05, 0.1):
        c = reliable_spectral_efficiency(state, epsilon)
        expected = math.log2(1.0 + state.snr_linear * (-math.log(1.0 - epsilon)))
        assert float(c.mean()) == pytest.approx(expected, rel=0.02)


def test_reliable_efficiency_is_monotone_in_epsilon_snr_and_csi_quality():
    channel = RayleighChannel(snr_db=15, csi_error_variance=0.01, seed=5)
    state = channel.sample((64,))
    values = [float(reliable_spectral_efficiency(state, eps).mean()) for eps in (0.01, 0.05, 0.1)]
    assert values[0] <= values[1] <= values[2]

    low = RayleighChannel(snr_db=0, csi_error_variance=0.01, seed=5).sample((64,))
    assert float(reliable_spectral_efficiency(low, 0.05).mean()) < values[1]

    noisy = RayleighChannel(snr_db=15, csi_error_variance=0.5, seed=5).sample((64,))
    assert float(reliable_spectral_efficiency(noisy, 0.05).mean()) < values[1]


def test_reliable_budget_scales_with_bandwidth_and_slot():
    state = RayleighChannel(snr_db=15, csi_error_variance=0.01, seed=5).sample((8,))
    assert torch.allclose(reliable_budget(state, 0.05, bandwidth=2.0),
                          reliable_spectral_efficiency(state, 0.05) * 2.0)
    assert torch.allclose(reliable_budget(state, 0.05, slot_duration=3.0),
                          reliable_spectral_efficiency(state, 0.05) * 3.0)


def test_gain_power_cdf_matches_the_quantile():
    mu, nu = posterior_gain_parameters(torch.tensor([0.8 + 0.2j]), 0.3)
    quantile = gain_power_quantile(mu, nu, 0.05)
    assert float(gain_power_cdf(mu, nu, quantile)[0]) == pytest.approx(0.05, abs=1e-4)


def test_robust_capacity_is_conservative_and_converges():
    state = RayleighChannel(snr_db=15, csi_error_variance=0.05, seed=9).sample((16,))
    nominal = reliable_spectral_efficiency(state, 0.05)
    robust = robust_reliable_spectral_efficiency(state, 0.05, num_calibration=512,
                                                 delta_amb=0.05,
                                                 generator=torch.Generator().manual_seed(1))
    assert float(robust.mean()) <= float(nominal.mean()) + 1e-3

    plentiful = robust_reliable_spectral_efficiency(state, 0.05, num_calibration=200000,
                                                    delta_amb=0.05,
                                                    generator=torch.Generator().manual_seed(2))
    assert float(plentiful.mean()) == pytest.approx(float(nominal.mean()), rel=0.02)


def test_ideal_channel_has_no_outage():
    state = IdealChannel().sample((4,))
    assert math.isinf(reliable_spectral_efficiency(state, 0.05).flatten()[0].item())
    q = decoding_probability(state, ProtectionCodebook())
    assert torch.allclose(q, torch.ones_like(q))


def test_protection_codebook_rates_and_ordering():
    book = ProtectionCodebook()
    assert book.code_rates == pytest.approx((1 / 3, 1 / 2, 1.0))
    # A stronger level costs more equivalent bits and runs at a lower code rate.
    assert book.levels[0] > book.levels[1] > book.levels[2]
    assert book.code_rates[0] < book.code_rates[1] < book.code_rates[2]


def test_decoding_probability_orders_levels_and_falls_with_snr():
    book = ProtectionCodebook()
    for snr_db in (0, 15):
        state = RayleighChannel(snr_db=snr_db, csi_error_variance=0.01, seed=4).sample((256,))
        q = decoding_probability(state, book).mean(dim=0)
        assert 0.0 <= float(q.min()) and float(q.max()) <= 1.0
        assert q[0] > q[1] > q[2]

    bad = RayleighChannel(snr_db=0, csi_error_variance=0.01, seed=4).sample((256,))
    good = RayleighChannel(snr_db=15, csi_error_variance=0.01, seed=4).sample((256,))
    assert float(decoding_probability(good, book).mean()) > \
           float(decoding_probability(bad, book).mean())


def test_packet_loss_scales_the_decoding_probability():
    book = ProtectionCodebook()
    clean = RayleighChannel(snr_db=15, csi_error_variance=0.01, seed=4,
                            packet_loss_rate=0.0).sample((128,))
    lossy = RayleighChannel(snr_db=15, csi_error_variance=0.01, seed=4,
                            packet_loss_rate=0.1).sample((128,))
    assert torch.allclose(decoding_probability(lossy, book),
                          decoding_probability(clean, book) * 0.9)


def test_reception_degree_matches_equation_33():
    q = torch.tensor([0.9, 0.7, 0.5])
    mask = [True, False, True]
    levels = [0, 2, 1]
    z = reception_degree(mask, levels, q)
    assert torch.allclose(z, torch.tensor([0.9, 0.0, 0.7]))


def test_build_channel_rejects_missing_deepmimo_export():
    with pytest.raises(FileNotFoundError):
        build_channel({"name": "deepmimo", "path": "data/channels/does_not_exist.npy",
                       "snr_db": 15})
    with pytest.raises(ValueError):
        build_channel({"name": "not_a_channel", "snr_db": 15})


def test_build_channel_from_config_block():
    channel = build_channel({"name": "rician", "k_factor": 2.0, "snr_db": 15,
                             "csi_error_variance": 0.01, "packet_loss_rate": 0.05,
                             "seed": 2026})
    assert isinstance(channel, RicianChannel)
    assert channel.k_factor == 2.0
    assert channel.snr_linear == pytest.approx(10 ** 1.5)


def test_frequency_selective_is_rayleigh_on_each_subcarrier():
    # The marginal on a single subcarrier is Rayleigh by construction: H is a
    # linear combination of independent complex Gaussian taps, so |H|^2 is
    # exponential with unit mean. Frequency selectivity is not a heavier
    # marginal tail, it is the correlation structure across the band.
    channel = FrequencySelectiveChannel(snr_db=10, seed=2)
    power = channel.sample_gain((200000,)).abs().pow(2)
    assert float(power.mean()) == pytest.approx(1.0, abs=0.02)
    assert float(power.var()) == pytest.approx(1.0, abs=0.05)   # exponential: var = mean^2


def test_frequency_selective_varies_across_the_band():
    # Within one tap realisation the gain must fluctuate strongly across
    # subcarriers; a flat channel would show no variation at all.
    channel = FrequencySelectiveChannel(snr_db=10, seed=2, num_taps=4, delay_spread=1.0)
    taps = channel.sample_taps((4000,))
    power = channel.frequency_response(taps).abs().pow(2)
    assert float(power.mean()) == pytest.approx(1.0, abs=0.02)
    assert float(power.var(dim=-1).mean()) > 0.5
    # ... and the response decorrelates with subcarrier separation.
    centred = power - power.mean(dim=-1, keepdim=True)
    norm = centred.pow(2).sum(dim=-1)
    def correlation(lag):
        return float((centred * torch.roll(centred, -lag, dims=-1)).sum(dim=-1).div(norm).mean())
    assert correlation(1) > correlation(8) > correlation(32)
