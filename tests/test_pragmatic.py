import pytest
import torch

from src.pragmatic.attention_score import AttentionStructuralScorer
from src.pragmatic.coupling import SemanticCouplingScorer, incident_coupling
from src.pragmatic.gradient_score import GradientScorer
from src.pragmatic.semantic_score import SemanticDirectionalScorer
from src.pragmatic.utility import (ScoreResult, UnifiedUtilityScorer, context_aware_utility,
                                   coupling_regime, set_utility)


def tokens(): return [10, 11], ["a", "b"]


def causal_attention():
    """alpha_{j,i} with a strict causal mask over three tokens, one head."""
    return torch.tensor([[
        [1.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.2, 0.3, 0.5],
    ]])


def test_attention_equation_10_normalizes_over_admissible_queries():
    ids, strings = [10, 11, 12], ["a", "b", "c"]
    result = AttentionStructuralScorer().score(causal_attention(), ids, strings)
    # The diagonal is excluded (Eq. 10 sums j >= i+1).
    # key 0: (0.5 + 0.2) / max{1, 3-1-0} = 0.7 / 2 = 0.35
    # key 1: (0.3)      / max{1, 3-1-1} = 0.3 / 1 = 0.30
    # key 2: no later query exists, so the incoming mass is 0.
    assert torch.allclose(result.scores, torch.tensor([0.35, 0.30, 0.0]))
    assert result.normalization["equation"] == 10
    assert result.normalization["query_normalization"] == "max{1,N-i}"


def test_attention_averages_over_masked_layers_by_default():
    ids, strings = [10, 11, 12], ["a", "b", "c"]
    # [L, B, H, Q, K] as returned layer-by-layer by Transformers.
    stacked = torch.stack([causal_attention().unsqueeze(0), (causal_attention() * 3.0).unsqueeze(0)])
    mean_scores = AttentionStructuralScorer().score(stacked, ids, strings).scores
    last_scores = AttentionStructuralScorer("last").score(stacked, ids, strings).scores
    # mean over the two layers is 2x the base matrix, "last" is 3x it.
    assert torch.allclose(mean_scores * 3.0, last_scores * 2.0)
    assert not torch.allclose(mean_scores, last_scores)


def test_coupling_is_strictly_upper_triangular_and_causal():
    coupling = SemanticCouplingScorer().score(causal_attention())
    # M_01 = alpha_{1,0} = 0.5, M_02 = alpha_{2,0} = 0.2, M_12 = alpha_{2,1} = 0.3
    expected = torch.tensor([[0.0, 0.5, 0.2],
                             [0.0, 0.0, 0.3],
                             [0.0, 0.0, 0.0]])
    assert torch.allclose(coupling, expected)
    assert torch.allclose(coupling, coupling.triu(1))
    # A symmetrised matrix would carry a reverse-direction weight and must not be produced.
    assert float(coupling[1, 0]) == 0.0


def test_incident_coupling_retrieves_the_same_coefficient():
    coupling = SemanticCouplingScorer().score(causal_attention())
    incident = incident_coupling(coupling)
    assert torch.allclose(incident, incident.T)
    assert float(incident[0, 1]) == float(incident[1, 0]) == 0.5
    assert float(incident.trace()) == 0.0


def test_semantic_and_gradient_shapes():
    ids, strings = tokens()
    embeddings = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
    semantic = SemanticDirectionalScorer().score(embeddings, torch.tensor([0.75, 0.25]), ids, strings)
    gradient = GradientScorer("toy_sum", "unit_test").score(
        (embeddings ** 2).sum(), embeddings, ids, strings,
        mask_embedding=torch.zeros(2))
    assert semantic.shape == gradient.shape == (2,)
    assert gradient.normalization["loss_type"] == "toy_sum"


def test_gradient_equation_14_multiplies_the_deletion_magnitude():
    ids, strings = tokens()
    embeddings = torch.tensor([[3., 4.], [0., 1.]], requires_grad=True)
    loss = (embeddings ** 2).sum()
    # ||grad||_2 = ||[6, 8]|| = 10 for token 0, ||[0, 2]|| = 2 for token 1.
    # ||e_i - e_mask||_2 with e_mask = 0 -> 5 and 1.
    result = GradientScorer("toy_sum", "unit_test").score(loss, embeddings, ids, strings,
                                                          mask_embedding=torch.zeros(2))
    assert torch.allclose(result.scores, torch.tensor([50.0, 2.0]))
    assert result.normalization["deletion_magnitude_factor"] is True


def test_gradient_requires_the_mask_embedding():
    ids, strings = tokens()
    embeddings = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
    with pytest.raises(ValueError, match="mask embedding"):
        GradientScorer("toy_sum", "unit_test").score((embeddings ** 2).sum(), embeddings, ids, strings)


def test_unified_utility_alignment():
    ids, strings = tokens(); mk = lambda x: ScoreResult(torch.tensor(x), ids, strings)
    result = UnifiedUtilityScorer(.25, .30, .45).score(mk([0., 1.]), mk([1., 0.]), mk([0., 1.]))
    assert result.shape == (2,)


def test_context_aware_utility_is_quadratic_in_the_base_scores():
    base = torch.tensor([1.0, 0.5])
    coupling = torch.tensor([[0.0, 0.4], [0.0, 0.0]])
    beta = 0.5
    # u_0 = 1.0 * (1 + 0.5 * 0.4 * 0.5) = 1.10 ; u_1 = 0.5 * (1 + 0.5 * 0.4 * 1.0) = 0.60
    assert torch.allclose(context_aware_utility(base, coupling, beta), torch.tensor([1.10, 0.60]))
    # Scaling every base score by two must scale the coupling term by four.
    scaled = context_aware_utility(base * 2.0, coupling, beta)
    assert torch.allclose(scaled, torch.tensor([2.0 * (1 + 0.5 * 0.4 * 1.0), 1.0 * (1 + 0.5 * 0.4 * 2.0)]))


def test_set_utility_matches_equation_17():
    base = torch.tensor([1.0, 0.5, 0.25])
    coupling = torch.tensor([[0.0, 0.4, 0.1],
                             [0.0, 0.0, 0.2],
                             [0.0, 0.0, 0.0]])
    beta = 0.5
    mask = torch.tensor([True, True, False])
    # F = (1.0 + 0.5) + 0.5 * M_01 * u0 * u1 = 1.5 + 0.5 * 0.4 * 1.0 * 0.5
    expected = 1.5 + 0.5 * 0.4 * 1.0 * 0.5
    assert torch.allclose(set_utility(mask, base, coupling, beta), torch.tensor(expected))
    # Dropping token 0 removes its coupling with token 1 as well as its own base score.
    only_last = set_utility(torch.tensor([False, True, False]), base, coupling, beta)
    assert torch.allclose(only_last, torch.tensor(0.5))


def test_coupling_regime_flags_strong_coupling():
    coupling = torch.tensor([[0.0, 3.0], [0.0, 0.0]])
    regime = coupling_regime(coupling, beta=1.0)
    assert regime["d_M"] == pytest.approx(3.0)
    assert regime["kappa"] == pytest.approx(1.5)
    assert regime["weak_coupling"] is False
    with pytest.raises(ValueError, match="weak-coupling"):
        context_aware_utility(torch.tensor([1.0, 1.0]), coupling, beta=1.0)
