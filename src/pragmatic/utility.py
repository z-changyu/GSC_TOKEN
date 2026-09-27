from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .coupling import incident_coupling


@dataclass
class ScoreResult:
    scores: torch.Tensor
    token_ids: list[int]
    token_strings: list[str]
    normalization: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, ...]: return tuple(self.scores.shape)


def minmax_normalize(values: torch.Tensor, eps: float = 1e-12) -> tuple[torch.Tensor, dict[str, float]]:
    lo, hi = values.min(), values.max()
    normalized = (values - lo) / (hi - lo).clamp_min(eps)
    return normalized, {"method": "minmax", "min": float(lo.detach()), "max": float(hi.detach()), "eps": eps}


class UnifiedUtilityScorer:
    """Paper Eq. (15): u'_i = lambda_1 A~_i + lambda_2 S~_i + lambda_3 G~_i."""

    def __init__(self, lambda_attention: float, lambda_semantic: float, lambda_gradient: float) -> None:
        weights = (lambda_attention, lambda_semantic, lambda_gradient)
        if any(w < 0 for w in weights) or abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("nonnegative pragmatic weights must sum to one")
        self.weights = weights

    def score(self, attention: ScoreResult, semantic: ScoreResult, gradient: ScoreResult) -> ScoreResult:
        if not (attention.token_ids == semantic.token_ids == gradient.token_ids):
            raise ValueError("score token alignments differ")
        normalized, metadata = [], []
        for result in (attention, semantic, gradient):
            values, norm = minmax_normalize(result.scores)
            normalized.append(values); metadata.append(norm)
        scores = sum(w * value for w, value in zip(self.weights, normalized))
        return ScoreResult(scores, attention.token_ids, attention.token_strings,
                           {"components": metadata, "weights": self.weights,
                            "legacy_paper_parameters": True})


def coupling_regime(coupling: torch.Tensor, beta: float) -> dict[str, float]:
    """Paper Eq. (41) and Eq. (62): d_M and the weak-coupling coefficient kappa.

    d_M,k = max_i ( sum_{j>i} M_ij + sum_{j<i} M_ji ),  kappa_k = beta * d_M,k / 2 < 1.
    """
    if coupling.ndim != 2 or coupling.shape[0] != coupling.shape[1]:
        raise ValueError("coupling must be a square matrix")
    if beta < 0:
        raise ValueError("beta must be nonnegative")
    d_m = float(incident_coupling(coupling).sum(dim=1).max())
    kappa = beta * d_m / 2.0
    return {"d_M": d_m, "kappa": kappa, "weak_coupling": bool(kappa < 1.0)}


def context_aware_utility(base: torch.Tensor, coupling: torch.Tensor, beta: float) -> torch.Tensor:
    """Paper Eq. (19): insertion gain of each token into the full token set.

    u_i = u'_i ( 1 + beta * sum_{j != i} M^{<->}_{ij} u'_j )

    This is Eq. (56) evaluated with T = every other token. The previously
    implemented form dropped the leading u'_i from the coupling term, which
    turned the quadratic set function of Eq. (17) into a linear expression and
    broke the link with the local-refinement gains of Eq. (56)-(59).
    """
    if coupling.shape != (base.numel(), base.numel()):
        raise ValueError("coupling shape mismatch")
    regime = coupling_regime(coupling, beta)
    if not regime["weak_coupling"]:
        raise ValueError(f"beta violates the paper's weak-coupling condition: kappa={regime['kappa']:.4f} >= 1")
    incident = incident_coupling(coupling)
    return base * (1.0 + beta * (incident @ base))


def set_utility(membership, base: torch.Tensor, coupling: torch.Tensor, beta: float) -> torch.Tensor:
    """Paper Eq. (17): utility of the token set that is actually received.

    F_T(T) = sum_{i in T} u'_i + beta * sum_{i<j, i,j in T} M_ij u'_i u'_j

    ``membership`` is a boolean mask or an index sequence over the full token
    set. The coupling term exists only when both related tokens are received,
    which is what removes the unselected-token leakage of a full-prompt
    pre-score.
    """
    if coupling.shape != (base.numel(), base.numel()):
        raise ValueError("coupling shape mismatch")
    regime = coupling_regime(coupling, beta)
    if not regime["weak_coupling"]:
        raise ValueError(f"beta violates the paper's weak-coupling condition: kappa={regime['kappa']:.4f} >= 1")
    index = torch.as_tensor(membership)
    if index.dtype != torch.bool:
        index = torch.zeros(base.numel(), dtype=torch.bool, device=base.device).scatter_(0, index, True)
    selected = base[index]
    if selected.numel() == 0:
        return base.new_zeros(())
    sub = coupling[index][:, index]
    pairwise = torch.outer(selected, selected)
    return selected.sum() + beta * (sub.triu(1) * pairwise).sum()
