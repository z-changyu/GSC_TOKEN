from __future__ import annotations

import torch


class SemanticCouplingScorer:
    """Paper Eq. (16): strictly upper-triangular causal coupling.

    M_ij = 1/(L_c H) * sum_l sum_h alpha^{(l,h)}_{j,i}   for i < j,
    M_ij = 0                                             for i >= j.

    The paper is explicit that "no bidirectional average or symmetric storage is
    used". Symmetrising with (M + M^T)/2 invents a reverse-direction
    (future -> past) attention weight, contradicts the causal-attention
    assumption of Section II-C, and corrupts d_M in Eq. (41) and therefore
    kappa in Eq. (62). The undirected incident matrix needed by the insertion
    and removal gains is obtained separately through Eq. (18), see
    :func:`incident_coupling`, which merely retrieves the same causal
    coefficient for an unordered pair.
    """

    LAYER_REDUCTIONS = ("mean", "last")

    def __init__(self, layer_reduction: str = "mean") -> None:
        if layer_reduction not in self.LAYER_REDUCTIONS:
            raise ValueError(f"layer_reduction must be one of {self.LAYER_REDUCTIONS}")
        self.layer_reduction = layer_reduction

    def score(self, attentions) -> torch.Tensor:
        if attentions is None:
            raise ValueError("genuine attention tensors required")
        tensor = torch.stack(list(attentions)) if isinstance(attentions, (tuple, list)) else attentions
        if tensor.ndim == 5:
            tensor = tensor[-1] if self.layer_reduction == "last" else tensor.mean(0)
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.ndim != 3 or tensor.shape[-1] != tensor.shape[-2]:
            raise ValueError(f"expected [H,N,N], got {tuple(tensor.shape)}")
        # directed[j, i] = alpha_{j,i}; transposing gives entry [i, j] = alpha_{j,i}.
        directed = tensor.float().mean(0)
        causal = directed.transpose(0, 1)
        return causal.triu(1)


def incident_coupling(coupling: torch.Tensor) -> torch.Tensor:
    """Paper Eq. (18): M^{<->}_{ij} = M_{min{i,j}, max{i,j}} for i != j.

    Because ``M`` is strictly upper triangular, ``M + M^T`` simply places the
    unique causal coefficient of each unordered pair in both slots; it does not
    create a new reverse-direction weight.
    """
    if coupling.ndim != 2 or coupling.shape[0] != coupling.shape[1]:
        raise ValueError("coupling must be a square matrix")
    incident = coupling + coupling.transpose(0, 1)
    incident.fill_diagonal_(0)
    return incident
