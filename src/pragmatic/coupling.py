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


NORMALIZATIONS = ("none", "row")

# Coupling definitions compared by scripts/eval_coupling_variants.py. Measured on
# V2X-QA at equal coupling strength, "B" is the only one that makes the set score
# depend on *which* tokens were chosen rather than only on how many (rank agreement
# with the additive score 0.951 versus 0.997 for the others, and the only variant
# whose pair term tracks the measured complementarity, +0.14 versus ~0).
COUPLING_VARIANTS = ("raw", "row", "A", "B")


def cosine_similarity(embeddings: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine similarity of per-token embeddings."""
    x = embeddings.float()
    x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return x @ x.t()


def build_coupling_variant(kind: str, coupling: torch.Tensor,
                           embeddings: torch.Tensor) -> torch.Tensor:
    """Apply a coupling definition to the raw Eq. (16) coefficients.

    ``raw``  the attention coefficients themselves;
    ``row``  unit-row (Sinkhorn) scaling, so ``d_M = 1`` and ``beta`` is usable up to 2;
    ``A``    ``a_ij * (1 - cos(e_i, e_j))`` then unit-row scaling -- keep attention
             weight only where the two tokens do not already say the same thing;
    ``B``    ``M_ij = -cos(e_i, e_j)``: an explicit redundancy *penalty*, i.e.
             ``F_T(T) = sum_i u'_i - gamma sum_{i<j} sim_ij u'_i u'_j``. This is the
             variant the measurements favour and it is automatically inside the
             weak-coupling regime because its coupling is negative.
    """
    if kind not in COUPLING_VARIANTS:
        raise ValueError(f"unknown coupling variant {kind!r}; available: {COUPLING_VARIANTS}")
    if kind == "raw":
        return coupling
    if kind == "row":
        return normalize_coupling(coupling, "row")
    similarity = cosine_similarity(embeddings)
    if kind == "A":
        return normalize_coupling((coupling * (1.0 - similarity)).triu(1), "row")
    return (-similarity).triu(1)


def normalize_coupling(coupling: torch.Tensor, mode: str = "row",
                       iterations: int = 50, tolerance: float = 1e-6) -> torch.Tensor:
    """Rescale the incident coupling so the weak-coupling bound stops binding.

    ``none`` returns the raw Eq. (16) coefficients. ``row`` applies a Sinkhorn-style
    scaling ``B_ij = x_i * M^{<->}_ij * x_j`` chosen so that ``B`` stays **symmetric**
    and every row sums to one. Symmetry is required, not cosmetic: ``M^{<->}`` is
    rebuilt from the output by :func:`incident_coupling`, so a merely row-normalised
    (asymmetric) matrix would be re-symmetrised into something with row sums != 1.
    With ``B`` symmetric with unit rows, ``d_M = 1`` in Eq. (41) and therefore
    ``kappa = beta / 2`` in Eq. (62) -- the usable range becomes ``beta in [0, 2)``.

    Why rescaling alone cannot substitute: both the coupling term and ``d_M`` are
    homogeneous of degree one in ``M``, so multiplying ``M`` by a constant leaves
    the ratio "coupling share / additive share" *and* the bound ``beta < 2 / d_M``
    unchanged. The binding quantity is the *concentration* of ``M`` relative to the
    weighting it is multiplied by: a raw attention matrix spread over ~130 tokens has
    a row sum of 1.5-2.9 while any single coefficient is ~0.01, which pins the pair
    term at ~1e-3 -- three orders of magnitude below any measurable receiver effect.
    Unit-row scaling gives ``sum_j B_ij u'_j ~ mean(u')``, so a 25% coupling share
    costs ``beta ~= 1``, comfortably inside the weak-coupling regime.

    The semantics of ``M_ij`` become "share of the coupling budget attributed to
    token j", the influence-coefficient role Eq. (16) plays in Eq. (17) and (56).
    """
    if mode not in NORMALIZATIONS:
        raise ValueError(f"unknown coupling normalisation {mode!r}; available: {NORMALIZATIONS}")
    if mode == "none":
        return coupling
    incident = incident_coupling(coupling)
    scaled = incident.clone()
    for _ in range(int(iterations)):
        rows = scaled.sum(dim=1, keepdim=True)
        scaled = scaled / rows.clamp_min(1e-30)
        scaled = (scaled + scaled.transpose(0, 1)) / 2.0
        scaled.fill_diagonal_(0)
        deviation = float((scaled.sum(dim=1) - 1.0).abs().max()) if scaled.numel() else 0.0
        if deviation < tolerance:
            break
    scaled.fill_diagonal_(0)
    # Store with the same strictly-upper-triangular convention as the raw matrix.
    return scaled.triu(1)


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
