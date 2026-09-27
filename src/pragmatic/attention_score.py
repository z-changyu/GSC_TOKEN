from __future__ import annotations

import torch

from .utility import ScoreResult


class AttentionStructuralScorer:
    """Implements paper Eq. (10) from genuine [layers?, heads, query, key] attention.

    A_i = 1/(L_c H) * sum_l sum_h [ 1 / max{1, N - i} ] * sum_{j = i+1}^{N} alpha^{(l,h)}_{j,i}

    Three details are mandated by the paper and were previously violated:

    1. The inner average runs over the strictly later queries ``j >= i + 1``, so
       the normaliser is ``max{1, N - i}`` and not the full query dimension
       ``N``. Dividing by ``N`` systematically under-weights late tokens.
    2. The sum starts at ``j = i + 1``: Section III-A states the causal mask
       gives ``alpha_{j,i} = 0`` for ``i >= j``, so the diagonal self-attention
       weight is excluded. ``A_i`` measures support lent to *subsequent*
       autoregressive states, which the diagonal does not represent. Real
       decoders return a nonzero diagonal, so it is zeroed explicitly here.
    3. ``A_i`` averages over the whole set of masked decoder layers ``L_c``, not
       over the last layer alone. ``layer_reduction="last"`` is kept only as an
       explicitly labelled ablation.
    """

    LAYER_REDUCTIONS = ("mean", "last")

    def __init__(self, layer_reduction: str = "mean") -> None:
        if layer_reduction not in self.LAYER_REDUCTIONS:
            raise ValueError(f"layer_reduction must be one of {self.LAYER_REDUCTIONS}")
        self.layer_reduction = layer_reduction

    def score(self, attentions, token_ids: list[int], token_strings: list[str]) -> ScoreResult:
        if attentions is None:
            raise ValueError("attention tensors are required; proxies are not accepted")
        tensor = torch.stack(list(attentions)) if isinstance(attentions, (tuple, list)) else attentions
        # Accept [L,B,H,Q,K], [B,H,Q,K], or [H,Q,K].
        if tensor.ndim == 5:
            tensor = tensor[-1] if self.layer_reduction == "last" else tensor.mean(0)
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.ndim != 3:
            raise ValueError(f"expected attention rank 3 after reduction, got {tensor.shape}")
        n = len(token_ids)
        if tensor.shape[-2:] != (n, n):
            raise ValueError(f"attention/token mismatch: {tensor.shape[-2:]} versus {(n, n)}")
        # tensor[h, j, i] = alpha_{j,i}. Average over heads, drop the self-attention
        # diagonal (Eq. 10 sums j >= i+1), then sum over the query axis.
        directed = tensor.float().mean(dim=0)
        directed = directed - torch.diag_embed(torch.diagonal(directed, dim1=-2, dim2=-1))
        incoming = directed.sum(dim=0)
        # Eq. (10) normaliser max{1, N - i}; with 0-indexed positions p, N - i = N - 1 - p.
        positions = torch.arange(n, device=incoming.device, dtype=incoming.dtype)
        admissible = (n - positions - 1).clamp_min(1.0)
        values = incoming / admissible
        return ScoreResult(values, token_ids, token_strings,
                           {"equation": 10, "heads": tensor.shape[0],
                            "layer_reduction": self.layer_reduction,
                            "query_normalization": "max{1,N-i}",
                            "self_attention_diagonal": "excluded"})
