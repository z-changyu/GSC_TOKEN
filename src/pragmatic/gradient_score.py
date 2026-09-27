from __future__ import annotations

from typing import Optional, Sequence, Union

import torch

from .utility import ScoreResult


class GradientScorer:
    """Auditable G_i (Eq. 14); callers must construct the loss, the mask embedding,
    and the teacher-forcing target explicitly.

    G_i = || dL_T / de_i ||_2 * || e_i - e_mask ||_2

    The deletion magnitude ``||e_i - e_mask||_2`` is the norm of the perturbation
    ``Delta e_i = e_mask - e_i`` introduced by Eq. (12); it is part of Eq. (14)
    and is only optional in the special case flagged by the paper, where all
    embeddings are normalised and every deletion magnitude is nearly equal.
    That degenerate shortcut is not applied silently here.
    """

    def __init__(self, loss_type: str, target_mode: str, aggregation: str = "l2") -> None:
        self.loss_type, self.target_mode, self.aggregation = loss_type, target_mode, aggregation
        if aggregation != "l2":
            raise ValueError("only the paper's L2 aggregation is implemented")

    def score(self, loss: torch.Tensor, input_embeddings: torch.Tensor,
              token_ids: list[int], token_strings: list[str],
              mask_embedding: Optional[torch.Tensor] = None,
              positions: Optional[Union[slice, Sequence[int]]] = None,
              retain_graph: bool = False) -> ScoreResult:
        if not input_embeddings.requires_grad:
            raise ValueError("input_embeddings.requires_grad must be True")
        if mask_embedding is None:
            raise ValueError("Eq. (14) requires the untransmitted-token mask embedding e_mask")

        grad, = torch.autograd.grad(loss, input_embeddings, retain_graph=retain_graph, allow_unused=False)
        if grad.ndim == 3 and grad.shape[0] == 1:
            grad = grad[0]
        embeddings = input_embeddings.detach()
        if embeddings.ndim == 3 and embeddings.shape[0] == 1:
            embeddings = embeddings[0]
        if grad.ndim != 2 or embeddings.ndim != 2:
            raise ValueError("expected a single sequence of token embeddings")

        if positions is not None:
            grad = grad[positions]
            embeddings = embeddings[positions]
        if grad.shape[0] != len(token_ids):
            raise ValueError(f"gradient/token shape mismatch: {tuple(grad.shape)} versus {len(token_ids)}")
        if not torch.isfinite(grad).all():
            raise ValueError("non-finite embedding gradient")

        # Norms are taken in float32: the model runs in bfloat16 and a 2048-way
        # accumulation in bfloat16 loses too much resolution for a ranking score.
        mask_embedding = mask_embedding.detach().to(embeddings.device).float().reshape(1, -1)
        sensitivity = torch.linalg.vector_norm(grad.detach().float(), ord=2, dim=-1)
        deletion = torch.linalg.vector_norm(embeddings.detach().float() - mask_embedding, ord=2, dim=-1)
        return ScoreResult(sensitivity * deletion, token_ids, token_strings,
                           {"equation": 14, "loss_type": self.loss_type, "target_mode": self.target_mode,
                            "differentiated_tensor": "explicit caller-provided input_embeddings",
                            "aggregation": self.aggregation,
                            "deletion_magnitude_factor": True,
                            "positions": "prompt-token slice" if positions is not None else "full sequence"})
