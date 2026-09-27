import torch

from .utility import ScoreResult


class SemanticDirectionalScorer:
    """Implements paper Eq. (2) using token embeddings and A_i."""
    def score(self, embeddings: torch.Tensor, attention_scores: torch.Tensor,
              token_ids: list[int], token_strings: list[str]) -> ScoreResult:
        if embeddings.ndim != 2 or embeddings.shape[0] != len(token_ids): raise ValueError("embedding shape mismatch")
        if attention_scores.shape != (len(token_ids),): raise ValueError("attention shape mismatch")
        denom = attention_scores.sum().clamp_min(1e-12)
        direction = (attention_scores[:, None] * embeddings).sum(0) / denom
        values = torch.nn.functional.cosine_similarity(embeddings, direction[None, :], dim=-1)
        return ScoreResult(values, token_ids, token_strings, {"equation": 2, "direction_weight": "A_i"})

