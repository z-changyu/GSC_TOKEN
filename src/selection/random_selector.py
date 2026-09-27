import math
import random

from .interfaces import SelectionResult, TokenSelector


class RandomTokenSelector(TokenSelector):
    def __init__(self, seed: int) -> None: self.seed = seed

    def select(self, token_ids, *, retention_ratio: float, **kwargs) -> SelectionResult:
        del kwargs
        n = len(token_ids)
        if not 0 < retention_ratio <= 1: raise ValueError("retention_ratio must be in (0, 1]")
        k = min(n, max(1, math.ceil(n * retention_ratio))) if n else 0
        indices = sorted(random.Random(self.seed).sample(range(n), k))
        chosen = set(indices)
        return SelectionResult(indices, [i in chosen for i in range(n)],
                               {"method": "random", "seed": self.seed, "target_count": k})

