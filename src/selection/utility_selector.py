import math

from .interfaces import SelectionResult, TokenSelector


class UtilityTokenSelector(TokenSelector):
    """Minimal additive utility selector; coupling-aware swapping is kept opt-in."""
    def select(self, token_ids, *, retention_ratio: float, utilities=None, rates=None, **kwargs) -> SelectionResult:
        del kwargs
        n = len(token_ids)
        if utilities is None or len(utilities) != n: raise ValueError("one utility per token is required")
        rates = [1.0] * n if rates is None else list(rates)
        if len(rates) != n or any(r <= 0 for r in rates): raise ValueError("rates must be positive and aligned")
        k = min(n, max(1, math.ceil(n * retention_ratio))) if n else 0
        ranking = sorted(range(n), key=lambda i: (-float(utilities[i]) / rates[i], i))
        indices = sorted(ranking[:k]); chosen = set(indices)
        return SelectionResult(indices, [i in chosen for i in range(n)],
                               {"method": "utility", "target_count": k, "coupling_swap": False})


class FullPromptSelector(TokenSelector):
    def select(self, token_ids, *, retention_ratio: float = 1.0, **kwargs) -> SelectionResult:
        del retention_ratio, kwargs
        n = len(token_ids)
        return SelectionResult(list(range(n)), [True] * n, {"method": "full", "target_count": n})

