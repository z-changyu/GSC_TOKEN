from __future__ import annotations

from collections.abc import Iterable


def token_retention_ratio(selected_tokens: int, original_tokens: int) -> float:
    if original_tokens <= 0:
        raise ValueError("original_tokens must be positive")
    if not 0 <= selected_tokens <= original_tokens:
        raise ValueError("selected_tokens must lie in [0, original_tokens]")
    return selected_tokens / original_tokens


def transmitted_symbol_cost(selected: Iterable[bool | int], rates: Iterable[float]) -> float:
    xs, rs = list(selected), list(rates)
    if len(xs) != len(rs):
        raise ValueError("selected and rates must have equal length")
    return float(sum(float(x) * float(r) for x, r in zip(xs, rs)))

