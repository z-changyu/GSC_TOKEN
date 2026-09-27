from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class SelectionResult:
    indices: list[int]
    mask: list[bool]
    metadata: dict[str, Any]


class TokenSelector(ABC):
    @abstractmethod
    def select(self, token_ids, *, retention_ratio: float, **kwargs) -> SelectionResult: ...

