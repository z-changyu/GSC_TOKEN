"""Unified, lossless-on-metadata dataset adapter API."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence


@dataclass
class UnifiedSample:
    sample_id: str
    dataset: str
    split: str
    image: Any
    question: str
    choices: list[str] | None
    ground_truth: str
    answer_type: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DatasetAdapter(ABC, Sequence[dict[str, Any]]):
    name: str

    def __init__(self, split: str, data_dir: str | Path | None = None) -> None:
        self.split = split
        self.data_dir = Path(data_dir) if data_dir else None

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, index: int) -> dict[str, Any]: ...

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return (self[i] for i in range(len(self)))

    @staticmethod
    def validate(sample: Mapping[str, Any]) -> None:
        required = {"sample_id", "dataset", "split", "image", "question", "choices",
                    "ground_truth", "answer_type", "metadata"}
        missing = required - set(sample)
        if missing:
            raise ValueError(f"Unified sample missing fields: {sorted(missing)}")


_REGISTRY: dict[str, type[DatasetAdapter]] = {}


def register_adapter(*names: str) -> Callable[[type[DatasetAdapter]], type[DatasetAdapter]]:
    def decorator(cls: type[DatasetAdapter]) -> type[DatasetAdapter]:
        for name in names:
            _REGISTRY[name.lower()] = cls
        return cls
    return decorator


def load_dataset_adapter(name: str, split: str, **kwargs: Any) -> DatasetAdapter:
    # Imports populate the registry without imposing datasets at package import time.
    from . import aokvqa, gqa, scienceqa, v2xqa  # noqa: F401
    try:
        return _REGISTRY[name.lower()](split=split, **kwargs)
    except KeyError as exc:
        raise ValueError(f"Unknown dataset {name!r}; available={sorted(_REGISTRY)}") from exc

