import random

import numpy as np
import pytest

from src.data.base import DatasetAdapter
from src.metrics.communication import token_retention_ratio, transmitted_symbol_cost
from src.metrics.qa import normalize_answer
from src.metrics.statistics import aubc
from src.selection.random_selector import RandomTokenSelector
from src.utils.io import read_jsonl, write_jsonl
from src.utils.seed import set_global_seed


def test_dataset_adapter_required_fields():
    sample = {"sample_id": "1", "dataset": "x", "split": "val", "image": None,
              "question": "q", "choices": None, "ground_truth": "a", "answer_type": "direct", "metadata": {}}
    DatasetAdapter.validate(sample)
    with pytest.raises(ValueError): DatasetAdapter.validate({})


def test_trr(): assert token_retention_ratio(3, 4) == 0.75


def test_symbol_cost(): assert transmitted_symbol_cost([1, 0, 1], [48, 32, 16]) == 64


def test_seed_is_deterministic():
    set_global_seed(2026); first = (random.random(), np.random.rand())
    set_global_seed(2026); assert first == (random.random(), np.random.rand())


def test_random_selector_retention_and_determinism():
    selector = RandomTokenSelector(2026)
    a = selector.select(list(range(10)), retention_ratio=0.3)
    b = selector.select(list(range(10)), retention_ratio=0.3)
    assert len(a.indices) == 3 and a.indices == b.indices


def test_aubc(): assert aubc([0, 1, 2], [0, 1, 0]) == pytest.approx(0.5)


def test_qa_normalization(): assert normalize_answer("The, RED cube!") == "red cube"


def test_manifest_order(tmp_path):
    rows = [{"sample_id": str(i)} for i in [3, 1, 2]]
    path = tmp_path / "m.jsonl"; write_jsonl(path, rows)
    assert read_jsonl(path) == rows

