import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def vqa_consensus_accuracy(prediction: str, answers: list[str]) -> float:
    """Standard VQA soft score min(matches/3, 1)."""
    count = Counter(normalize_answer(a) for a in answers)[normalize_answer(prediction)]
    return min(count / 3.0, 1.0)

