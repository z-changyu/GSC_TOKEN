"""Prompt construction and answer canonicalisation, shared across drivers."""
from __future__ import annotations

import re

SYSTEM_PROMPT = (
    "You are the transmitter of a semantic communication system. You convert an image and a "
    "question into a compact text description for a text-only receiver model that cannot see "
    "the image."
)

TRANSMITTER_REQUEST = (
    "Question: {question}\n\nCandidate choices:\n{choices}\n\n"
    "Write a short semantic description that a text-only model can use to answer the question. "
    "Requirements: restate the question; list every candidate choice; describe only the visual "
    "and contextual evidence that discriminates the choices; do not state or imply which choice "
    "is correct. Output the description only, with no preamble, heading, or meta-commentary."
)

ANSWER_REQUEST = (
    "Answer the question using the semantic description below. "
    "Reply with the exact option text only, with no explanation."
)

# V2 transmitter prompt: binds evidence to each candidate choice, so the
# text-only receiver does not have to bridge the wording gap between a visual
# description and the option text (failure modes (b)/(d)). Conciseness keeps the
# report inside the 192-token envelope that the 24 GB scoring pipeline provably
# supports (attention memory grows quadratically with sequence length).
TRANSMITTER_REQUEST_V2 = (
    "Question: {question}\n\nCandidate choices:\n{choices}\n\n"
    "Write a semantic description that a text-only model can use to answer the "
    "question. Requirements, in this order:\n"
    "1. Restate the question in a few words.\n"
    "2. List every candidate choice verbatim.\n"
    "3. For EACH candidate choice, give ONE short sentence stating the visible "
    "evidence that supports or contradicts it (quote the specific detail; if "
    "there is no evidence for an option, say so).\n"
    "4. Add any extra contextual detail that discriminates between the "
    "choices.\n"
    "Be concise: the whole description must stay under 180 tokens. Do not "
    "state or imply which choice is correct. Output the description only."
)


def select_transmitter_request(config: dict) -> str:
    """Pick the transmitter prompt version declared in the experiment config."""
    version = str(config.get("transmitter", {}).get("prompt_version", "v1"))
    if version == "v2":
        return TRANSMITTER_REQUEST_V2
    return TRANSMITTER_REQUEST

# Patterns indicating the transmitter echoed the instruction instead of describing.
META_LEAK_PATTERNS = (
    "generate a", "create a", "write a semantic", "write a concise",
    "the prompt should", "semantic prompt", "your task is", "you are asked to",
    "instructions:", "requirements:", "task:", "here is a",
)

# An option letter only counts when it is uppercase and closed by a separator or
# the end of the string, so ordinary sentences are not read as an option.
LEADING_OPTION = re.compile(r"^\s*\(?\s*([A-Z])\s*(?:[\.\):]\s*|$)")


def build_choice_block(choices) -> str:
    return "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices or []))


def build_transmitter_messages(content: list[dict]) -> list[dict]:
    return [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": content}]


def build_scoring_messages(image, question, choices, semantic_prompt) -> list[dict]:
    """``[user: I, Q][assistant: P][user: request]`` for the Eq. (2) teacher forcing."""
    user_content = []
    if image is not None:
        user_content.append({"type": "image", "image": image})
    user_content.append({"type": "text", "text": f"Question: {question}\n\n"
                                                 f"Candidate choices:\n{build_choice_block(choices)}"})
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": [{"type": "text", "text": semantic_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": ANSWER_REQUEST}]},
    ]


def detect_meta_leak(text: str) -> bool:
    head = text.strip().lower()[:120]
    return any(pattern in head for pattern in META_LEAK_PATTERNS)


def locate_span(sequence: list[int], needle: list[int], hint: int) -> int:
    """First index of ``needle`` in ``sequence``, searched around ``hint`` first."""
    n, m = len(sequence), len(needle)
    if m == 0:
        return -1
    low = max(0, hint - 8)
    high = min(n - m, hint + 24)
    for start, end in ((low, high + 1), (0, n - m + 1)):
        for i in range(start, end):
            if sequence[i:i + m] == needle:
                return i
    return -1


def canonical_prediction(text, choices):
    """Return ``(prediction, parse_mode)``. Never searches an explanation for a choice."""
    from ..metrics.qa import normalize_answer

    clean = normalize_answer(text)
    if not choices:
        return text.strip(), "raw"
    norm_choices = [normalize_answer(choice) for choice in choices]

    exact = [choice for choice, norm in zip(choices, norm_choices) if norm == clean]
    if len(exact) == 1:
        return exact[0], "exact"

    match = LEADING_OPTION.match(text)
    if match:
        index = ord(match.group(1)) - 65
        if 0 <= index < len(choices):
            return choices[index], "leading_letter"

    if clean:
        hits = {choice for choice, norm in zip(choices, norm_choices)
                if norm and re.search(rf"\b{re.escape(norm)}\b", clean)}
        if len(hits) == 1:
            return hits.pop(), "unique_substring"
    return text.strip(), "raw"
