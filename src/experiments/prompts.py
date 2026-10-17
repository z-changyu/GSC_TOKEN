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

# The terminal's own decision, asked of the full terminal input (image + question +
# choices) *before* any compression. It replaces the ground-truth answer as the
# teacher-forcing target of the Eq. (14) gradient, so that scoring never reads the
# test label: what the terminal would have answered is what the report should carry.
TRANSMITTER_SELF_QUERY = (
    "Question: {question}\n\nCandidate choices:\n{choices}\n\n"
    "Reply with only the option letter (A, B, C or D), with no other text."
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


# V3 transmitter prompt: attacks the two failure modes measured by
# scripts/screen_transmitter_reports.py on V2X-QA --
#   (a) hedging/negation lists ("there is no X, no Y") get misread by the receiver
#       as a positive claim about X/Y;
#   (b) padding dilutes the single discriminating detail (distilled reports grew
#       from 629 to 758 chars and lost 11.7pp).
# So V3 fixes a per-candidate verdict format and a hard length cap, and forbids
# restating the question/choices (which V1/V2 spend ~45% of the report on).
TRANSMITTER_REQUEST_V3 = (
    "Question: {question}\n\nCandidate choices:\n{choices}\n\n"
    "Write the shortest report that lets a model which cannot see the image pick "
    "the right choice. Output exactly one line per candidate, in order, and "
    "nothing else:\n"
    "A: <VISIBLE|NOT VISIBLE|UNCLEAR> - <the concrete detail you can see>\n"
    "Rules:\n"
    "- Use only details actually present in the image. If a candidate's detail is "
    "absent, write 'NOT VISIBLE - no such detail in the image'.\n"
    "- Mention only what differs between the candidates; drop anything common to "
    "all of them.\n"
    "- Do not say or hint which candidate is correct. No preamble, no question "
    "restatement, no summary sentence.\n"
    "- Keep the whole report under 90 tokens."
)


# V4 transmitter prompt: V1 minus its measured waste. On V2X-QA the V1 report spends
# ~45% of its tokens restating the question and the choice texts, which the receiver
# already receives as RSU-issued task intent. V4 forbids that restatement and adds an
# explicit "no preamble / no summary" rule, giving the same free-form evidence in a
# much smaller payload -- which is what the reliable-capacity budget actually pays for.
TRANSMITTER_REQUEST_V4 = (
    "Question: {question}\n\nCandidate choices:\n{choices}\n\n"
    "Write a short semantic description that a text-only model can use to answer the "
    "question. Requirements:\n"
    "1. Do NOT repeat the question or the candidate choices; the receiver already has them.\n"
    "2. Describe the visible evidence and say how it bears on the choices, so the "
    "receiver can tell them apart.\n"
    "3. Quote only details that are actually visible; never invent detail.\n"
    "4. Do not state or imply which choice is correct.\n"
    "Output the description only, with no preamble and no closing summary. "
    "Keep it under 110 tokens."
)


def select_transmitter_request(config: dict) -> str:
    """Pick the transmitter prompt version declared in the experiment config."""
    version = str(config.get("transmitter", {}).get("prompt_version", "v1"))
    if version == "v2":
        return TRANSMITTER_REQUEST_V2
    if version == "v3":
        return TRANSMITTER_REQUEST_V3
    if version == "v4":
        return TRANSMITTER_REQUEST_V4
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
