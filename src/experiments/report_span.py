"""Locate the report body inside the scored payload.

The payload that gets compressed is ``question + option list + report``, but the
receiver builds its own prompt from ``build_question_block`` and therefore already
holds the question and the options. Everything before the report body is
duplicated: it costs transmit symbols and returns no information.

Measurement on 20 V2X-QA samples: the duplicated region is 30% of the payload but
absorbs 44% of the top-10% pragmatic utility, so the scheduler spends a
disproportionate share of the budget on it. ``report_span_start`` gives the token
index where the report body begins so callers can exclude the prefix.
"""
from __future__ import annotations

import re

# Qwen/GPT-2 byte-level BPE renders a space as "Ġ" and a newline as "Ċ".
_SPACE = "\u0120"
_NEWLINE = "\u010a"


def _plain(piece: str) -> str:
    return piece.replace(_SPACE, " ").replace(_NEWLINE, "\n")


def report_span_start(token_strings: list[str]) -> int:
    """First token index of the report body (0 when it cannot be located)."""
    if not token_strings:
        return 0
    full = "".join(_plain(piece) for piece in token_strings)
    last_option = full.rfind("D. ")
    if last_option < 0:
        return 0
    # The report starts after the blank line that closes the option list.
    breaks = [match.end() for match in re.finditer(r"\n\n", full)]
    start_char = next((end for end in breaks if end > last_option), None)
    if start_char is None:
        return 0
    consumed = 0
    for index, piece in enumerate(token_strings):
        consumed += len(_plain(piece))
        if consumed >= start_char:
            return index
    return 0


def body_mask(num_tokens: int, token_strings: list[str]) -> list[bool]:
    """``True`` for tokens that belong to the report body."""
    start = report_span_start(token_strings)
    return [index >= start for index in range(num_tokens)]
