"""Locate the report body inside the scored payload.

The payload that gets compressed is ``question + option list + report``, but the
receiver builds its own prompt from ``build_question_block`` and therefore already
holds the question and the options. Everything before the report body is
duplicated: it costs transmit symbols and returns no information.

Measurement on 20 V2X-QA samples: the duplicated region is 30% of the payload but
absorbs 44% of the top-10% pragmatic utility, so the scheduler spends a
disproportionate share of the budget on it. ``report_span_start`` gives the token
index where the report body begins so callers can exclude the prefix.

The locator anchors on the *last* option line ("D. <text>", or "C. <text>" for a
three-option item) and skips the blank line that closes the list. An earlier
version searched for the literal "D. ", which failed on 32% of the manifest --
every sample whose option D text itself contains "D. ", whose item has fewer than
four options, or whose newline layout differs -- silently disabling the mask on
those samples.
"""
from __future__ import annotations

# Qwen/GPT-2 byte-level BPE renders a space as "Ġ" and a newline as "Ċ".
_SPACE = "\u0120"
_NEWLINE = "\u010a"


def _plain(piece: str) -> str:
    return piece.replace(_SPACE, " ").replace(_NEWLINE, "\n")


def _char_to_token(token_strings: list[str], char_index: int) -> int:
    consumed = 0
    for index, piece in enumerate(token_strings):
        consumed += len(_plain(piece))
        if consumed >= char_index:
            return index
    return 0


def report_span_start(token_strings: list[str], choices: list[str] | None = None) -> int:
    """First token index of the report body (0 when it cannot be located)."""
    if not token_strings:
        return 0
    full = "".join(_plain(piece) for piece in token_strings)

    end_char = None
    if choices:
        letter = chr(64 + len(choices))
        marker = f"{letter}. {choices[-1]}"
        position = full.rfind(marker)
        if position >= 0:
            end_char = position + len(marker)
    if end_char is None:
        # Fallback: the last option line of any letter, then the blank line after it.
        for letter in "DCBA":
            position = full.rfind(f"\n{letter}. ")
            if position >= 0:
                end_char = position + 1
                while end_char < len(full) and full[end_char] != "\n":
                    end_char += 1
                break
    if end_char is None:
        return 0
    while end_char < len(full) and full[end_char] in " \n":
        end_char += 1
    if end_char >= len(full):
        return 0
    return _char_to_token(token_strings, end_char)


def body_mask(num_tokens: int, token_strings: list[str],
              choices: list[str] | None = None) -> list[bool]:
    """``True`` for tokens that belong to the report body."""
    start = report_span_start(token_strings, choices)
    return [index >= start for index in range(num_tokens)]
