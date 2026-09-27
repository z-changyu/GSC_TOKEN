"""Receiver-side answering protocols, shared by every experiment driver.

The channel experiments support two answering protocols:

``v0``
    The original protocol: the receiver sees only the surviving description
    text and the answer request, and must reply with the exact option text.

``v6``
    Task-intent protocol applied identically to every method:
    1. the RSU-issued query (question + candidate choices) is known to both
       ends, so it is placed in the prompt instead of being re-transmitted;
    2. a payload that lost at least ``1 - RESTORE_THRESHOLD`` of its tokens is
       repaired by the receiver first (restore-then-answer);
    3. the answer is constrained to the option letter, which removes the
       unparseable-output failure mode.

Keeping this logic in one module guarantees that the main channel experiment
and the Pareto/operating-point experiments answer under byte-identical
prompts; any divergence would make their accuracies incomparable.
"""
from __future__ import annotations

from typing import Any, Sequence

import torch

# A payload is repaired when the fraction of surviving tokens drops below this.
RESTORE_THRESHOLD = 0.6
# Restoration is a full-report rewrite, so it needs a report-sized budget.
RESTORE_MAX_NEW_TOKENS = 200

LETTER_ONLY_INSTRUCTION = (
    "Reply with only the option letter (A, B, C or D), with no other text.")


def _model_device(model) -> torch.device | str:
    """Device to place receiver inputs on.

    The answering prompts live next to the receiver, which is not necessarily on
    ``cuda:0``: drivers that need the transmitter and the receiver in memory at the
    same time (e.g. the cross-view fusion experiment with a 4B transmitter plus an
    8B coordinator) place the receiver on a second device. Following the model's
    own device keeps single-GPU behaviour identical to the historical ``cuda:0``.
    """
    device = getattr(model, "device", None)
    if device is None:
        try:
            device = next(model.parameters()).device
        except (AttributeError, StopIteration):
            device = torch.device("cuda:0")
    return device

RESTORE_INSTRUCTION = (
    "The semantic description below was partially destroyed by a noisy radio "
    "link. Using the question and the candidate choices, rewrite the most "
    "plausible complete description of the scene. Keep the same wording wherever "
    "it survives; fill gaps with the most likely continuation. Output the "
    "restored description only.")

COORDINATOR_LEAD_IN = (
    "A text-only coordinator sends the following semantic description of the scene:")


# --------------------------------------------------------------------- prompts
def build_question_block(row: dict) -> str:
    """Question plus lettered candidate choices, as issued by the RSU."""
    choices = "\n".join(f"{chr(65 + i)}. {choice}"
                        for i, choice in enumerate(row.get("choices") or []))
    return f"Question: {row['question']}\n\nCandidate choices:\n{choices}"


def build_v0_answer_content(received: str, answer_request: str) -> str:
    """Original protocol: description only."""
    return f"{answer_request}\n\n{received}"


def build_restore_prompt(question_block: str, received: str) -> str:
    return (f"{question_block}\n\nDamaged description:\n{received}\n\n"
            f"{RESTORE_INSTRUCTION}")


def build_v6_answer_content(question_block: str, received: str,
                            answer_request: str) -> str:
    return (f"{question_block}\n\n{COORDINATOR_LEAD_IN}\n{received}\n\n"
            f"{answer_request}\n\n{LETTER_ONLY_INSTRUCTION}")


# -------------------------------------------------------- discriminative answer
# ``v6_letter`` answers with a *single forward pass*: the option letter is read
# off the output layer instead of being decoded token by token. The prompt is the
# same as ``v6`` (task intent + letter-only instruction), so the two protocols
# differ only in how the answer is read out -- which isolates the latency and
# accuracy effect of the output format in the warning-latency study.
OPTION_LETTERS = "ABCD"


def letter_token_ids(tokenizer) -> list[list[int]]:
    """Single-token spellings of every option letter, with and without a space."""
    groups: list[list[int]] = []
    for letter in OPTION_LETTERS:
        ids = set()
        for form in (letter, " " + letter):
            encoded = tokenizer.encode(form, add_special_tokens=False)
            if len(encoded) == 1:                  # multi-token spellings are unusable
                ids.add(int(encoded[0]))
        groups.append(sorted(ids))
    return groups


@torch.no_grad()
def answer_letter_only(*, model, tokenizer, row: dict, received: str,
                       answer_request: str, question_block: str) -> dict[str, Any]:
    """One forward pass + argmax over the four option letters (no decoding loop)."""
    content = build_v6_answer_content(question_block, received, answer_request)
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(chat, return_tensors="pt").to(_model_device(model))
    logits = model(**inputs).logits[0, -1]
    groups = letter_token_ids(tokenizer)
    scores = [float(logits[ids].max()) if ids else float("-inf") for ids in groups]
    choices = row.get("choices") or []
    index = int(max(range(len(scores)), key=lambda i: scores[i]))
    prediction = choices[index] if index < len(choices) else OPTION_LETTERS[index]
    del inputs, logits
    return {"raw": OPTION_LETTERS[index], "prediction": prediction,
            "parse_mode": "logit_letter", "letter_scores": scores,
            "restored": False, "received_after": received, "answer_content": content}


# ------------------------------------------------------------------ generation
def generate_text(model, tokenizer, content: str, max_new_tokens: int) -> str:
    """Greedy chat completion for one user turn."""
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(chat, return_tensors="pt").to(_model_device(model))
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                do_sample=False)
    text = tokenizer.decode(output[0, inputs.input_ids.shape[1]:],
                            skip_special_tokens=True).strip()
    del inputs, output
    return text


# ------------------------------------------------------------------- protocol
def answer_with_protocol(*, model, tokenizer, row: dict, received: str,
                         survived_tokens: int, total_tokens: int,
                         answer_request: str, max_new_tokens: int,
                         protocol: str = "v6",
                         restore_threshold: float = RESTORE_THRESHOLD) -> dict[str, Any]:
    """Answer one run under ``protocol``; returns the raw text and the parse.

    The returned ``received_after`` is the text actually used for answering
    (post-restoration when the restoration run ran).
    """
    from .prompts import canonical_prediction

    question_block = build_question_block(row)
    restored = False
    if protocol == "v6_letter":
        return answer_letter_only(model=model, tokenizer=tokenizer, row=row,
                                  received=received, answer_request=answer_request,
                                  question_block=question_block)
    if protocol == "v6":
        fraction = survived_tokens / max(1, total_tokens)
        if fraction < restore_threshold:
            received = generate_text(model, tokenizer,
                                     build_restore_prompt(question_block, received),
                                     RESTORE_MAX_NEW_TOKENS)
            restored = True
        content = build_v6_answer_content(question_block, received, answer_request)
    elif protocol == "v0":
        content = build_v0_answer_content(received, answer_request)
    else:
        raise ValueError(f"unknown receiver protocol {protocol!r}")

    raw = generate_text(model, tokenizer, content, max_new_tokens)
    prediction, parse_mode = canonical_prediction(raw, row.get("choices"))
    return {"raw": raw, "prediction": prediction, "parse_mode": parse_mode,
            "restored": restored, "received_after": received,
            "answer_content": content}
