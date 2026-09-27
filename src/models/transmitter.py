from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TransmitterOutput:
    text: str
    token_ids: list[int]
    hidden_states: Any = None
    attentions: Any = None


class Transmitter:
    def __init__(self, model, processor) -> None:
        self.model, self.processor = model, processor

    def encode(self, image, question: str, **generation_kwargs) -> TransmitterOutput:
        inputs = self.processor(images=image, text=question, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, return_dict_in_generate=True,
                                     output_hidden_states=True, output_attentions=True, **generation_kwargs)
        sequence = output.sequences[0]
        return TransmitterOutput(self.processor.decode(sequence, skip_special_tokens=True), sequence.tolist(),
                                 getattr(output, "hidden_states", None), getattr(output, "attentions", None))

