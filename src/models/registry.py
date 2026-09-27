from __future__ import annotations

from typing import Any


def resolve_dtype(name: str):
    import torch
    aliases = {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
               "float16": torch.float16, "fp16": torch.float16, "float32": torch.float32}
    try: return aliases[name.lower()]
    except KeyError as exc: raise ValueError(f"Unsupported dtype {name}") from exc


def load_hf_model(model_name: str, *, dtype: str, model_kind: str,
                  attention_implementation: str = "eager", **kwargs: Any):
    from transformers import AutoModelForCausalLM
    common = {"torch_dtype": resolve_dtype(dtype), "attn_implementation": attention_implementation, **kwargs}
    if model_kind == "causal_lm": return AutoModelForCausalLM.from_pretrained(model_name, **common)
    if model_kind == "vision_text_generation":
        try:
            from transformers import Qwen3VLForConditionalGeneration
            return Qwen3VLForConditionalGeneration.from_pretrained(model_name, **common)
        except ImportError:
            from transformers import AutoModelForImageTextToText
            return AutoModelForImageTextToText.from_pretrained(model_name, **common)
    raise ValueError(f"Unknown model_kind={model_kind}")


def load_generation_receiver(model_name: str, *, dtype: str = "bfloat16",
                             device: str | None = "cuda:0",
                             attention_implementation: str = "eager",
                             local_files_only: bool = True,
                             **kwargs: Any):
    """Load a decoder used as the answering receiver.

    Text-only checkpoints load as causal LMs; vision-language checkpoints driven
    in text-only mode (e.g. ``Qwen3-VL-8B-Instruct``, the edge coordinator of the
    4B->8B main line) are rejected by ``AutoModelForCausalLM`` and must go through
    the image-text-to-text class instead. Memory from the failed attempt is
    released before the fallback is tried, mirroring the receiver loader in
    ``scripts/run_channel_experiment.py``.
    """
    import gc

    import torch
    from transformers import AutoModelForCausalLM

    common = {"local_files_only": local_files_only,
              "attn_implementation": attention_implementation,
              "dtype": resolve_dtype(dtype), **kwargs}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **common)
    except (ValueError, KeyError, RuntimeError, TypeError):
        gc.collect()
        torch.cuda.empty_cache()
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_name, **common)
    if device:
        try:
            model = model.to(device)
        except (ValueError, RuntimeError):
            pass
    return model.eval()
