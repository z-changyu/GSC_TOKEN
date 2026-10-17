"""One-sample pragmatic scoring, shared by the pilot and the cross-validation driver.

The pipeline is the operational reading of Section III:

1. the transmitter turns ``(image, question, choices)`` into a semantic prompt ``P``;
2. Eq. (2) is built by teacher forcing the ground-truth answer after
   ``[user: I, Q][assistant: P][user: request]`` and the input embeddings of the
   ``P`` slice are the differentiation target;
3. ``A_i`` (Eq. 10), ``S_i`` (Eq. 11), ``G_i`` (Eq. 14) and ``M_ij`` (Eq. 16) are
   read from that single forward pass.

The per-component scores are stored separately so that the pragmatic weights can
be re-calibrated without recomputing any model forward pass, which is what makes
fold-wise cross-validation affordable.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import contextlib

import torch
from PIL import Image


@contextlib.contextmanager
def _stage(collector: dict[str, float] | None, name: str):
    """Accumulate the wall time of one scoring block, GPU-synchronised.

    Timings are collected only when a collector is passed (i.e. when
    ``PragmaticScorer.collect_timings`` is enabled), so the production path pays
    nothing for the instrumentation.
    """
    if collector is None:
        yield
        return
    synchronise = torch.cuda.is_available()
    if synchronise:
        torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        yield
    finally:
        if synchronise:
            torch.cuda.synchronize()
        collector[name] = collector.get(name, 0.0) + (time.perf_counter() - start)


def _to_list(value) -> list[float]:
    """Batched GPU->CPU conversion: one transfer instead of N scalar transfers."""
    if hasattr(value, "detach"):
        return value.detach().to("cpu").float().tolist()
    return [float(v) for v in value]


def _to_nested_list(value) -> list[list[float]] | None:
    """Same as :func:`_to_list` for the coupling matrix (N^2 entries)."""
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().to("cpu").float().tolist()
    return [[float(v) for v in row] for row in value]


@dataclass
class SampleScores:
    sample_id: str
    token_ids: list[int]
    token_strings: list[str]
    semantic_prompt: str
    attention: list[float]
    semantic: list[float]
    gradient: list[float]
    coupling: list[list[float]] | None
    answer_nll: float
    transmit_seconds: float
    score_seconds: float
    scoring_peak_memory_gib: float
    sequence_length: int
    prompt_meta_leak: bool
    prompt_truncated: bool
    coupling_kappa: float | None = None
    coupling_d_M: float | None = None
    # ---- provenance of the teacher-forcing target (added with the predicted-target
    # revision). ``score_target`` says which branch produced this row; the two
    # branches must never be mixed inside one cache (guarded by ``fingerprint``).
    score_target: str = "ground_truth"      # "predicted" | "ground_truth"
    predicted_choice: int | None = None     # terminal's own choice index
    predicted_text: str | None = None       # ... and its option text
    prediction_scores: list[float] | None = None
    coupling_enabled: bool = False
    model_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> dict[str, Any]:
        """Identity of the scoring configuration this row was produced under.

        ``build_score_cache.py`` refuses to resume a cache whose fingerprint differs,
        because a mixed cache silently blends two different scoring targets.
        """
        return {"score_target": self.score_target,
                "coupling_enabled": bool(self.coupling_enabled),
                "model_id": self.model_id,
                "mask_token": (self.metadata or {}).get("mask_token"),
                "max_prompt_tokens": (self.metadata or {}).get("max_prompt_tokens"),
                "layer_reduction": (self.metadata or {}).get("layer_reduction")}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(row: dict[str, Any]) -> "SampleScores":
        return SampleScores(**row)

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)


def combine_scores(attention, semantic, gradient, weights) -> torch.Tensor:
    """Eq. (15) after per-sample min-max normalisation.

    Kept as a free function so the calibration step can re-weight cached
    components without touching a model.
    """
    from ..pragmatic.utility import UnifiedUtilityScorer

    token_ids = list(range(len(attention)))
    scorer = UnifiedUtilityScorer(*weights)
    result = scorer.score(
        _wrap(attention, token_ids), _wrap(semantic, token_ids), _wrap(gradient, token_ids))
    return result.scores.detach()


def _wrap(values, token_ids):
    from ..pragmatic.utility import ScoreResult

    tensor = torch.as_tensor(values, dtype=torch.float32)
    return ScoreResult(tensor, token_ids, [str(i) for i in token_ids])


class PragmaticScorer:
    """Scores one ScienceQA sample with a loaded Qwen3-VL transmitter."""

    def __init__(self, transmitter, processor, config: dict) -> None:
        self.transmitter = transmitter
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.config = config
        # Optional per-sub-stage timing, used by scripts/measure_stage_costs.py to
        # report the cost of every scoring block (teacher-forced forward, gradient
        # backward, attention aggregation, coupling construction). Disabled by
        # default so the experimental pipeline is untouched.
        self.collect_timings = False
        self.timings: dict[str, float] = {}
        self.im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.max_prompt_tokens = int(config["transmitter"]["max_prompt_tokens"])
        self.max_new_tokens = int(config["transmitter"]["max_new_tokens"])

        from ..pragmatic.attention_score import AttentionStructuralScorer
        from ..pragmatic.coupling import SemanticCouplingScorer
        from ..pragmatic.gradient_score import GradientScorer
        from ..pragmatic.semantic_score import SemanticDirectionalScorer
        from ..pragmatic.utility import coupling_regime

        reduction = config["pragmatic"].get("layer_reduction", "mean")
        self.layer_reduction = reduction
        self.attention_scorer = AttentionStructuralScorer(reduction)
        self.semantic_scorer = SemanticDirectionalScorer()
        self.coupling_scorer = SemanticCouplingScorer(reduction)
        self.gradient_scorer = GradientScorer("eq2_answer_nll_given_I_Q_P",
                                              "ground_truth_answer")
        self.coupling_regime = coupling_regime
        self.coupling_enabled = bool(config["pragmatic"].get("coupling_enabled", False))
        self.beta = float(config["pragmatic"]["beta"])
        # Eq. (14)'s teacher-forcing target. "predicted" is the revised method: the
        # terminal answers its own full input and that choice becomes the target, so
        # no test label is read. "ground_truth" is kept only as an offline upper bound.
        self.score_target = str(config["pragmatic"].get("score_target", "predicted"))
        if self.score_target not in ("predicted", "ground_truth"):
            raise SystemExit(f"unknown pragmatic.score_target {self.score_target!r}")
        self.model_id = str(getattr(transmitter.config, "_name_or_path", "unknown"))

        mask_token_id = self.tokenizer.convert_tokens_to_ids(config["pragmatic"]["mask_token"])
        if mask_token_id is None or mask_token_id < 0:
            raise SystemExit(f"unknown mask token {config['pragmatic']['mask_token']!r}")
        with torch.no_grad():
            self.mask_embedding = transmitter.get_input_embeddings()(
                torch.tensor([mask_token_id], device="cuda:0"))[0]

        # Only the prompt-token embeddings are differentiated, so the vision tower
        # is frozen: it saves activation memory and builds no unused graph.
        vision = transmitter.model.get_image_features

        @torch.no_grad()
        def frozen_vision(pixel_values, image_grid_thw):
            return vision(pixel_values, image_grid_thw)

        transmitter.model.get_image_features = frozen_vision

    # ---------------------------------------------------------------- transmit
    def generate_prompt(self, sample: dict, image) -> str:
        from .prompts import build_transmitter_messages

        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": sample["_request"]})
        # The report generation is the parent stage in the paper's cost model, so its
        # own sub-stages are timed too: the processor pass over the image and the
        # decode/allocator release used to fall between the instrumented blocks and
        # were the bulk of the parent/child residual.
        stage = self.timings if self.collect_timings else None
        with _stage(stage, "report_prepare"):
            inputs = self.processor.apply_chat_template(
                build_transmitter_messages(content), tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt")
            inputs = {k: (v.to("cuda:0") if hasattr(v, "to") else v) for k, v in inputs.items()}
        # The scorer regenerates the report internally, so the cost pass needs this
        # block timed separately: otherwise it is hidden inside "scoring" and the
        # pipeline would look as if it paid for the report generation twice.
        with _stage(stage, "report"):
            with torch.no_grad():
                generated = self.transmitter.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                                      do_sample=False)
        with _stage(stage, "report_decode"):
            text = self.tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:],
                                         skip_special_tokens=True).strip()
            del generated, inputs
            torch.cuda.empty_cache()
        return text

    # ---------------------------------------------------------- self prediction
    def self_predict(self, sample: dict, image) -> dict[str, Any]:
        """The terminal's own choice for its *full* input, without reading any label.

        Read off a single forward pass over the option-letter logits (the same
        deterministic readout the receiver uses), so it costs one prefill instead of
        a decoding loop and always yields an index -- there is no "unparseable"
        branch that would tempt a fallback to the ground truth.
        """
        from .prompts import TRANSMITTER_SELF_QUERY, build_transmitter_messages
        from .receiver_protocol import letter_token_ids

        choices = list(sample["choices"])
        request = TRANSMITTER_SELF_QUERY.format(
            question=sample["question"], choices="\n".join(
                f"{chr(65 + i)}. {c}" for i, c in enumerate(choices)))
        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": request})
        inputs = self.processor.apply_chat_template(
            build_transmitter_messages(content), tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt")
        inputs = {k: (v.to("cuda:0") if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.transmitter(**inputs).logits[0, -1]
        groups = letter_token_ids(self.tokenizer)
        scores = [float(logits[ids].max()) if ids else float("-inf") for ids in groups]
        index = int(max(range(min(len(scores), len(choices))),
                        key=lambda i: scores[i]))
        del inputs, logits
        torch.cuda.empty_cache()
        return {"index": index, "text": choices[index], "letter": "ABCD"[index],
                "scores": scores, "parse_mode": "logit_letter"}

    # ------------------------------------------------------------------ score
    def score(self, sample: dict, image) -> SampleScores:
        from .prompts import build_scoring_messages, locate_span

        start = time.perf_counter()
        semantic_prompt = self.generate_prompt(sample, image)
        transmit_seconds = time.perf_counter() - start

        # Teacher-forcing target of Eq. (14). The revised method never touches the
        # test label: the target is the terminal's own choice on its full input.
        if self.score_target == "predicted":
            prediction = self.self_predict(sample, image)
            target_text = prediction["text"]
        else:
            prediction = None
            target_text = str(sample["choices"][int(sample["answer"])])

        with _stage(self.timings if self.collect_timings else None, "tokenize"):
            token_ids = self.tokenizer(semantic_prompt, add_special_tokens=False)["input_ids"]
            truncated = len(token_ids) > self.max_prompt_tokens
            if truncated:
                token_ids = token_ids[:self.max_prompt_tokens]
                semantic_prompt = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                token_ids = self.tokenizer(semantic_prompt, add_special_tokens=False)["input_ids"]
            if not token_ids:
                raise RuntimeError(f"empty prompt for sample {sample.get('row_idx')}")
            token_strings = self.tokenizer.convert_ids_to_tokens(token_ids)

        # Scoring input preparation: two processor passes over the image (the full
        # scoring prompt and the image-only hint used to locate the prompt span).
        # Without this stage the parent/child residual of the scoring total stays
        # unexplained, because the image work happens before the embedding block.
        timings = self.timings if self.collect_timings else None
        with _stage(timings, "prepare"):
            start = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            messages = build_scoring_messages(image, sample["question"], sample["choices"],
                                              semantic_prompt)
            batch = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True,
                return_tensors="pt")
            expected = self.tokenizer(semantic_prompt, add_special_tokens=False)["input_ids"]
            sequence_list = batch["input_ids"][0].tolist()
            hint = len(self.processor.apply_chat_template(
                messages[:1], tokenize=True, add_generation_prompt=True, return_dict=True,
                return_tensors="pt")["input_ids"][0])
            p_start = locate_span(sequence_list, expected, hint)
            if p_start < 0:
                raise RuntimeError(f"cannot locate the prompt span for sample {sample.get('row_idx')}")
            p_end = p_start + len(expected)

        # Prompt-to-device transfer and label construction. Timed explicitly: this
        # block used to sit between "prepare" and "assemble" uninstrumented and was
        # the bulk of the parent/child residual.
        with _stage(timings, "payload_transfer"):
            sequence_ids = batch["input_ids"].to("cuda:0")
        attention_mask = batch["attention_mask"].to("cuda:0")
        answer_ids = self.tokenizer(target_text,
                                    add_special_tokens=False)["input_ids"] + [self.im_end_id]
        tail = torch.tensor([answer_ids], device="cuda:0")
        input_ids = torch.cat([sequence_ids, tail], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(tail)], dim=1)
        labels = torch.full_like(input_ids, -100)
        labels[0, -len(answer_ids):] = tail[0]

        # The collector is owned by the caller (scripts/measure_stage_costs.py),
        # which resets it before each sample: clearing it here would also wipe the
        # report generation that ``score()`` performs above this point.
        timings = self.timings if self.collect_timings else None

        with _stage(timings, "assemble"):
            pixel_values = batch["pixel_values"].to("cuda:0") if "pixel_values" in batch else None
            image_grid_thw = batch["image_grid_thw"].to("cuda:0") if "image_grid_thw" in batch else None
            position_ids, _ = self.transmitter.model.get_rope_index(
                input_ids, image_grid_thw, None, attention_mask=attention_mask)
            embeddings = self.transmitter.get_input_embeddings()(input_ids).detach().to(torch.bfloat16)
            embeddings.requires_grad_(True)
        with _stage(timings, "forward"):
            output = self.transmitter(inputs_embeds=embeddings, attention_mask=attention_mask,
                                      position_ids=position_ids, pixel_values=pixel_values,
                                      image_grid_thw=image_grid_thw, labels=labels,
                                      output_attentions=True, return_dict=True)

        prompt_slice = slice(p_start, p_end)
        with _stage(timings, "attention_aggregate"):
            prompt_attentions = tuple(layer[:, :, prompt_slice, prompt_slice]
                                      for layer in output.attentions)
            attention = self.attention_scorer.score(prompt_attentions, token_ids, token_strings)
            semantic = self.semantic_scorer.score(embeddings.detach()[0, prompt_slice].float(),
                                                  attention.scores.float(), token_ids, token_strings)
        with _stage(timings, "backward"):
            gradient = self.gradient_scorer.score(output.loss, embeddings, token_ids, token_strings,
                                                  mask_embedding=self.mask_embedding,
                                                  positions=prompt_slice)
        with _stage(timings, "coupling"):
            coupling = self.coupling_scorer.score(prompt_attentions) if self.coupling_enabled else None
            regime = self.coupling_regime(coupling, self.beta) if coupling is not None else {}
        # Result serialisation and cache release, split into three stages so the
        # per-element GPU->CPU conversions can be separated from the allocator
        # flush. The official path is the batched transfer; the legacy per-element
        # conversion is re-run only as a diagnostic comparison (its result is
        # discarded) and is excluded from the reported scoring total.
        with _stage(timings, "nll_sync"):
            answer_nll = float(output.loss.detach())
        with _stage(timings, "serialize"):
            scores = SampleScores(
                sample_id=str(sample.get("row_idx")),
                token_ids=token_ids, token_strings=token_strings,
                semantic_prompt=semantic_prompt,
                attention=_to_list(attention.scores),
                semantic=_to_list(semantic.scores),
                gradient=_to_list(gradient.scores),
                coupling=_to_nested_list(coupling),
                answer_nll=answer_nll,
                transmit_seconds=transmit_seconds,
                score_seconds=time.perf_counter() - start,
                scoring_peak_memory_gib=torch.cuda.max_memory_allocated() / 2 ** 30,
                sequence_length=int(input_ids.shape[1]),
                prompt_meta_leak=False,
                prompt_truncated=truncated,
                coupling_kappa=regime.get("kappa"),
                coupling_d_M=regime.get("d_M"),
                score_target=self.score_target,
                predicted_choice=prediction["index"] if prediction else None,
                predicted_text=prediction["text"] if prediction else None,
                prediction_scores=prediction["scores"] if prediction else None,
                coupling_enabled=self.coupling_enabled,
                model_id=self.model_id,
                metadata={"mask_token": str(self.config["pragmatic"]["mask_token"]),
                          "max_prompt_tokens": self.max_prompt_tokens,
                          "max_new_tokens": self.max_new_tokens,
                          "layer_reduction": str(self.layer_reduction),
                          "beta": float(self.beta)})
        with _stage(timings, "cache_release"):
            del output, embeddings, batch, input_ids, labels, position_ids
            torch.cuda.empty_cache()
        if self.collect_timings:
            with _stage(timings, "serialize_legacy"):
                legacy = ([float(v) for v in attention.scores],
                          [float(v) for v in semantic.scores],
                          [float(v) for v in gradient.scores],
                          [[float(v) for v in row] for row in coupling]
                          if coupling is not None else None)
                del legacy
        return scores

    def pruned_prompt_nll(self, sample: dict, image, prompt_text: str,
                          target_text: str | None = None) -> float:
        """Eq. (21)-(23): transmitter-side NLL of the answer given a pruned prompt.

        This is the calibration target for the pragmatic weights: the receiver is
        not involved, so a grid search costs one forward per candidate rather
        than one generation per candidate.

        ``target_text`` defaults to the ground-truth option because this quantity is
        only ever consumed offline (an upper-bound diagnostic). Callers that must not
        touch the label should pass the terminal's own prediction instead.
        """
        from .prompts import build_scoring_messages

        messages = build_scoring_messages(image, sample["question"], sample["choices"],
                                          prompt_text)
        batch = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt")
        sequence_ids = batch["input_ids"].to("cuda:0")
        attention_mask = batch["attention_mask"].to("cuda:0")
        if target_text is None:
            target_text = str(sample["choices"][int(sample["answer"])])
        answer_ids = self.tokenizer(target_text,
                                    add_special_tokens=False)["input_ids"] + [self.im_end_id]
        tail = torch.tensor([answer_ids], device="cuda:0")
        input_ids = torch.cat([sequence_ids, tail], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(tail)], dim=1)
        labels = torch.full_like(input_ids, -100)
        labels[0, -len(answer_ids):] = tail[0]
        pixel_values = batch["pixel_values"].to("cuda:0") if "pixel_values" in batch else None
        image_grid_thw = batch["image_grid_thw"].to("cuda:0") if "image_grid_thw" in batch else None
        position_ids, _ = self.transmitter.model.get_rope_index(
            input_ids, image_grid_thw, None, attention_mask=attention_mask)
        with torch.no_grad():
            output = self.transmitter(input_ids=input_ids, attention_mask=attention_mask,
                                      position_ids=position_ids, pixel_values=pixel_values,
                                      image_grid_thw=image_grid_thw, labels=labels,
                                      return_dict=True)
        loss = float(output.loss.detach())
        del output, batch, input_ids, labels, position_ids
        torch.cuda.empty_cache()
        return loss


def load_image(path):
    return Image.open(path).convert("RGB") if path else None
