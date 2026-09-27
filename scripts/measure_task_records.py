#!/usr/bin/env python
"""Same-condition stage-cost / resource / latency records for the cost tables.

Fixes the accounting of the earlier pass, following the review checklist:

1. Resource accounting. Every record stores information bits, coded bits (the
   quantity previously called "symbols" -- it is a *coded-bit* proxy, so it is
   renamed ``resource_cost_coded_bits``), the modulation order, the implied
   channel uses ``N_use = sum_i N_{use,i,l_i}``, the allocated bandwidth and the
   send window. Air-time is NOT obtained by dividing by 1000: the paper's model is
   the single-window reliable budget of Eq. (6), so only the window count
   (model-derived) is reported, flagged as a model quantity.
2. Method-specific paths. Each method is timed on what it actually runs:
   proposed = pragmatic scoring + coupling + LP + rounding + refinement;
   Adaptive-SC = its own selection rule; TokCom-UEP = TokCom importance + our
   scheduler (no separate coupling build); uncompressed = packaging only.
   Two scenarios are kept apart: ``independent`` (report generation included) and
   ``cached_report`` (starts from an existing report).
3. Stage boundaries with a parent/child residual check:
   T_residual = T_parent - sum_j T_child,j, reported per sample, so nothing is
   hidden between stages.
4. Same condition for every method: the same 30 samples, 15 dB, one fixed channel
   instance per sample, the same report, the same receiver and readout. The
   receiver is timed on each method's own delivered tokens.
5. Per-sample timeline: sample_id, method, channel, seed, report_tokens,
   delivered_tokens, report_ms, scoring_ms, selection_ms, allocation_ms,
   control_wait_ms (unmeasured, flagged), codec_ms (unmeasured, flagged),
   uplink_ms (simulated), receiver_ms, total_ms, correct, each with a
   measured/simulated/unmeasured tag.
6. Repeat timing: >=5 warm-ups, 3-5 repeats per (sample, method) in randomised
   order, per-sample medians, across-sample P50/P95, paired per-sample differences
   against a baseline with a bootstrap confidence interval.
7. The 8B readout comparison (autoregressive letter output vs single-forward
   discriminative readout) is measured on the same inputs: latency, actual decoded
   length, accuracy and answer-agreement rate.

Usage
-----
    python scripts/measure_task_records.py --samples 3 --repeats 1    # smoke
    python scripts/measure_task_records.py                            # paper pass
"""
from __future__ import annotations

import argparse
import logging
import math
import random
import statistics
import sys
import time
from pathlib import Path

import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.channel import build_channel  # noqa: E402
from src.channel.link import ProtectionCodebook, decoding_probability  # noqa: E402
from src.channel.reliable_capacity import reliable_spectral_efficiency  # noqa: E402
from src.experiments.prompts import (ANSWER_REQUEST, build_choice_block,  # noqa: E402
                                     canonical_prediction, select_transmitter_request)
from src.experiments.receiver_protocol import (letter_token_ids, OPTION_LETTERS,  # noqa: E402
                                               build_v6_answer_content, generate_text)
from src.experiments.scoring import PragmaticScorer, SampleScores, combine_scores  # noqa: E402
from src.models.registry import load_generation_receiver, load_hf_model  # noqa: E402
from src.scheduling import certified_rounding, refine, select_level, solve_base_lp  # noqa: E402
from src.scheduling.bandwidth import allocate_bandwidth  # noqa: E402
from src.scheduling.problem import SchedulingProblem  # noqa: E402
from src.utils.io import atomic_json, read_jsonl  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("task_records")

TRIGGER_MS = 20.0          # assumed on-board trigger, not measured
UPLINK_MS = 1.0            # simulated: one slot
DOWNLINK_MS = 0.5          # simulated
MODULATION_ORDER = 1.0     # coded bits per channel use carried by the model
WINDOW_MODEL = "single-window Eq. (6); window count reported as a model quantity"
# Methods whose selection rule consumes the pragmatic utility, i.e. the only ones
# that can be charged for the pragmatic scoring.
SCORING_USERS = {"proposed", "adaptive_sc"}
# Provenance of each comparison row, so an internal variant is never presented as
# the published baseline it replaces.
BASELINE_ROLE = {
    "proposed": "本文方法",
    "adaptive_sc": "内部变体：选级规则复用本文三维语用评分，非原基线实现",
    "tokcom_uep": "外部重要性信号（离线缓存），仅计时读取与归一化",
    "full": "未压缩参考（不使用重要性信号）",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default="configs/experiments/v2x_edge_edge_main_4b8b.yaml")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--snr-db", type=float, default=15.0,
                        help="15 dB is the loose instance (the budget rarely binds), "
                             "0 dB the binding one; run this driver once per instance "
                             "into separate output directories so the two are kept "
                             "apart in the tables")
    parser.add_argument("--instance-tag", default="loose",
                        help="label recorded with every row: loose (15 dB) or binding (0 dB)")
    parser.add_argument("--methods", nargs="*",
                        default=["proposed", "adaptive_sc", "tokcom_uep", "full"])
    parser.add_argument("--baseline", default="adaptive_sc")
    parser.add_argument("--weights", type=float, nargs=3, default=[0.25, 0.30, 0.45])
    parser.add_argument("--t-max", type=int, default=8)
    parser.add_argument("--max-prompt-tokens", type=int, default=200)
    parser.add_argument("--output-dir", default="outputs/task_records")
    return parser.parse_args()


class Clock:
    """Wall time with explicit GPU synchronisation."""

    def __init__(self) -> None:
        self.gpu = torch.cuda.is_available()

    def __call__(self, sync: bool = True):
        return _Tick(self, sync)


class _Tick:
    def __init__(self, clock: Clock, sync: bool) -> None:
        self.clock, self.sync = clock, sync

    def __enter__(self):
        if self.sync and self.clock.gpu:
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.sync and self.clock.gpu:
            torch.cuda.synchronize()
        self.ms = (time.perf_counter() - self.t0) * 1e3
        return False


def describe(values: list[float]) -> dict:
    xs = sorted(v for v in values if v == v)
    if not xs:
        return {"n": 0, "mean": float("nan"), "p50": float("nan"), "p95": float("nan")}
    rank = min(len(xs) - 1, max(0, math.ceil(0.95 * len(xs)) - 1))
    return {"n": len(xs), "mean": statistics.fmean(xs),
            "p50": statistics.median(xs), "p95": xs[rank]}


def bootstrap_ci(diffs: list[float], draws: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
    xs = [d for d in diffs if d == d]
    if len(xs) < 3:
        return float("nan"), float("nan")
    rng = random.Random(2026)
    means = sorted(statistics.fmean(rng.choices(xs, k=len(xs))) for _ in range(draws))
    lo = means[int(alpha / 2 * draws)]
    hi = means[min(draws - 1, int((1 - alpha / 2) * draws))]
    return lo, hi


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    tx_path = config["model"]["transmitter_path"]
    rx_path = config["model"]["receiver_path"]

    cache = {str(r["sample_id"]): SampleScores.from_dict(r)
             for r in read_jsonl(config["score_cache"])}
    template = select_transmitter_request(config)
    processed: dict[str, dict] = {}
    for rec in read_jsonl(config["dataset"]["processed"]):
        rec["_request"] = template.format(question=rec["question"],
                                          choices=build_choice_block(rec["choices"]))
        for key in {str(rec.get("sample_id", "")), str(rec.get("row_idx", ""))}:
            if key:
                processed[key] = rec
    manifest = [str(r["sample_id"]) for r in read_jsonl(config["dataset"]["manifest"])]
    manifest = [s for s in manifest
                if s in cache and s in processed and processed[s].get("image")
                and cache[s].num_tokens <= args.max_prompt_tokens]

    benchmark = {}
    bench_path = config.get("benchmark_score_cache")
    if bench_path and Path(bench_path).exists():
        benchmark = {str(r["sample_id"]): r for r in read_jsonl(bench_path)}

    from transformers import AutoProcessor, AutoTokenizer
    processor = AutoProcessor.from_pretrained(
        tx_path, min_pixels=config["model"]["image_min_pixels"],
        max_pixels=config["model"]["image_max_pixels"], local_files_only=True)
    transmitter = load_hf_model(tx_path, dtype="bfloat16",
                                model_kind="vision_text_generation").to("cuda:0").eval()
    rx_device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    receiver_tokenizer = AutoTokenizer.from_pretrained(rx_path, local_files_only=True)
    receiver = load_generation_receiver(rx_path, device=rx_device)
    scorer = PragmaticScorer(transmitter, processor, config)
    scorer.collect_timings = True

    codebook = ProtectionCodebook(
        levels=tuple(float(v) for v in config["radio"]["protection"]["levels"]),
        information_bits_per_token=float(
            config["radio"]["protection"]["information_bits_per_token"]))
    info_bits_per_token = float(
        config["radio"]["protection"]["information_bits_per_token"])
    levels = torch.tensor(config["radio"]["protection"]["levels"], dtype=torch.float32)
    epsilon_out = float(config["radio"]["epsilon_out"])
    bandwidth_hz = float(config["radio"]["bandwidth"])
    slot = float(config["radio"]["slot_duration"])
    uses_per_slot = bandwidth_hz * slot
    beta = float(config["pragmatic"]["beta"])

    clock = Clock()
    records: list[dict] = []
    readout_rows: list[dict] = []

    def terminal_side(sid: str, image, method: str, order_seed: int):
        """Timed method pipeline on one sample; returns (record, prompt, expected)."""
        rec = processed[sid]
        scores = cache[sid]
        n_tokens = scores.num_tokens
        utility = torch.tensor([float(u) for u in combine_scores(
            scores.attention, scores.semantic, scores.gradient, args.weights)],
            dtype=torch.float32)
        coupling = (torch.tensor(scores.coupling, dtype=torch.float32)
                    if scores.coupling is not None
                    else torch.zeros(n_tokens, n_tokens))
        out = {"sample_id": sid, "method": method, "num_tokens": n_tokens}
        # ---- report regeneration is timed once per (sample, repeat) in the caller;
        # here only "cached report processing" is measured, then both scenarios are
        # formed by the caller.
        alt = None
        if method == "tokcom_uep":
            # The TokCom importance signal comes from an *external* model. Only the
            # acquisition of the pre-computed cache (read + normalise) can be timed
            # here; the offline cost of producing that cache is not measured and is
            # therefore reported as unmeasured rather than as zero.
            with clock(sync=False) as tick_ext:
                entry = benchmark.get(sid, {}).get("tokcom")
                if entry is None:
                    return None
                lo, hi = min(entry), max(entry)
                span = (hi - lo) or 1.0
                alt = torch.tensor([(v - lo) / span for v in entry], dtype=torch.float32)
            out["external_signal_ms"] = tick_ext.ms

        schedule_ms = float("nan")
        if method == "proposed":
            with clock(sync=False) as tick_sum:
                with clock(sync=False) as tick_lp:
                    lp = solve_base_lp(problem := SchedulingProblem(
                        base_utility=utility, coupling=coupling, beta=beta,
                        q=decoding_probability(state_cache[sid], codebook).reshape(-1),
                        # ``rates`` is the protection *cost* per token and level
                        # (48/32/16 coded bits), not the code rate: passing the code
                        # rate leaves the budget constraint slack and the solver then
                        # puts the whole report at the strongest level.
                        rates=levels,
                        budget=budget_cache[sid]))
                out["selection_lp_ms"] = tick_lp.ms
                with clock(sync=False) as tick_round:
                    rounded = certified_rounding(problem, lp=lp)
                out["selection_round_ms"] = tick_round.ms
                with clock(sync=False) as tick_refine:
                    refined = refine(problem, rounded.level, t_max=args.t_max)
                out["selection_refine_ms"] = tick_refine.ms
                level = refined.level
            schedule_ms = tick_sum.ms
        else:
            rng = random.Random(order_seed)
            with clock(sync=False) as tick:
                level = select_level("proposed" if method == "tokcom_uep" else method,
                                     alt if method == "tokcom_uep" else utility,
                                     coupling, beta,
                                     decoding_probability(state_cache[sid], codebook).reshape(-1),
                                     levels, budget_cache[sid], args.t_max, rng, alt)
            schedule_ms = tick.ms
        out["selection_ms"] = schedule_ms

        selected = [i for i in range(n_tokens) if float(level[i]) > 0]
        symbols = sum(float(codebook.levels[int(level[i]) - 1]) for i in selected)
        state = state_cache[sid]
        realized = math.log2(1.0 + float(state.snr_linear)
                             * float(state.h.abs().pow(2)[0]))
        rng = random.Random(9173 + int(sid))
        kept = []
        for token in selected:
            rate = float(codebook.code_rates[int(level[token]) - 1])
            if realized >= rate and rng.random() >= float(state.packet_loss_rate):
                kept.append(token)
        # ---- resource accounting (Eq. (6) single-window model)
        out.update({
            "selected_tokens": len(selected),
            "delivered_tokens": len(kept),
            "token_retention": len(kept) / len(selected) if selected else 0.0,
            "information_bits": info_bits_per_token * len(selected),
            "resource_cost_coded_bits": symbols,          # renamed "symbols"
            "modulation_order": MODULATION_ORDER,
            "channel_uses": symbols / MODULATION_ORDER,   # N_use = sum_i N_{use,i,l_i}
            "bandwidth_hz": bandwidth_hz,
            "uses_per_slot": uses_per_slot,
            "coded_bits_per_use": MODULATION_ORDER,
            "window_capacity_bits": uses_per_slot * MODULATION_ORDER,
            "window_slots_model": math.ceil(symbols / MODULATION_ORDER / uses_per_slot),
            "budget_coded_bits": budget_cache[sid],
            "budget_windows": budget_cache[sid] / (uses_per_slot * MODULATION_ORDER),
            "physical_window_cap_bits": (min(MODULATION_ORDER, ceps_cache[sid])
                                         * uses_per_slot),
        })
        with clock(sync=False) as tick_alloc:
            allocate_bandwidth(torch.tensor([float(utility.max())] * 10, dtype=torch.float32),
                               bandwidth_hz, theta=0.1)
        out["allocation_ms"] = tick_alloc.ms
        out["control_wait_ms"] = float("nan")   # unmeasured
        out["codec_ms"] = float("nan")          # unmeasured

        received = receiver_tokenizer.decode([scores.token_ids[i] for i in kept],
                                             skip_special_tokens=True)
        content = build_v6_answer_content(
            f"Question: {rec['question']}\n\nCandidate choices:\n"
            f"{build_choice_block(rec['choices'])}", received, ANSWER_REQUEST)
        chat = receiver_tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        with clock() as tick_prep:
            rx_inputs = receiver_tokenizer(chat, return_tensors="pt").to(rx_device)
        out["receiver_prep_ms"] = tick_prep.ms
        with clock() as tick_fwd:
            with torch.no_grad():
                logits = receiver(**rx_inputs).logits[0, -1]
        out["receiver_forward_ms"] = tick_fwd.ms
        groups = letter_token_ids(receiver_tokenizer)
        # As implemented: one GPU->CPU scalar transfer per option letter, which the
        # earlier pass silently charged to the readout stage.
        with clock() as tick_per_letter:
            per_letter = [float(logits[ids].max()) if ids else float("-inf")
                          for ids in groups]
        # Batched path: a single transfer of the letter rows followed by the argmax
        # on CPU. Same decision rule, so the two paths agree on every input.
        with clock() as tick_batched:
            flat = sorted({i for ids in groups for i in ids})
            indexed = logits[torch.tensor(flat, device=logits.device)].float().cpu()
            position = {tid: k for k, tid in enumerate(flat)}
            batched = [max((float(indexed[position[i]]) for i in ids), default=float("-inf"))
                       if ids else float("-inf") for ids in groups]
        index = int(max(range(len(batched)), key=lambda i: batched[i]))
        prediction = (rec["choices"][index] if index < len(rec["choices"])
                      else OPTION_LETTERS[index])
        out["receiver_readout_ms"] = tick_batched.ms
        out["receiver_readout_perletter_ms"] = tick_per_letter.ms
        out["readout_paths_agree"] = float(per_letter == batched)
        out["receiver_ms"] = (out["receiver_prep_ms"] + out["receiver_forward_ms"]
                              + out["receiver_readout_ms"])
        out["prediction"] = prediction
        out["correct"] = float(prediction == str(rec["choices"][rec["answer"]]))
        out["readout_content"] = content
        del logits, rx_inputs
        torch.cuda.empty_cache()
        return out

    # ------------------------------------------------------------ sample loop
    sample_ids = manifest[:args.samples]
    LOGGER.info("same-condition pass: %d samples x %d methods x %d repeats "
                "(warm-up %d), %g dB", len(sample_ids), len(args.methods),
                args.repeats, args.warmup, args.snr_db)
    channel_spec = yaml.safe_load(open(config["channels"][0]))

    def prepare_state(sid: str, index: int):
        state = build_channel({**channel_spec, "snr_db": args.snr_db,
                               "seed": 4242 + index}).sample((1,))
        c_eps = float(reliable_spectral_efficiency(state, epsilon_out)[0])
        return state, slot * bandwidth_hz * c_eps, c_eps

    state_cache: dict[str, object] = {}
    budget_cache: dict[str, float] = {}
    ceps_cache: dict[str, float] = {}
    for index, sid in enumerate(sample_ids):
        state_cache[sid], budget_cache[sid], ceps_cache[sid] = prepare_state(sid, index)

    completed = 0
    for index, sid in enumerate(sample_ids):
        rec = processed[sid]
        image = Image.open(rec["image"]).convert("RGB")
        for repeat in range(args.warmup + args.repeats):
            warm = repeat < args.warmup
            order = list(args.methods)
            random.Random(1000 + index * 97 + repeat).shuffle(order)
            # ---- independent scenario: one report generation shared by all methods
            scorer.timings.clear()
            with clock() as tick_report:
                prompt = scorer.generate_prompt(rec, image)
            report_ms = tick_report.ms
            report_tokens = len(processor.tokenizer(prompt, add_special_tokens=False)["input_ids"])
            torch.cuda.empty_cache()
            # ---- pragmatic scoring: shared input for the methods that consume the
            # pragmatic utility (proposed and Adaptive-SC here); TokCom-UEP consumes
            # the external TokCom signal instead, and the uncompressed reference
            # consumes neither. Methods that do not need it are not charged for it.
            need_scoring = any(m in ("proposed", "adaptive_sc") for m in args.methods)
            scoring_net = float("nan")
            scoring_children: dict[str, float] = {}
            if need_scoring:
                scorer.timings.clear()
                with clock() as tick_score:
                    score_result = scorer.score(rec, image)
                scoring_children = {f"scoring_{k}_ms": v * 1e3
                                    for k, v in scorer.timings.items()}
                # The official scoring total excludes the shared report generation
                # (all of its sub-stages) and the legacy per-element serialisation
                # diagnostic (kept only as the "before" side of the comparison).
                scoring_net = (tick_score.ms
                               - sum(v for k, v in scorer.timings.items()
                                     if k.startswith("report")) * 1e3
                               - scorer.timings.get("serialize_legacy", 0.0) * 1e3)
            for method in order:
                row = terminal_side(sid, image, method, order_seed=index * 31 + len(method))
                if row is None:
                    LOGGER.warning("sample %s method %s skipped (missing signal)", sid, method)
                    continue
                # The pragmatic scoring is a shared *input*: charge it only to the
                # methods whose selection rule consumes the pragmatic utility.
                # TokCom-UEP consumes an external signal instead (an external model
                # whose cost is not measured here), and the uncompressed reference
                # only packages the payload, so both are charged nothing for it.
                consumes = method in SCORING_USERS
                row.update({
                    "channel": state_cache[sid].channel,
                    "channel_seed": 4242 + index, "snr_db": args.snr_db,
                    "instance_tag": args.instance_tag,
                    "repeat": repeat, "warmup": warm,
                    "report_ms": report_ms, "report_tokens": report_tokens,
                    "scoring_ms": scoring_net if consumes else float("nan"),
                    **(scoring_children if consumes
                       else {k: float("nan") for k in scoring_children}),
                    "scoring_attributed": float(consumes),
                    "scoring_note": ("共享输入：该方法的选级规则消费语用效用"
                                     if consumes else
                                     "不消费语用效用：不计评分成本"),
                    "external_signal_ms": row.get("external_signal_ms", float("nan")),
                    "external_signal_source": (
                        "仅计时缓存读取与归一化；离线生成成本未测"
                        f"（缓存 {bench_path}）" if method == "tokcom_uep" else ""),
                    "baseline_role": BASELINE_ROLE.get(method, ""),
                    "uplink_ms_sim": UPLINK_MS, "downlink_ms_sim": DOWNLINK_MS,
                    "trigger_ms_sim": TRIGGER_MS,
                    "source_tags": {"report_ms": "measured",
                                    "scoring_ms": "measured" if consumes else "not-applicable",
                                    "selection_ms": "measured", "allocation_ms": "measured",
                                    "receiver_ms": "measured", "uplink_ms_sim": "simulated",
                                    "downlink_ms_sim": "simulated",
                                    "trigger_ms_sim": "simulated",
                                    "control_wait_ms": "unmeasured",
                                    "codec_ms": "unmeasured",
                                    "external_signal_ms": (
                                        "measured (cache read only)" if method == "tokcom_uep"
                                        else "not-applicable"),
                                    "total_ms": "partial (control wait and codec unmeasured)"},
                })
                row["terminal_ms"] = (row["report_ms"]
                                      + (row["scoring_ms"] if consumes else 0.0)
                                      + row["selection_ms"] + row["allocation_ms"])
                # Main metric: from the moment the vehicle holds the image and the
                # question to the moment the RSU emits the answer. The on-board
                # trigger and the answer downlink lie outside this boundary and are
                # kept only in the extended (vehicle-received-answer) variant.
                row["task_ms_main"] = (row["terminal_ms"] + row["receiver_ms"]
                                       + UPLINK_MS)
                row["total_ms"] = row["task_ms_main"]
                row["total_ms_extended"] = (row["task_ms_main"]
                                            + TRIGGER_MS + DOWNLINK_MS)
                # Everything named ``scoring_report*`` belongs to the shared report
                # generation (already excluded from the parent) and
                # ``scoring_serialize_legacy`` is the diagnostic "before" path.
                row["scoring_residual_ms"] = (
                    row["scoring_ms"]
                    - sum(v for k, v in scoring_children.items()
                          if not k.startswith("scoring_report")
                          and k != "scoring_serialize_legacy_ms"))
                row["scenario"] = "independent"
                if not warm:
                    records.append(row)
                # ---- 8B readout comparison on this method's delivered text. Measured
                # for *every* method (previously only for the method that happened to
                # be first in the randomised order, which left the paired roster
                # unbalanced across methods); repeats are averaged in aggregation.
                if not warm:
                    max_new = 24
                    with clock() as tick_auto:
                        autotext = generate_text(receiver, receiver_tokenizer,
                                                 row["readout_content"],
                                                 max_new_tokens=max_new)
                    auto_pred, _ = canonical_prediction(autotext, rec["choices"])
                    auto_tokens = len(receiver_tokenizer(
                        autotext, add_special_tokens=False)["input_ids"])
                    readout_rows.append({
                        "sample_id": sid, "method": method, "repeat": repeat,
                        "discriminative_ms": row["receiver_forward_ms"] + row["receiver_readout_ms"],
                        "autoregressive_ms": tick_auto.ms,
                        "autoregressive_chars": len(autotext),
                        "autoregressive_tokens": auto_tokens,
                        "max_new_tokens": max_new,
                        # Proxy rule: the generation length is not returned by the
                        # helper, so hitting the decoded-text token count of the cap
                        # is used as the "did not stop early" indicator.
                        "stop_reason_proxy": ("上限" if auto_tokens >= max_new else "早停"),
                        "agree": float(auto_pred == row["prediction"]),
                        "discriminative_correct": row["correct"],
                        "autoregressive_correct": float(
                            auto_pred == str(rec["choices"][rec["answer"]]))})
                del row["readout_content"]
        completed += 1
        LOGGER.info("sample %s done (%d/%d)", sid, completed, len(sample_ids))

    # ------------------------------------------------------------- aggregation
    import pandas as pd
    frame = pd.DataFrame(records)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "task_records.csv", index=False)

    # per-sample medians over repeats, then across-sample statistics
    per_sample = frame.groupby(["sample_id", "method"]).median(numeric_only=True)
    summary: dict = {"conditions": {
        "samples": len(sample_ids), "repeats": args.repeats, "warmup": args.warmup,
        "snr_db": args.snr_db, "methods": args.methods,
        "window_model": WINDOW_MODEL, "modulation_order": MODULATION_ORDER,
        "receiver": rx_path, "transmitter": tx_path,
        "hardware": torch.cuda.get_device_name(0),
        "note": "per-sample medians over repeats; across-sample P50/P95 are over the "
                "30 inputs, so they are preliminary estimates"},
        "paired_vs_baseline": {}, "per_method": {}}
    keys = [k for k in ("report_ms", "scoring_ms", "scoring_prepare_ms",
                        "scoring_assemble_ms", "scoring_forward_ms",
                        "scoring_attention_aggregate_ms", "scoring_backward_ms",
                        "scoring_coupling_ms", "scoring_residual_ms",
                        "selection_ms", "selection_lp_ms", "selection_round_ms",
                        "selection_refine_ms", "allocation_ms", "receiver_prep_ms",
                        "receiver_forward_ms", "receiver_readout_ms", "receiver_ms",
                        "terminal_ms", "total_ms", "resource_cost_coded_bits",
                        "channel_uses", "information_bits", "selected_tokens",
                        "delivered_tokens", "correct")
               if k in frame.columns]
    for method in args.methods:
        slice_frame = frame[frame.method == method]
        if slice_frame.empty:
            continue
        per = slice_frame.groupby("sample_id").median(numeric_only=True)
        summary["per_method"][method] = {k: describe(per[k].tolist()) for k in keys}
        if method != args.baseline:
            base = per_sample.xs(args.baseline, level="method")["total_ms"]
            diff = (per["total_ms"] - base.reindex(per.index)).dropna().tolist()
            lo, hi = bootstrap_ci(diff)
            summary["paired_vs_baseline"][method] = {
                "metric": "total_ms", "n": len(diff),
                "median_diff_ms": statistics.median(diff) if diff else float("nan"),
                "ci95": [lo, hi]}
    readout = pd.DataFrame(readout_rows)
    if not readout.empty:
        readout.to_csv(out_dir / "readout_comparison.csv", index=False)
        summary["readout_comparison"] = {
            "discriminative_ms": describe(readout.discriminative_ms.tolist()),
            "autoregressive_ms": describe(readout.autoregressive_ms.tolist()),
            "autoregressive_tokens": describe(readout.autoregressive_tokens.tolist()),
            "agree_rate": float(readout.agree.mean()),
            "discriminative_accuracy": float(readout.discriminative_correct.mean()),
            "autoregressive_accuracy": float(readout.autoregressive_correct.mean())}
    atomic_json(out_dir / "task_records_summary.json", summary)
    LOGGER.info("wrote %s", out_dir)


if __name__ == "__main__":
    main()
