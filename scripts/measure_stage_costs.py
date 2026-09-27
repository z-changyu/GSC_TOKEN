#!/usr/bin/env python
"""Experiments 1-3: compute-cost breakdown, baseline cost comparison and the
end-to-end task timeline of the main line (terminal 4B -> edge 8B).

Experiment 1 (``--out-dir``/``stage_costs``)
    Per-stage cost of the proposed pipeline with the sub-stages the paper claims:
    report generation (preprocess / vision tower / text decoding), pragmatic
    scoring (embedding+rope assembly / teacher-forced forward / attention and
    semantic aggregation / gradient backward / coupling construction), token
    scheduling (base LP / certified rounding / local refinement), multi-terminal
    coordination (semantic price and bandwidth allocation for K in {4,10,20}) and
    receiver inference (input preparation / forward / letter readout).
    Reported as median, P95 and the number of observations, together with the
    hardware, precision, batch size, token length, warm-up and the timing method
    (``torch.cuda.synchronize`` around every GPU stage).

Experiment 2 (``--methods``)
    The same images, reports, receiver and hardware, timed for the proposed
    method and for Adaptive-SC, TokCom-UEP and the uncompressed reference.
    Report generation and receiver inference are shared and measured once; the
    method-specific stage (``select_level``) is timed independently per method,
    which is what answers "does the saved spectrum cost more computation?".

Experiment 3 (``--timeline``)
    One row per sample from image input to RSU answer: on-terminal
    (preprocess/vision/report/scoring/scheduling) and edge-side (answering)
    stages are measured; the native trigger, the uplink slot and the downlink are
    simulated/set (flagged as such). Per-sample totals are formed first and the
    median/P95 are then taken over samples -- never by summing stage medians.
    This is the rich-semantics channel, not the warning-gating path of Fig. 9.

Usage
-----
    python scripts/measure_stage_costs.py --samples 6            # smoke
    python scripts/measure_stage_costs.py --samples 40           # paper numbers
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import statistics
import sys
import time
from pathlib import Path

# The 4B transmitter and the 8B receiver are both resident for the whole pass, so
# the allocator is asked for expandable segments (the same setting the channel and
# fusion drivers use) to keep interleaved stages from fragmenting the card.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.channel.link import ProtectionCodebook, decoding_probability  # noqa: E402
from src.experiments.prompts import ANSWER_REQUEST  # noqa: E402
from src.experiments.receiver_protocol import letter_token_ids, OPTION_LETTERS  # noqa: E402
from src.experiments.scoring import PragmaticScorer, SampleScores, combine_scores  # noqa: E402
from src.models.registry import load_generation_receiver, load_hf_model  # noqa: E402
from src.scheduling import certified_rounding, refine, select_level, solve_base_lp  # noqa: E402
from src.scheduling.bandwidth import allocate_bandwidth  # noqa: E402
from src.scheduling.problem import SchedulingProblem  # noqa: E402
from src.utils.io import atomic_json, read_jsonl  # noqa: E402
from src.utils.logging import environment_snapshot  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("stage_costs")

# The native perception trigger of the vehicular stack is assumed, not measured
# (the paper keeps it at 10-30 ms); the radio segments are software-simulated.
TRIGGER_MS = 20.0
UPLINK_MS = 1.0           # one slot, Eq. (6) budget respected
DOWNLINK_MS = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default="configs/experiments/v2x_edge_edge_main_4b8b.yaml")
    parser.add_argument("--samples", type=int, default=40,
                        help="number of questions visited (warm-up excluded)")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--limit", type=int, help="cap the candidate pool")
    parser.add_argument("--methods", nargs="*",
                        default=["proposed", "adaptive_sc", "tokcom_uep", "full"])
    parser.add_argument("--ks", type=int, nargs="*", default=[4, 10, 20])
    parser.add_argument("--snr-db", type=float, default=15.0)
    parser.add_argument("--weights", type=float, nargs=3, default=[0.25, 0.30, 0.45])
    parser.add_argument("--t-max", type=int, default=8)
    parser.add_argument("--max-prompt-tokens", type=int, default=200,
                        help="the scoring peak scales with the sequence length; the "
                             "cost pass targets the typical report length, and samples "
                             "beyond this cap are reported as skipped")
    parser.add_argument("--output-dir", default="outputs/stage_costs")
    return parser.parse_args()


class Stopwatch:
    """Wall-clock time with an explicit GPU synchronisation policy."""

    def __init__(self) -> None:
        self.gpu = torch.cuda.is_available()

    def __call__(self, gpu: bool = True):
        return _Block(self, gpu)


class _Block:
    def __init__(self, watch: Stopwatch, gpu: bool) -> None:
        self.watch = watch
        self.gpu = gpu

    def __enter__(self):
        if self.gpu and self.watch.gpu:
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.gpu and self.watch.gpu:
            torch.cuda.synchronize()
        self.elapsed_ms = (time.perf_counter() - self.start) * 1e3
        return False


def stats(values: list[float]) -> dict:
    xs = sorted(v for v in values if v == v)
    if not xs:
        return {"n": 0, "mean": float("nan"), "p50": float("nan"), "p95": float("nan")}
    rank = min(len(xs) - 1, max(0, math.ceil(0.95 * len(xs)) - 1))
    return {"n": len(xs), "mean": statistics.fmean(xs),
            "p50": statistics.median(xs), "p95": xs[rank]}


def main() -> None:
    args = parse_args()
    # Global grad mode stays on: the gradient-sensitivity stage differentiates
    # through the prompt embeddings; the scoring blocks manage their own scopes.
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    tx_path = config["model"]["transmitter_path"]
    rx_path = config["model"]["receiver_path"]

    cache = {str(r["sample_id"]): SampleScores.from_dict(r)
             for r in read_jsonl(config["score_cache"])}
    # Processed rows are keyed by row_idx and carry the transmitter request the
    # scorer expects, exactly as build_score_cache.py prepares them.
    from src.experiments.prompts import build_choice_block, select_transmitter_request
    request_template = select_transmitter_request(config)
    processed: dict[str, dict] = {}
    for _record in read_jsonl(config["dataset"]["processed"]):
        # Index under both keys: the manifest refers to samples by ``sample_id``
        # while the cache builder keys processed rows by ``row_idx``.
        for _key in {str(_record.get("sample_id", "")), str(_record.get("row_idx", ""))}:
            if _key:
                processed[_key] = _record
    for _row in processed.values():
        _row["_request"] = request_template.format(
            question=_row["question"], choices=build_choice_block(_row["choices"]))
    manifest = [str(r["sample_id"])
                for r in read_jsonl(config["dataset"]["manifest"])]
    manifest = [s for s in manifest if s in cache and processed.get(s, {}).get("image")]
    if args.limit:
        manifest = manifest[:args.limit]
    long_reports = [s for s in manifest
                    if cache[s].num_tokens > args.max_prompt_tokens]
    manifest = [s for s in manifest if cache[s].num_tokens <= args.max_prompt_tokens]
    LOGGER.info("prompt-length cap %d keeps %d samples (%d longer reports are "
                "outside the single-card scoring envelope)",
                args.max_prompt_tokens, len(manifest), len(long_reports))
    # Warm-up samples are visited first and dropped, they only fill caches/allocators.
    visit = manifest[:args.warmup + args.samples]
    LOGGER.info("stage-cost run: %d samples (%d warm-up), methods=%s",
                args.samples, min(args.warmup, len(visit)), args.methods)

    from transformers import AutoProcessor, AutoTokenizer
    # The pixel bounds carried by every other driver in this repo: without them the
    # processor falls back to its default resolution, which inflates the visual
    # token count and blows the single-card scoring envelope.
    processor = AutoProcessor.from_pretrained(
        tx_path, min_pixels=config["model"]["image_min_pixels"],
        max_pixels=config["model"]["image_max_pixels"], local_files_only=True)
    transmitter = load_hf_model(tx_path, dtype="bfloat16",
                                model_kind="vision_text_generation")
    transmitter = transmitter.to("cuda:0").eval()
    receiver_tokenizer = AutoTokenizer.from_pretrained(rx_path, local_files_only=True)
    # The 4B transmitter and the 8B receiver must both stay resident for the whole
    # pass, which does not fit on one 24 GiB card: place the receiver on a second
    # visible device. Device placement only -- prompts and numerics are unchanged.
    rx_device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    LOGGER.info("receiver on %s", rx_device)
    receiver = load_generation_receiver(rx_path, device=rx_device)
    scorer = PragmaticScorer(transmitter, processor, config)
    scorer.collect_timings = True

    codebook = ProtectionCodebook(
        levels=tuple(float(v) for v in config["radio"]["protection"]["levels"]),
        information_bits_per_token=float(
            config["radio"]["protection"]["information_bits_per_token"]))
    epsilon_out = float(config["radio"]["epsilon_out"])
    bandwidth_hz = float(config["radio"]["bandwidth"])
    slot = float(config["radio"]["slot_duration"])
    levels = torch.tensor(config["radio"]["protection"]["levels"], dtype=torch.float32)
    beta = float(config["pragmatic"]["beta"])
    weights = args.weights

    # TokCom-UEP consumes an external importance signal; reuse the cached one so
    # its method-specific cost is measured on the same footing as the others.
    benchmark = {}
    bench_path = config.get("benchmark_score_cache")
    if bench_path and Path(bench_path).exists():
        benchmark = {str(r["sample_id"]): r for r in read_jsonl(bench_path)}

    watch = Stopwatch()
    rows: list[dict] = []
    timeline_rows: list[dict] = []
    skipped = 0

    for visited, sid in enumerate(visit):
        warm = visited < args.warmup
        row_meta = processed[sid]
        image = Image.open(row_meta["image"]).convert("RGB")
        scores = cache[sid]
        n_tokens = scores.num_tokens
        record = {"sample_id": sid, "num_tokens": n_tokens, "warmup": warm}

        # ------------------------------------------------ 1. report generation
        from src.experiments.prompts import build_transmitter_messages
        with watch() as blk:
            inputs = processor.apply_chat_template(
                build_transmitter_messages([{"type": "image", "image": image},
                                           {"type": "text", "text": row_meta["_request"]}]),
                tokenize=True, add_generation_prompt=True, return_dict=True,
                return_tensors="pt")
        record["preprocess_ms"] = blk.elapsed_ms
        device_inputs = {k: (v.to("cuda:0") if hasattr(v, "to") else v)
                         for k, v in inputs.items()}
        vision_ms = float("nan")
        if "pixel_values" in device_inputs:
            visual = getattr(getattr(transmitter, "model", None), "visual", None)
            if visual is not None:
                with watch() as blk:
                    with torch.no_grad():
                        visual(device_inputs["pixel_values"],
                               device_inputs.get("image_grid_thw"))
                vision_ms = blk.elapsed_ms
        record["vision_ms"] = vision_ms
        with watch() as blk:
            with torch.no_grad():
                generated = transmitter.generate(**device_inputs,
                                                 max_new_tokens=int(config["transmitter"]["max_new_tokens"]),
                                                 do_sample=False)
        record["report_ms"] = blk.elapsed_ms
        LOGGER.info("  [mem] after report: gpu0=%.2f GiB gpu1=%.2f GiB",
                    torch.cuda.memory_allocated(0) / 2 ** 30,
                    (torch.cuda.memory_allocated(1) / 2 ** 30)
                    if torch.cuda.device_count() > 1 else 0.0)
        del generated, inputs
        torch.cuda.empty_cache()

        # ------------------------------------------------- 2. pragmatic scoring
        scorer.timings.clear()
        try:
            with watch() as blk:
                _ = scorer.score(row_meta, image)
        except torch.OutOfMemoryError:
            # Long image+prompt sequences can exceed a single 24 GiB card during the
            # teacher-forced forward with attentions (the paper reports a 22.6 GiB
            # peak at <=500 sequence tokens on the 8B scorer). Such samples are
            # skipped explicitly and counted, never silently averaged in.
            LOGGER.warning("sample %s skipped: scoring exceeded device memory", sid)
            skipped += 1
            gc.collect()
            torch.cuda.empty_cache()
            continue
        record["scoring_total_ms"] = blk.elapsed_ms
        for name in ("assemble", "forward", "attention_aggregate", "backward", "coupling",
                     "report"):
            record[f"scoring_{name}_ms"] = scorer.timings.get(name, float("nan")) * 1e3
        # ``score()`` regenerates the report internally, so this is the report cost
        # the pipeline actually pays; the pure scoring cost is the remainder.
        record["scoring_net_ms"] = (record["scoring_total_ms"]
                                    - record["scoring_report_ms"])
        torch.cuda.empty_cache()

        # ----------------------------------------------- 3. token scheduling
        channel_spec = yaml.safe_load(open(config["channels"][0]))
        from src.channel import build_channel
        state = build_channel({**channel_spec, "snr_db": args.snr_db,
                               "seed": int(visited)}).sample((1,))
        from src.channel.reliable_capacity import reliable_spectral_efficiency
        budget = float(slot * bandwidth_hz
                       * float(reliable_spectral_efficiency(state, epsilon_out)[0]))
        q = decoding_probability(state, codebook).reshape(-1)
        utility = torch.tensor([float(u) for u in combine_scores(
            scores.attention, scores.semantic, scores.gradient, weights)],
            dtype=torch.float32)
        coupling = (torch.tensor(scores.coupling, dtype=torch.float32)
                    if scores.coupling is not None
                    else torch.zeros(n_tokens, n_tokens))
        problem = SchedulingProblem(base_utility=utility, coupling=coupling, beta=beta,
                                    q=q, rates=torch.tensor(codebook.code_rates,
                                                            dtype=torch.float32),
                                    budget=budget)
        with watch(gpu=False) as blk:
            lp = solve_base_lp(problem)
        record["schedule_lp_ms"] = blk.elapsed_ms
        with watch(gpu=False) as blk:
            rounded = certified_rounding(problem, lp=lp)
        record["schedule_rounding_ms"] = blk.elapsed_ms
        with watch(gpu=False) as blk:
            refined = refine(problem, rounded.level, t_max=args.t_max)
        record["schedule_refine_ms"] = blk.elapsed_ms
        record["schedule_proposed_ms"] = (record["schedule_lp_ms"]
                                          + record["schedule_rounding_ms"]
                                          + record["schedule_refine_ms"])

        import random as _random
        rng = _random.Random(2026 + visited)
        alt = None
        if sid in benchmark and "tokcom" in benchmark[sid]:
            raw = benchmark[sid]["tokcom"]
            lo, hi = min(raw), max(raw)
            span = (hi - lo) or 1.0
            alt = torch.tensor([(v - lo) / span for v in raw], dtype=torch.float32)
        for method in args.methods:
            if method == "proposed":
                continue
            if method == "tokcom_uep":
                # tokcom_uep is not a standalone solver in the library: it feeds the
                # external TokCom importance signal into our scheduler, exactly as
                # the channel runner does. Its method-specific cost is therefore
                # measured on the same solver with the TokCom utility.
                if alt is None:
                    LOGGER.warning("tokcom_uep needs the benchmark cache; skipped")
                    continue
                with watch(gpu=False) as blk:
                    _ = select_level("proposed", alt, coupling, beta, q, levels,
                                     budget, args.t_max, rng)
                record["schedule_tokcom_uep_ms"] = blk.elapsed_ms
                continue
            with watch(gpu=False) as blk:
                _ = select_level(method, utility, coupling, beta, q, levels, budget,
                                 args.t_max, rng, alt)
            record[f"schedule_{method}_ms"] = blk.elapsed_ms

        # ------------------------------------------ 4. multi-terminal coordination
        reports = torch.tensor([utility.max().item() for _ in range(max(args.ks))],
                               dtype=torch.float32)
        for k in args.ks:
            with watch(gpu=False) as blk:
                alloc = allocate_bandwidth(reports[:k], bandwidth_hz, theta=0.1)
            record[f"coord_k{k}_ms"] = blk.elapsed_ms
            record[f"coord_k{k}_jain"] = float(alloc.jain) if hasattr(alloc, "jain") else float("nan")

        # ------------------------------------------------ 5. receiver inference
        received = receiver_tokenizer.decode(
            [scores.token_ids[i] for i in range(0, n_tokens, max(1, n_tokens // 8))],
            skip_special_tokens=True)
        content = f"Question: {row_meta['question']}\n\nSurviving description:\n{received}\n\n{ANSWER_REQUEST}"
        with watch() as blk:
            chat = receiver_tokenizer.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            rx_inputs = receiver_tokenizer(chat, return_tensors="pt").to(rx_device)
        record["rx_prep_ms"] = blk.elapsed_ms
        with watch() as blk:
            with torch.no_grad():
                logits = receiver(**rx_inputs).logits[0, -1]
        record["rx_forward_ms"] = blk.elapsed_ms
        with watch() as blk:
            groups = letter_token_ids(receiver_tokenizer)
            letter_scores = [float(logits[ids].max()) if ids else float("-inf")
                             for ids in groups]
            index = int(max(range(len(letter_scores)), key=lambda i: letter_scores[i]))
            prediction = OPTION_LETTERS[index]
        record["rx_readout_ms"] = blk.elapsed_ms
        record["correct"] = float(prediction == str(
            row_meta["choices"][row_meta["answer"]]))
        del logits, rx_inputs

        if not warm:
            rows.append(record)
            # --------------------------------------------- 6. task timeline row
            terminal_ms = (record["preprocess_ms"] + record["vision_ms"]
                           + record["scoring_report_ms"] + record["scoring_net_ms"]
                           + record["schedule_proposed_ms"])
            edge_ms = (record["rx_prep_ms"] + record["rx_forward_ms"]
                       + record["rx_readout_ms"])
            timeline_rows.append({
                "sample_id": sid, "num_tokens": n_tokens,
                "trigger_ms_sim": TRIGGER_MS,
                "terminal_ms_measured": terminal_ms,
                "uplink_ms_sim": UPLINK_MS,
                "edge_ms_measured": edge_ms,
                "downlink_ms_sim": DOWNLINK_MS,
                "total_ms": TRIGGER_MS + terminal_ms + UPLINK_MS + edge_ms + DOWNLINK_MS})
            LOGGER.info("sample %s (%d tokens): report %.0f + scoring %.0f + sched %.1f "
                        "| rx %.0f ms", sid, n_tokens, record["report_ms"],
                        record["scoring_total_ms"], record["schedule_proposed_ms"], edge_ms)

    # ---------------------------------------------------------------- aggregate
    if not rows:
        raise SystemExit("no sample completed the pass; every candidate exceeded "
                         "device memory during scoring")
    stages = [k for k in rows[0] if k.endswith("_ms") and k != "vision_ms"] + ["vision_ms"]
    summary = {"stages": {k: stats([r[k] for r in rows]) for k in stages},
               "metadata": {
                   "config": args.config, "snr_db": args.snr_db,
                   "samples": len(rows), "warmup": args.warmup,
                   "samples_visited": len(visit),
                   "samples_skipped_device_memory": skipped,
                   "token_length": stats([float(r["num_tokens"]) for r in rows]),
                   "precision": "bfloat16", "batch_size": 1,
                   "timing": "perf_counter with torch.cuda.synchronize around every "
                             "GPU stage; CPU scheduling stage without sync",
                   "transmitter": tx_path, "receiver": rx_path,
                   "hardware": torch.cuda.get_device_name(0),
                   "environment": environment_snapshot()}}
    timeline_summary = {k: stats([r[k] for r in timeline_rows])
                        for k in ("terminal_ms_measured", "edge_ms_measured",
                                  "total_ms", "trigger_ms_sim", "uplink_ms_sim",
                                  "downlink_ms_sim")}
    summary["timeline"] = {"per_stage_of_total": timeline_summary,
                           "note": "median/P95 taken over per-sample totals; the "
                                   "trigger, uplink and downlink entries are "
                                   "simulated or assumed, everything else measured",
                           "rows": len(timeline_rows)}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(out_dir / "stage_costs.csv", index=False)
    pd.DataFrame(timeline_rows).to_csv(out_dir / "task_timeline.csv", index=False)
    atomic_json(out_dir / "stage_costs_summary.json", summary)

    LOGGER.info("%-28s %9s %9s %9s %6s", "stage", "mean_ms", "p50_ms", "p95_ms", "n")
    for key, value in sorted(summary["stages"].items()):
        LOGGER.info("%-28s %9.2f %9.2f %9.2f %6d", key, value["mean"], value["p50"],
                    value["p95"], value["n"])
    LOGGER.info("timeline: terminal %.1f ms | edge %.1f ms | total p50 %.1f / p95 %.1f ms",
                timeline_summary["terminal_ms_measured"]["p50"],
                timeline_summary["edge_ms_measured"]["p50"],
                timeline_summary["total_ms"]["p50"], timeline_summary["total_ms"]["p95"])
    LOGGER.info("wrote %s", out_dir)
    del transmitter, scorer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
