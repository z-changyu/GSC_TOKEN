#!/usr/bin/env python
"""E6 / R2 cross-view fusion for V2X-QA cooperative (CO) samples — 端→边.

For every CO sample the terminal runs *twice*, once per viewpoint (vehicle-side
and infrastructure-side camera), producing two independent semantic reports.
Each report is pragmatically scored and transmitted over **its own realised
channel** — the two views therefore see independent fading, and the SNR pair is
swept so that symmetric (both good / both bad) and asymmetric (one side deep in
fade) operating points are both covered. The edge RSU text coordinator (the
*receiver*, `Qwen3-4B`) then ingests the surviving report(s) and answers.

Conditions emitted per (channel, SNR pair, seed, method):

``vehicle_only``        only the vehicle-view report arrives  (single-view ref.)
``infrastructure_only`` only the infrastructure-view report arrives
``fused``               both surviving reports, source-labelled, concatenated
``single_double``       vehicle-view report transmitted with the **combined**
                        budget of both radios (same total bandwidth as fusion,
                        but no information fusion) — this is the control that
                        separates "fusion helps" from "more bandwidth helps".

Answering is run under **both** receiver protocols by default:

``v0``  payload content only — accuracy here reflects what actually arrived;
``v6``  RSU-issued task intent + damage-triggered restoration + letter-only reply.

The dual-protocol default exists because the modes carry different payload sizes
(one report, two reports, or one report with both radios' budget), so under ``v6``
the restoration pass fires at different rates per mode and would otherwise
confound "fusion helps" with "this payload happened to look damaged". Restoration
rate is recorded per row so the two effects can be separated.

Cost accounting matches the main channel experiment: ``transmitted_symbols`` is
the sum of the protection levels of the selected tokens (``selected_tokens`` is
recorded separately).

Model footprint: the transmitter VLM and the receiver LM are resident at the same
time. With the 端 2B / 边 4B ladder this is ~12 GiB and fits a 24 GiB card; an
8B transmitter would not (load it only with the cloud/upper-bound variant).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODES = ("vehicle_only", "infrastructure_only", "fused", "single_double")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/v2x_edge_edge_main_2b4b.yaml")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/v2x_co_fusion")
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--modes", nargs="*", default=list(MODES))
    parser.add_argument("--channels", nargs="*", default=None,
                        help="subset of config channel paths")
    parser.add_argument("--snr-db", type=float, nargs="*", default=None,
                        help="SNR values swept pairwise over the two views")
    parser.add_argument("--channel-seeds", type=int, nargs="*", default=None)
    parser.add_argument("--protocol", nargs="*", choices=["v0", "v6", "v6_norestore"],
                        default=["v0", "v6_norestore", "v6"],
                        help="receiver protocol(s). All three are run by default: "
                             "v6's damage-triggered restoration fires at a rate set by "
                             "how much payload each mode happens to deliver, so "
                             "v6_norestore (task intent + letter output, restoration "
                             "off) is the arm where accuracy reflects content only, and "
                             "v0 is the historical description-only prompt.")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    visible = str(config["experiment"].get("cuda_visible_devices", "0"))
    # An explicit CUDA_VISIBLE_DEVICES in the environment wins over the config.
    # This driver keeps the 4B transmitter and the 8B coordinator resident at the
    # same time, which does not fit on one 24 GiB card, so a dual-device override
    # is required; placement does not change prompts, tokens or numerics. Without
    # the override the historical single-GPU guard applies unchanged.
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        if "," in visible:
            raise SystemExit("single GPU only")
        os.environ["CUDA_VISIBLE_DEVICES"] = visible
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    from transformers import (AutoImageProcessor, AutoModelForCausalLM, AutoProcessor,
                              AutoTokenizer, Qwen3VLForConditionalGeneration)
    from src.channel import build_channel, decoding_probability
    from src.channel.link import ProtectionCodebook
    from src.channel.reliable_capacity import reliable_spectral_efficiency
    from src.experiments.prompts import (ANSWER_REQUEST, TRANSMITTER_REQUEST,
                                         build_choice_block)
    from src.experiments.receiver_protocol import RESTORE_THRESHOLD, answer_with_protocol
    from src.experiments.scoring import PragmaticScorer, combine_scores, load_image
    from src.metrics.qa import exact_match
    from src.scheduling import select_level
    from src.utils.io import atomic_json, read_jsonl
    from src.utils.logging import configure_logging, create_run_directory

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run = create_run_directory("outputs/runs", "v2x_co_fusion", config)
    logger = configure_logging(run / "run.log")

    tx_path = config["model"]["transmitter_path"]
    rx_path = config["model"]["receiver_path"]
    processor = AutoProcessor.from_pretrained(tx_path, local_files_only=True)
    processor.image_processor = AutoImageProcessor.from_pretrained(
        tx_path, min_pixels=config["model"]["image_min_pixels"],
        max_pixels=config["model"]["image_max_pixels"], local_files_only=True)
    tokenizer = processor.tokenizer
    transmitter = Qwen3VLForConditionalGeneration.from_pretrained(
        tx_path, dtype=torch.bfloat16, attn_implementation="eager",
        local_files_only=True).to("cuda:0").eval()
    rx_tokenizer = AutoTokenizer.from_pretrained(rx_path, local_files_only=True)
    from src.models.registry import load_generation_receiver
    # The 4B transmitter and the 8B coordinator both stay resident for the whole
    # sweep, which does not fit on a single 24 GiB card: place the receiver on a
    # second visible device. This is device placement only -- prompts, tokens and
    # numerics are unchanged, so results remain comparable with the single-GPU runs.
    rx_device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    receiver = load_generation_receiver(rx_path, device=rx_device)

    scorer = PragmaticScorer(transmitter, processor, config)
    codebook = ProtectionCodebook(
        levels=tuple(float(v) for v in config["radio"]["protection"]["levels"]),
        information_bits_per_token=float(config["radio"]["protection"]["information_bits_per_token"]))
    bandwidth = float(config["radio"]["bandwidth"])
    slot_duration = float(config["radio"]["slot_duration"])
    epsilon_out = float(config["radio"]["epsilon_out"])
    beta = float(config["pragmatic"]["beta"])
    t_max = int(config["scheduling"]["t_max"])
    prag = config["pragmatic"]
    weights = tuple(float(w) for w in (prag.get("weights") or [prag["lambda_attention"],
                                                              prag["lambda_semantic"],
                                                              prag["lambda_gradient"]]))
    levels = torch.tensor(codebook.levels, dtype=torch.float32)
    methods = args.methods or list(config["methods"])
    modes = [m for m in args.modes if m in MODES]
    if not modes:
        raise SystemExit(f"no valid mode in {args.modes}; expected subset of {MODES}")
    channel_paths = args.channels or list(config["channels"])
    snrs = [float(v) for v in (args.snr_db or config["snr_db"])]
    seeds = [int(v) for v in (args.channel_seeds or config["experiment"]["channel_seeds"])]
    protocols = list(args.protocol)

    processed = {str(r["row_idx"]): r for r in read_jsonl(config["dataset"]["processed"])}
    co_ids = [sid for sid, r in processed.items()
              if str(r.get("task_id", "")).startswith("CO") and r.get("image")
              and r.get("image_path_is")]
    if args.num_samples:
        co_ids = co_ids[: args.num_samples]
    logger.info("CO fusion over %d samples; modes=%s protocol=%s", len(co_ids), modes, args.protocol)

    def build_view(row, image):
        choices = list(row["choices"])
        return {"image": image, "question": row["question"], "choices": choices,
                "answer": row["answer"],
                "_request": TRANSMITTER_REQUEST.format(
                    question=row["question"], choices=build_choice_block(choices))}

    def transmit_view(sc, level, rng, ch, state):
        """Per-token decode test against this view's own realised channel.

        Returns ``(text, survived_tokens, transmitted_symbols, selected_tokens)``.
        ``transmitted_symbols`` uses the same accounting as the main channel
        experiment (sum of the protection levels of the selected tokens) so the
        two experiments' cost columns are directly comparable.
        """
        selected = [i for i in range(sc.num_tokens) if level[i] > 0]
        symbols = sum(float(codebook.levels[int(level[i]) - 1]) for i in selected)
        realized = math.log2(1.0 + float(state.snr_linear) * float(state.h.abs().pow(2)[0]))
        kept = []
        for token in selected:
            rate = float(codebook.code_rates[int(level[token]) - 1])
            decoded = realized >= rate
            dropped = rng.random() < float(state.packet_loss_rate)
            if decoded and not dropped:
                kept.append(token)
        text = tokenizer.decode([sc.token_ids[i] for i in kept], skip_special_tokens=True)
        return ch.transmit(text), len(kept), symbols, len(selected)

    records = []
    for sid in co_ids:
        row = processed[sid]
        views = [("vehicle", row["image"]), ("infrastructure", row["image_path_is"])]
        # Score every viewpoint once; reuse across all channel conditions.
        view_scores = [(vname, scorer.score(build_view(row, vimg), load_image(vimg)))
                       for vname, vimg in views]
        (_, sc_v), (_, sc_i) = view_scores
        ground_truth = str(row["choices"][row["answer"]])
        base_row = {"sample_id": sid, "channel_seed": None}

        for channel_path in channel_paths:
            base = yaml.safe_load(Path(channel_path).read_text(encoding="utf-8"))
            channel_name = base.get("name", Path(channel_path).stem)
            for snr_v in snrs:
                for snr_i in snrs:
                    for seed in seeds:
                        # Independent fading per view: separate channel instances
                        # and separate seeds (the two edge radios are not shared).
                        ch_v = build_channel({**base, "snr_db": float(snr_v), "seed": int(seed)})
                        ch_i = build_channel({**base, "snr_db": float(snr_i),
                                              "seed": int(seed) + 7919})
                        state_v = ch_v.sample((1,))
                        state_i = ch_i.sample((1,))
                        budget_v = slot_duration * bandwidth * float(
                            reliable_spectral_efficiency(state_v, epsilon_out)[0])
                        budget_i = slot_duration * bandwidth * float(
                            reliable_spectral_efficiency(state_i, epsilon_out)[0])
                        q_v = decoding_probability(state_v, codebook).reshape(-1)
                        q_i = decoding_probability(state_i, codebook).reshape(-1)

                        for method in methods:
                            # One independent RNG per transmission arm. Sharing a single
                            # stream made each arm's packet-loss draws depend on how many
                            # tokens the *earlier* arms happened to select -- which leaked
                            # the vehicle's SNR into the infrastructure arm and made the
                            # 2x2 SNR cross internally inconsistent.
                            base_seed = int(seed) * 1000003 + len(method)
                            rng_level = random.Random(base_seed)
                            rng_v = random.Random(base_seed + 11)
                            rng_i = random.Random(base_seed + 23)
                            rng_v2 = random.Random(base_seed + 37)
                            ut_v = combine_scores(sc_v.attention, sc_v.semantic,
                                                  sc_v.gradient, weights).float()
                            ut_i = combine_scores(sc_i.attention, sc_i.semantic,
                                                  sc_i.gradient, weights).float()
                            zeros_v = torch.zeros(sc_v.num_tokens, sc_v.num_tokens)
                            zeros_i = torch.zeros(sc_i.num_tokens, sc_i.num_tokens)

                            lv_v = select_level(method, ut_v, zeros_v, beta, q_v, levels,
                                                budget_v, t_max, rng_level)
                            text_v, surv_v, sym_v, sel_v = transmit_view(sc_v, lv_v, rng_v,
                                                                          ch_v, state_v)
                            lv_i = select_level(method, ut_i, zeros_i, beta, q_i, levels,
                                                budget_i, t_max, rng_level)
                            text_i, surv_i, sym_i, sel_i = transmit_view(sc_i, lv_i, rng_i,
                                                                         ch_i, state_i)
                            # Control: one report with both radios' budget.
                            lv_v2 = select_level(method, ut_v, zeros_v, beta, q_v, levels,
                                                 budget_v + budget_i, t_max, rng_level)
                            text_v2, surv_v2, sym_v2, sel_v2 = transmit_view(sc_v, lv_v2,
                                                                             rng_v2, ch_v,
                                                                             state_v)

                            survivors = {
                                "vehicle_only": (text_v, surv_v, sym_v, sel_v, sc_v.num_tokens),
                                "infrastructure_only": (text_i, surv_i, sym_i, sel_i,
                                                        sc_i.num_tokens),
                                "fused": (f"[vehicle view] {text_v}\n\n"
                                          f"[infrastructure view] {text_i}",
                                          surv_v + surv_i, sym_v + sym_i, sel_v + sel_i,
                                          sc_v.num_tokens + sc_i.num_tokens),
                                "single_double": (text_v2, surv_v2, sym_v2, sel_v2,
                                                  sc_v.num_tokens),
                            }

                            for mode in modes:
                                received, survived, symbols, selected_tokens, total = survivors[mode]
                                for arm in protocols:
                                    # v6_norestore keeps v6's task-intent + letter-only
                                    # prompt but makes the restoration trigger unreachable.
                                    active = "v6" if arm == "v6_norestore" else arm
                                    threshold = (0.0 if arm == "v6_norestore"
                                                 else RESTORE_THRESHOLD)
                                    answer = answer_with_protocol(
                                        model=receiver, tokenizer=rx_tokenizer, row=row,
                                        received=received, survived_tokens=int(survived),
                                        total_tokens=int(total), answer_request=ANSWER_REQUEST,
                                        max_new_tokens=int(config["receiver"]["max_new_tokens"]),
                                        protocol=active, restore_threshold=threshold)
                                    records.append({
                                        **base_row, "channel": channel_name,
                                        "snr_vehicle": float(snr_v),
                                        "snr_infrastructure": float(snr_i),
                                        "channel_seed": int(seed), "method": method, "mode": mode,
                                        "protocol": arm,
                                        "correct": float(exact_match(answer["prediction"],
                                                                     ground_truth)),
                                        "parse_mode": answer["parse_mode"],
                                        "restored": int(answer["restored"]),
                                        "transmitted_symbols": int(symbols),
                                        "selected_tokens": int(selected_tokens),
                                        "survived_tokens": int(survived),
                                        "total_tokens": int(total),
                                    })
                        torch.cuda.empty_cache()
        logger.info("sample %s done (%d records)", sid, len(records))

    import pandas as pd
    frame = pd.DataFrame(records)
    frame.to_csv(out_dir / "co_fusion_results.csv", index=False)
    summary_rows = []
    for (method, mode, protocol, sv, si), g in frame.groupby(
            ["method", "mode", "protocol", "snr_vehicle", "snr_infrastructure"], sort=True):
        summary_rows.append({
            "method": method, "mode": mode, "protocol": protocol,
            "snr_vehicle": float(sv), "snr_infrastructure": float(si),
            "accuracy": float(g["correct"].mean()), "observations": len(g),
            "mean_symbols": float(g["transmitted_symbols"].mean()),
            "mean_selected_tokens": float(g["selected_tokens"].mean()),
            "mean_survived_tokens": float(g["survived_tokens"].mean()),
            "mean_restore_rate": float(g["restored"].mean()),
        })
    summary = {"samples": len(co_ids), "protocols": protocols,
               "modes": modes, "methods": methods, "conditions": summary_rows}
    atomic_json(out_dir / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("samples", "protocols", "modes")}, indent=2))
    print(f"{len(summary_rows)} condition rows -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
