#!/usr/bin/env python
"""End-to-end channel experiment: Section VI-B-1 and VI-B-2.

For every (channel, SNR, seed, sample) the script

1. samples a channel realisation and the imperfect CSI of Eq. (4);
2. turns it into the reliable budget ``C_tau = T_s B c_eps(h_hat)`` of Eq. (6)
   and the conditional decoding probabilities ``q_{i,ell,tau}``;
3. solves the reliability-aware schedule with Algorithm 1 — Lagrangian LP
   (Eq. 53), certified extreme-point rounding (Eq. 54) and coupling-aware local
   refinement (Eq. 56-59);
4. applies the *realised* channel to the selected tokens, so a token survives
   only if the realised rate clears its code rate and the packet is not lost;
5. lets the receiver answer from the surviving text only.

Method ``plugin_capacity`` is the internal baseline of Section VI-A: it uses the
plug-in capacity ``T_s B log2(1 + |h_hat|^2 gamma_bar)`` in place of Eq. (6),
which isolates the contribution of the reliable-capacity model.

The realised outage of VI-B-2 is reported alongside, because Eq. (8) promises
``Pr[T_s B psi(h) < C_tau | h_hat] <= epsilon_out`` and that promise has to be
measured, not assumed.
"""
from __future__ import annotations

import argparse, gc, json, math, os, random, sys, time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_channel_spec(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/channel_scienceqa.yaml")
    parser.add_argument("--num-samples", type=int, help="Override only for smoke testing")
    parser.add_argument("--channel-seeds", type=int, nargs="*")
    parser.add_argument("--snr-db", type=float, nargs="*")
    parser.add_argument("--channels", nargs="*", help="Subset of channel config paths")
    parser.add_argument("--methods", nargs="*")
    parser.add_argument("--body-mask", default=None, choices=["off", "proposed", "all"],
                        help="exclude the duplicated question/option prefix from the "
                             "candidate set: 'off' (default) keeps every method as is, "
                             "'proposed' applies it to proposed_body only, 'all' applies "
                             "it to every method except full so the comparison holds the "
                             "information available to each method fixed")
    parser.add_argument("--manifest", help="Override the sample manifest (e.g. one CV fold)")
    parser.add_argument("--score-cache", help="Override the cached pragmatic scores")
    parser.add_argument("--benchmark-scores",
                        help="JSONL sidecar of per-token LM self-information; required by "
                             "the selective_context / llmlingua baselines")
    parser.add_argument("--answer-scores",
                        help="JSONL sidecar of answer-aware importance (LOO / answer "
                             "attention) from scripts/build_answer_aware_scores.py")
    parser.add_argument("--coupling-variant", default=None,
                        choices=["raw", "row", "A", "B"],
                        help="coupling definition: 'raw' = Eq. (16) attention, "
                             "'row' = unit-row scaled, 'A' = attention masked by "
                             "token dissimilarity, 'B' = redundancy penalty -cos. "
                             "'A'/'B' need --token-embeddings.")
    parser.add_argument("--token-embeddings", default="outputs/cache/v2xqa_token_embeddings_8b.pt")
    parser.add_argument("--coupling-shuffle", action="store_true",
                        help="ablation arm: apply a fixed symmetric permutation to the "
                             "coupling matrix, preserving its multiset of coefficients "
                             "but destroying which token each coefficient belongs to. "
                             "Isolates 'the coupling carries structure' from 'any "
                             "coupling of this magnitude would do'.")
    parser.add_argument("--protection-mode", default=None, choices=["differentiated", "fixed"],
                        help="'fixed' forces every transmitted token to the strongest "
                             "protection level while keeping the set-aware ranking "
                             "(ablation arm: set score with fixed protection)")
    parser.add_argument("--lp-utility", default=None, choices=["marginal", "context"],
                        help="base utility of the LP/rounding: 'marginal' = u'_i (the "
                             "historical additive surrogate, coupling only acts in "
                             "refinement), 'context' = the Eq. (19) context-aware u_i, "
                             "which lets the coupling steer token selection itself")
    parser.add_argument("--beta", type=float, default=None,
                        help="override pragmatic.beta (needed when the coupling variant "
                             "changes the admissible range)")
    parser.add_argument("--coupling-model", default=None, choices=["joint", "independent"],
                        help="pairwise reception model used by the scheduler: 'joint' = "
                             "P(both received) under the shared instantaneous channel "
                             "(Eq. 34 as revised), 'independent' = the historical "
                             "q_i * q_j product (ablation only)")
    parser.add_argument("--weights", type=float, nargs=3,
                        help="Override the pragmatic weights (must sum to one)")
    parser.add_argument("--output-dir", default="outputs/channel")
    parser.add_argument("--receiver-protocol",
                        choices=["v0", "v6", "v6_norestore", "v6_letter"],
                        default="v0",
                        help="v0 = description-only answer prompt (original); "
                             "v6 = task intent + damage-triggered restoration + letter output; "
                             "v6_norestore = task intent + letter output with restoration OFF "
                             "(fair arm: the restore trigger fires at payload-size dependent "
                             "rates, so plain v6 rewards under-delivery); "
                             "v6_letter = same prompt as v6_norestore but the answer is read "
                             "off a single forward pass (output-layer argmax), i.e. no "
                             "autoregressive decoding -- used for the warning-latency study")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    visible = str(config["experiment"].get("cuda_visible_devices", "0"))
    if "," in visible:
        raise SystemExit("This driver is restricted to exactly one visible GPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
    from src.channel import build_channel, decoding_probability
    from src.channel.link import ProtectionCodebook, joint_decoding_probability
    from src.channel.reliable_capacity import reliable_spectral_efficiency
    from src.experiments.prompts import ANSWER_REQUEST
    from src.experiments.receiver_protocol import RESTORE_THRESHOLD, answer_with_protocol
    from src.experiments.report_span import report_span_start
    from src.experiments.scoring import SampleScores, combine_scores
    from src.pragmatic.utility import context_aware_utility
    from src.metrics.communication import token_retention_ratio
    from src.metrics.qa import exact_match
    from src.metrics.statistics import mean_std_ci95
    from src.scheduling import select_level
    from src.scheduling.problem import SchedulingProblem
    from src.utils.io import atomic_json, read_jsonl
    from src.utils.logging import configure_logging, create_run_directory
    from src.utils.power import PowerSampler

    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    run = create_run_directory("outputs/runs", "scienceqa_channel_experiment", config)
    logger = configure_logging(run / "run.log")

    cache = {row["sample_id"]: SampleScores.from_dict(row)
             for row in read_jsonl(args.score_cache or config["score_cache"])}
    # Per-token self-information produced by scripts/build_benchmark_scores.py. It
    # is the ranking signal of the external prompt-compression baselines, which
    # never see the pragmatic A/S/G scores.
    benchmark_scores: dict[str, dict[str, list[float]]] = {}
    bench_path = args.benchmark_scores or config.get("benchmark_score_cache")
    if bench_path and Path(bench_path).exists():
        benchmark_scores = {
            str(row["sample_id"]): {f: list(row.get(f, []))
                                    for f in ("surprisal", "surprisal_q",
                                              "llmlingua2", "tokcom")}
            for row in read_jsonl(bench_path)}
        logger.info("benchmark self-information: %d samples", len(benchmark_scores))
    # Answer-aware importance (LOO / answer attention) produced by
    # scripts/build_answer_aware_scores.py: the consumer-side, answer-aware
    # signals that the lexical baselines lack.
    answer_scores: dict[str, dict[str, list[float]]] = {}
    ans_path = args.answer_scores or config.get("answer_score_cache")
    if ans_path and Path(ans_path).exists():
        answer_scores = {
            str(row["sample_id"]): {"answer_loo": list(row.get("answer_loo", [])),
                                    "answer_attn": list(row.get("answer_attn", []))}
            for row in read_jsonl(ans_path)}
        logger.info("answer-aware importance: %d samples", len(answer_scores))
    processed = {str(row["row_idx"]): row for row in read_jsonl(config["dataset"]["processed"])}
    manifest = [str(entry["sample_id"])
                for entry in read_jsonl(args.manifest or config["dataset"]["manifest"])]
    if config["dataset"].get("require_image", True):
        manifest = [sid for sid in manifest if processed[sid].get("image")]
    limit = args.num_samples or int(config["experiment"]["num_samples"])
    manifest = [sid for sid in manifest if sid in cache][:limit]
    logger.info("channel experiment: %d samples", len(manifest))

    pragmatic_cfg = config["pragmatic"]
    if args.weights:
        weights = tuple(float(w) for w in args.weights)
    elif "weights" in pragmatic_cfg:
        weights = tuple(float(w) for w in pragmatic_cfg["weights"])
    else:
        # The experiment configs spell the same three numbers as lambda_* keys;
        # accept either spelling so a config cannot silently lose its weights.
        weights = (float(pragmatic_cfg["lambda_attention"]),
                   float(pragmatic_cfg["lambda_semantic"]),
                   float(pragmatic_cfg["lambda_gradient"]))
    if abs(sum(weights) - 1.0) > 1e-9 or any(w < 0 for w in weights):
        raise SystemExit(f"pragmatic weights must be nonnegative and sum to one, got {weights}")
    beta = float(config["pragmatic"]["beta"])
    if args.beta is not None:
        beta = float(args.beta)
    codebook = ProtectionCodebook(
        levels=tuple(float(v) for v in config["radio"]["protection"]["levels"]),
        information_bits_per_token=float(
            config["radio"]["protection"]["information_bits_per_token"]))
    bandwidth = float(config["radio"]["bandwidth"])
    slot_duration = float(config["radio"]["slot_duration"])
    epsilon_out = float(config["radio"]["epsilon_out"])
    t_max = int(config["scheduling"]["t_max"])
    do_refine = bool(config["scheduling"].get("refine", True))
    # "joint" uses P(both received) under the shared instantaneous channel;
    # "independent" restores the historical q_i * q_j coupling term for ablation.
    coupling_model = args.coupling_model or str(
        config["scheduling"].get("coupling_model", "joint"))
    if coupling_model not in ("joint", "independent"):
        raise SystemExit(f"unknown coupling model {coupling_model!r}")
    from src.pragmatic.coupling import build_coupling_variant
    coupling_variant = args.coupling_variant or str(
        pragmatic_cfg.get("coupling_variant", "raw"))
    lp_utility = args.lp_utility or str(config["scheduling"].get("lp_utility", "marginal"))
    body_mask_mode = args.body_mask or "off"
    protection_mode = args.protection_mode or str(
        config["scheduling"].get("protection_mode", "differentiated"))
    token_embeddings = {}
    if coupling_variant in ("A", "B"):
        if not Path(args.token_embeddings).exists():
            raise SystemExit(f"variant {coupling_variant} needs {args.token_embeddings}; "
                             f"run scripts/build_token_embeddings.py")
        token_embeddings = torch.load(args.token_embeddings, weights_only=False)
    logger.info("coupling: variant=%s model=%s lp_utility=%s beta=%s",
                coupling_variant, coupling_model, lp_utility, beta)
    # "v6_norestore" keeps the fair parts of v6 (RSU-issued task intent, letter-only
    # reply) but disables damage-triggered restoration. Restoration fires whenever
    # survived/original < RESTORE_THRESHOLD, i.e. at a rate set by how much each
    # method happens to deliver -- which rewards under-delivery and confounds the
    # cross-method comparison. Passing restore_threshold=0.0 makes the trigger
    # unreachable while keeping the answer prompt identical.
    protocol = args.receiver_protocol
    restore_threshold = (0.0 if protocol in ("v6_norestore", "v6_letter")
                         else RESTORE_THRESHOLD)
    if protocol == "v6_norestore":
        protocol = "v6"

    receiver_tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["receiver_path"], local_files_only=True)

    def _load_receiver(model_path: str):
        # Most receivers are text-only causal LMs; vision-language receivers
        # (e.g. Qwen3-VL-8B used in text-only mode) fall back to the
        # image-text-to-text class so the same driver supports both. Optional
        # 8-bit quantisation keeps a large receiver on a 24 GiB card; memory from
        # a failed first attempt is released before the fallback is tried.
        load_8bit = bool(config["model"].get("receiver_load_in_8bit", False))
        device_map = config["model"].get("receiver_device_map")
        kwargs = dict(local_files_only=True, attn_implementation="eager")
        if load_8bit:
            kwargs["load_in_8bit"] = True
        else:
            kwargs["dtype"] = torch.bfloat16
        if device_map:
            kwargs["device_map"] = device_map
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        except (ValueError, KeyError, RuntimeError, TypeError):
            gc.collect(); torch.cuda.empty_cache()
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(model_path, **kwargs)
        if not device_map:
            # Quantised (4/8-bit) models are already placed on the target device
            # by the quantiser; .to() is unsupported on old bitsandbytes builds.
            try:
                model = model.to("cuda:0")
            except (ValueError, RuntimeError):
                pass
        return model.eval()

    receiver = _load_receiver(config["model"]["receiver_path"])
    # The transmitted token ids come from the transmitter's vocabulary, so they
    # must be decoded with the transmitter tokenizer before the receiver
    # re-encodes the resulting text.
    transmitter_tokenizer = AutoProcessor.from_pretrained(
        config["model"]["transmitter_path"], local_files_only=True).tokenizer

    power = PowerSampler()
    logger.info("Power sampling backend: %s", power.backend or "unavailable")

    channel_paths = args.channels or config["channels"]
    snr_values = args.snr_db or [float(v) for v in config["snr_db"]]
    seeds = args.channel_seeds or [int(v) for v in config["experiment"]["channel_seeds"]]
    methods = args.methods or list(config["methods"])
    proposed_budget_mode = config["experiment"].get("proposed_budget", "conservative")
    # External compression baselines rank tokens with the receiver LM's
    # self-information, which lives in a sidecar rather than the pragmatic cache.
    LM_BASELINES = {"selective_context", "llmlingua", "longllmlingua",
                    "tokcom", "llmlingua2", "efpc"}
    TASK_AWARE = {"answer_loo", "answer_loo_uep", "answer_attn"}
    REQUIRED_FIELD = {"selective_context": ("surprisal",),
                      "llmlingua": ("surprisal",),
                      "longllmlingua": ("surprisal", "surprisal_q"),
                      "tokcom": ("tokcom",),
                      "llmlingua2": ("llmlingua2",),
                      "efpc": ("surprisal", "surprisal_q"),
                      "answer_loo": ("answer_loo",),
                      "answer_loo_uep": ("answer_loo",),
                      "answer_attn": ("answer_attn",)}
    if LM_BASELINES.intersection(methods):
        if not benchmark_scores:
            raise SystemExit("the LM baselines need the per-token self-information "
                             "sidecar; run scripts/build_benchmark_scores.py and pass "
                             "--benchmark-scores")

    if TASK_AWARE.intersection(methods) and not answer_scores:
        raise SystemExit("the answer-aware baselines need the LOO sidecar; run "
                         "scripts/build_answer_aware_scores.py and pass "
                         "--answer-scores")

    def complete(sid: str, method_name: str) -> bool:
        entry = (benchmark_scores.get(sid) if method_name in LM_BASELINES
                 else answer_scores.get(sid))
        if not entry:
            return False
        return all(len(entry.get(f, [])) == cache[sid].num_tokens
                   for f in REQUIRED_FIELD[method_name])

    lacking = [(sid, m) for sid in manifest for m in methods
               if (m in LM_BASELINES or m in TASK_AWARE)
               and not complete(sid, m)]
    if lacking:
        sample = lacking[0]
        raise SystemExit(
            f"{len(lacking)} (sample, method) pairs lack ranking scores, "
            f"e.g. {sample}; rebuild the sidecar with "
            f"scripts/build_benchmark_scores.py / build_answer_aware_scores.py")

    def benchmark_signal(method_name: str, sid: str):
        """Ranking signal of the external baselines; ``None`` when not needed."""
        if method_name in TASK_AWARE:
            field = ("answer_loo" if method_name.startswith("answer_loo")
                     else "answer_attn")
            return torch.tensor(answer_scores[sid][field], dtype=torch.float32)
        if method_name not in LM_BASELINES:
            return None
        entry = benchmark_scores[sid]

        def minmax(xs):
            lo, hi = min(xs), max(xs)
            return [(v - lo) / ((hi - lo) or 1.0) for v in xs]

        if method_name == "longllmlingua":
            # Contrastive relevance: how much the pending question explains the token.
            values = [a - b for a, b in zip(entry["surprisal"], entry["surprisal_q"])]
        elif method_name == "efpc":
            # EFPC (Cao et al., 2025) unifies task-aware and task-agnostic
            # compression. Its released scorer is a BERT trained on GPT-4
            # distillation; this training-free operationalisation combines the two
            # signals EFPC unifies: self-information (task-agnostic) and
            # question-conditioned relevance (task-aware).
            task_agnostic = minmax(entry["surprisal"])
            task_aware = minmax([a - b for a, b in
                                 zip(entry["surprisal"], entry["surprisal_q"])])
            values = [a + b for a, b in zip(task_agnostic, task_aware)]
        elif method_name in ("tokcom", "llmlingua2"):
            values = entry[method_name]
        else:
            values = entry["surprisal"]
        return torch.tensor(values, dtype=torch.float32)
    records = []

    for channel_path in channel_paths:
        base_spec = load_channel_spec(channel_path)
        for snr_db in snr_values:
            for seed in seeds:
                channel = build_channel({**base_spec, "snr_db": float(snr_db),
                                         "seed": int(seed)})
                logger.info("channel=%s snr=%s seed=%s", channel.name, snr_db, seed)

                for position, sample_id in enumerate(manifest):
                    scores = cache[sample_id]
                    row = processed[sample_id]
                    utilities = [float(u) for u in combine_scores(
                        scores.attention, scores.semantic, scores.gradient, weights)]
                    utility_tensor = torch.tensor(utilities, dtype=torch.float32)
                    # ``proposed_body`` runs the same scheduler with the duplicated
                    # question/option prefix removed from the candidate set: those
                    # tokens already reach the receiver through build_question_block,
                    # so symbols spent on them carry no information. The prefix takes
                    # 30% of the payload and 44% of the top-10% utility, which is
                    # precisely the budget a body-only selection gets back.
                    body_start = report_span_start(scores.token_strings,
                                                   row.get("choices"))
                    coupling = (torch.tensor(scores.coupling, dtype=torch.float32)
                                if scores.coupling is not None
                                else torch.zeros(scores.num_tokens, scores.num_tokens))
                    if coupling_variant != "raw":
                        coupling = build_coupling_variant(
                            coupling_variant, coupling,
                            token_embeddings[sample_id])
                    if args.coupling_shuffle:
                        # Symmetric permutation: same coefficients, wrong partners.
                        generator = torch.Generator().manual_seed(
                            20261012 + int(sample_id))
                        perm = torch.randperm(coupling.shape[0], generator=generator)
                        coupling = coupling[perm][:, perm]
                    # Eq. (19): the utility of a token given that the rest of the set
                    # is also received. Used as the LP's surrogate when --lp-utility
                    # context, so the coupling steers the selection and not only the
                    # local refinement.
                    selection_utility = None
                    if lp_utility == "context":
                        selection_utility = context_aware_utility(
                            utility_tensor, coupling, beta)

                    state = channel.sample((1,))
                    c_eps = reliable_spectral_efficiency(state, epsilon_out)
                    budget = slot_duration * bandwidth * float(c_eps[0])
                    plugin = slot_duration * bandwidth * math.log2(
                        1.0 + float(state.snr_linear) * float(state.h_hat.abs().pow(2)[0]))
                    realized = math.log2(1.0 + float(state.snr_linear)
                                         * float(state.h.abs().pow(2)[0]))
                    q = decoding_probability(state, codebook).reshape(-1)
                    # Pairwise reception probability. Every token of one report shares
                    # this instantaneous channel, so the scheduler must price a pair
                    # with Q[ell, m] (the weaker level), not with q_ell * q_m.
                    joint_q = (joint_decoding_probability(state, codebook).reshape(
                        codebook.num_levels, codebook.num_levels)
                        if coupling_model == "joint" else None)

                    rng = random.Random(int(seed) * 1000003 + position)
                    # "max" lets proposed use the genie (plug-in) capacity budget,
                    # i.e. the same budget as plugin_capacity -> structurally maximal.
                    proposed_budget = plugin if proposed_budget_mode == "max" else budget
                    for method in methods:
                        start = time.perf_counter()
                        is_plugin = method == "plugin_capacity"
                        use_body_mask = (method == "proposed_body"
                                         or (body_mask_mode == "all" and method != "full"))
                        candidate_mask = None
                        if use_body_mask and body_start > 0:
                            candidate_mask = torch.zeros(scores.num_tokens, dtype=torch.bool)
                            candidate_mask[body_start:] = True
                        # Budget routing. ``plugin_capacity`` *is* the internal
                        # baseline that replaces Eq. (6) with the plug-in capacity,
                        # and ``proposed`` may deliberately be run at the genie
                        # budget. Every other method -- all external baselines
                        # included -- must stay under the same realistic reliable
                        # budget of Eq. (6), otherwise proposed would win on
                        # budget instead of on the schedule.
                        if is_plugin:
                            method_budget = plugin
                        elif method == "proposed":
                            method_budget = proposed_budget
                        else:
                            method_budget = budget
                        # External lexical baselines rank tokens by the receiver
                        # LM's self-information instead of the pragmatic utility.
                        alt = benchmark_signal(method, sample_id)
                        sched_method = ("proposed"
                                        if is_plugin or method in ("tokcom_uep",
                                                                   "answer_loo_uep",
                                                                   "proposed_body")
                                        else method)
                        if method == "proposed_body":
                            # Same scheduler and the same budget as ``proposed``; the
                            # only change is that the duplicated prefix leaves the
                            # candidate set, so the freed symbols are re-optimised
                            # over the report body.
                            method_budget = proposed_budget
                        if protection_mode == "fixed" and sched_method == "proposed":
                            sched_method = "fixed_H"
                        if method in ("tokcom_uep", "answer_loo_uep"):
                            # An external importance signal driving *our* full
                            # scheduler (Lagrangian + certified rounding +
                            # differentiated protection): isolates the utility
                            # score -- the core contribution -- from the machinery
                            # around it. tokcom = 2025 cross-modal attention;
                            # answer_loo = the exact leave-one-out gold standard.
                            raw = (benchmark_scores[sample_id]["tokcom"]
                                   if method == "tokcom_uep"
                                   else answer_scores[sample_id]["answer_loo"])
                            lo, hi = min(raw), max(raw)
                            span = (hi - lo) or 1.0
                            utility = torch.tensor(
                                [(v - lo) / span for v in raw], dtype=torch.float32)
                            level = select_level(
                                sched_method, utility, coupling, beta, q,
                                torch.tensor(codebook.levels, dtype=torch.float32),
                                method_budget, t_max if do_refine else 0, rng,
                                joint_q=joint_q, selection_utility=selection_utility,
                                candidate_mask=candidate_mask)
                        else:
                            level = select_level(
                                sched_method,
                                utility_tensor, coupling, beta, q,
                                torch.tensor(codebook.levels, dtype=torch.float32),
                                method_budget,
                                t_max if do_refine else 0, rng, alt,
                                joint_q=joint_q, selection_utility=selection_utility,
                                candidate_mask=candidate_mask)
                        schedule_seconds = time.perf_counter() - start

                        selected = [i for i in range(scores.num_tokens) if level[i] > 0]
                        kept, lost = [], 0
                        for token in selected:
                            rate = float(codebook.code_rates[int(level[token]) - 1])
                            decoded = realized >= rate
                            dropped = rng.random() < float(state.packet_loss_rate)
                            if decoded and not dropped:
                                kept.append(token)
                            else:
                                lost += 1
                        received = channel.transmit(transmitter_tokenizer.decode(
                            [scores.token_ids[i] for i in kept], skip_special_tokens=True))

                        torch.cuda.reset_peak_memory_stats()
                        baseline = torch.cuda.memory_allocated() / 2 ** 30
                        watts_start = power.watts()
                        start = time.perf_counter()
                        answer = answer_with_protocol(
                            model=receiver, tokenizer=receiver_tokenizer, row=row,
                            received=received, survived_tokens=len(kept),
                            total_tokens=scores.num_tokens,
                            answer_request=ANSWER_REQUEST,
                            max_new_tokens=int(config["receiver"]["max_new_tokens"]),
                            protocol=protocol,
                            restore_threshold=restore_threshold)
                        receive_seconds = time.perf_counter() - start
                        watts_end = power.watts()
                        peak = torch.cuda.max_memory_allocated() / 2 ** 30

                        raw = answer["raw"]
                        prediction = answer["prediction"]
                        parse_mode = answer["parse_mode"]
                        ground_truth = str(row["choices"][row["answer"]])
                        symbols = sum(float(codebook.levels[int(level[i]) - 1])
                                      for i in selected)
                        records.append({
                            "sample_id": sample_id, "channel": channel.name, "snr_db": float(snr_db),
                            "channel_seed": int(seed),                             "method": method,
                            "receiver_protocol": args.receiver_protocol,
                            "coupling_model": coupling_model,
                            "coupling_variant": coupling_variant,
                            "lp_utility": lp_utility,
                            "protection_mode": protection_mode,
                            "coupling_shuffled": int(bool(args.coupling_shuffle)),
                            "sched_method": sched_method,
                            "beta": beta,
                            "original_tokens": scores.num_tokens,
                            "selected_tokens": len(selected), "received_tokens": len(kept),
                            "TRR": token_retention_ratio(len(selected), scores.num_tokens),
                            "transmitted_symbols": symbols,
                            "budget": budget, "plugin_budget": plugin,
                            "reliable_efficiency": float(c_eps[0]),
                            "realized_efficiency": realized,
                            "capacity_outage": int(slot_duration * bandwidth * realized < budget),
                            "plugin_outage": int(slot_duration * bandwidth * realized < plugin),
                            "token_erasure_rate": (lost / len(selected)) if selected else 0.0,
                            "any_token_lost": int(lost > 0),
                            "correct": exact_match(prediction, ground_truth),
                            "parse_mode": parse_mode,
                            "schedule_seconds": schedule_seconds,
                            "receive_seconds": receive_seconds,
                            "gpu_peak_memory_delta_gib": peak - baseline,
                            "energy_j": PowerSampler.joules(watts_start, watts_end,
                                                            receive_seconds),
                            "selected_text": answer["received_after"],
                            "restored": int(answer["restored"]),
                        })
                    gc.collect(); torch.cuda.empty_cache()
                frame = pd.DataFrame(records)
                frame.to_csv(output_dir / "channel_results.partial.csv", index=False)

    frame = pd.DataFrame(records)
    frame.to_csv(output_dir / "channel_results.csv", index=False)
    summary_rows = []
    for keys, group in frame.groupby(["channel", "snr_db", "method"], sort=True):
        stats = mean_std_ci95(group["correct"].astype(float).tolist())
        summary_rows.append({
            "channel": keys[0], "snr_db": float(keys[1]), "method": keys[2],
            "observations": len(group), "accuracy": stats["mean"],
            "accuracy_ci95": stats["ci95"],
            "mean_TRR": float(group["TRR"].mean()),
            "mean_symbols": float(group["transmitted_symbols"].mean()),
            "capacity_outage_rate": float(group["capacity_outage"].mean()),
            "plugin_outage_rate": float(group["plugin_outage"].mean()),
            "mean_token_erasure_rate": float(group["token_erasure_rate"].mean()),
            "mean_schedule_seconds": float(group["schedule_seconds"].mean()),
            "mean_receive_seconds": float(group["receive_seconds"].mean()),
            "mean_gpu_peak_memory_delta_gib": float(group["gpu_peak_memory_delta_gib"].mean()),
            "mean_energy_j": (float(group["energy_j"].mean())
                              if group["energy_j"].notna().any() else None)})
    summary = {"samples": len(manifest), "channels": [Path(p).stem for p in channel_paths],
               "snr_db": [float(v) for v in snr_values], "channel_seeds": list(seeds),
               "epsilon_out": epsilon_out, "methods": methods,
               "weights": list(weights), "beta": beta,
               "codebook_levels": list(codebook.levels),
               "code_rates": list(codebook.code_rates),
               "energy_backend": power.backend,
               "conditions": summary_rows}
    atomic_json(output_dir / "summary.json", summary)
    atomic_json(run / "summary.json", summary)
    frame.to_csv(run / "channel_results.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
