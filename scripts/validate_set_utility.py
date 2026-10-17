#!/usr/bin/env python
"""Group 1a -- does the set-utility proxy predict what the RSU actually does?

The paper optimises a set function ``F_T(T) = sum_{i in T} u'_i + beta sum_{i<j in T}
M_ij u'_i u'_j`` (Eq. 17) as a stand-in for the RSU's judgment quality. This script
tests that stand-in directly:

* the report, the question and the RSU model are held fixed;
* different *received sets* are constructed -- scheduler-produced (differentiated
  protection, and top-k under a perfect channel), random deletions of matched size,
  and the full report -- over tight / medium / wide symbol budgets;
* for every set we record the proxy utility (additive and set), the RSU's
  teacher-forced NLL of the ground truth, and whether the RSU answers correctly.

It then reports the rank correlation and prediction error of the two proxies against
the two ground-truth quality measures, and splits the proxy behaviour by whether the
terminal's own (label-free) prediction was right.

Ground truth is used **only** here, offline, for evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/v2x_edge_edge_8b_rx8b_smoke.yaml")
    parser.add_argument("--score-cache", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--num-samples", type=int, default=20)
    # Budgets are chosen so the scheduler still produces a *strict subset* of the
    # report: at 4000 symbols it simply selects everything and the tier degenerates.
    parser.add_argument("--budgets", type=float, nargs="+", default=[400.0, 800.0, 1500.0])
    parser.add_argument("--random-per-budget", type=int, default=6)
    parser.add_argument("--coupling-variant", default="raw", choices=["raw", "row", "A", "B"])
    parser.add_argument("--token-embeddings", default="outputs/cache/v2xqa_token_embeddings_8b.pt")
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--output", default="outputs/set_utility_validation.json")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import yaml
    from src.channel import build_channel, decoding_probability
    from src.channel.link import ProtectionCodebook, joint_decoding_probability
    from src.channel.reliable_capacity import reliable_spectral_efficiency
    from src.experiments.prompts import (ANSWER_REQUEST, build_choice_block,
                                         canonical_prediction)
    from src.experiments.receiver_protocol import (COORDINATOR_LEAD_IN,
                                                   answer_letter_only,
                                                   build_question_block)
    from src.experiments.scoring import combine_scores
    from src.pragmatic.coupling import build_coupling_variant
    from src.pragmatic.utility import set_utility
    from src.scheduling.methods import select_level
    from src.utils.io import read_jsonl

    from transformers import AutoModelForCausalLM, AutoProcessor

    def load_channel_spec(path: str) -> dict:
        spec = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        spec.setdefault("name", Path(path).stem)
        return spec

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    pragmatic = config["pragmatic"]
    weights = tuple(pragmatic.get("weights") or [
        pragmatic["lambda_attention"], pragmatic["lambda_semantic"], pragmatic["lambda_gradient"]])
    codebook = ProtectionCodebook(
        levels=tuple(float(v) for v in config["radio"]["protection"]["levels"]),
        information_bits_per_token=float(
            config["radio"]["protection"]["information_bits_per_token"]))
    rates = torch.tensor(codebook.levels, dtype=torch.float32)
    t_max = int(config["scheduling"]["t_max"])

    cache_path = args.score_cache or config["score_cache"]
    rows = {str(r["sample_id"]): r for r in
            (json.loads(line) for line in open(cache_path, encoding="utf-8"))}
    processed = {str(r["row_idx"]): r for r in read_jsonl(config["dataset"]["processed"])}
    manifest = [str(e["sample_id"]) for e in
                read_jsonl(args.manifest or config["dataset"]["manifest"])]
    manifest = [sid for sid in manifest if sid in rows and sid in processed][:args.num_samples]
    print(f"{len(manifest)} samples | cache={Path(cache_path).name} | beta={args.beta} | "
          f"coupling={args.coupling_variant}")
    token_embeddings = {}
    if args.coupling_variant != "raw":
        if not Path(args.token_embeddings).exists():
            raise SystemExit(f"variant {args.coupling_variant} needs {args.token_embeddings}")
        token_embeddings = {str(k): v for k, v in
                            torch.load(args.token_embeddings, weights_only=False).items()}

    # ---------------------------------------------------------------- models
    tx_path = config["model"]["transmitter_path"]
    rx_path = config["model"]["receiver_path"]
    tx_tokenizer = AutoProcessor.from_pretrained(tx_path, local_files_only=True).tokenizer
    rx_tokenizer = AutoProcessor.from_pretrained(rx_path, local_files_only=True).tokenizer
    try:
        receiver = AutoModelForCausalLM.from_pretrained(
            rx_path, dtype=torch.bfloat16, attn_implementation="eager",
            local_files_only=True).to("cuda:0").eval()
    except Exception:                                # noqa: BLE001
        from transformers import AutoModelForImageTextToText
        receiver = AutoModelForImageTextToText.from_pretrained(
            rx_path, dtype=torch.bfloat16, attn_implementation="eager",
            local_files_only=True).to("cuda:0").eval()
    im_end = rx_tokenizer.convert_tokens_to_ids("<|im_end|>")

    # ------------------------------------------------- channels used to build sets
    fixed = build_channel({**load_channel_spec("configs/channels/rayleigh.yaml"),
                           "snr_db": 15.0, "seed": 0})
    state = fixed.sample((1,))
    q_eval = decoding_probability(state, codebook).reshape(-1)
    joint_eval = joint_decoding_probability(state, codebook).reshape(
        codebook.num_levels, codebook.num_levels)

    def scheduler_set(utility, coupling, budget, joint, rng):
        return select_level("proposed", utility, coupling, args.beta, q_eval, rates,
                            float(budget), t_max, rng, joint_q=joint)

    def received_text(token_ids, kept):
        return tx_tokenizer.decode([token_ids[i] for i in sorted(kept)],
                                   skip_special_tokens=True).strip()

    def rsu_nll(row, received, truth):
        content = (f"{build_question_block(row)}\n\n{COORDINATOR_LEAD_IN}\n{received}\n\n"
                   f"{ANSWER_REQUEST}")
        chat = rx_tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        prompt_ids = rx_tokenizer(chat, return_tensors="pt")["input_ids"].to("cuda:0")
        answer_ids = rx_tokenizer(truth, add_special_tokens=False)["input_ids"] + [im_end]
        tail = torch.tensor([answer_ids], device="cuda:0")
        input_ids = torch.cat([prompt_ids, tail], dim=1)
        labels = torch.full_like(input_ids, -100)
        labels[0, -len(answer_ids):] = tail[0]
        with torch.no_grad():
            loss = float(receiver(input_ids=input_ids, labels=labels).loss)
        del prompt_ids, input_ids, labels, tail
        return loss

    records = []
    for sid in manifest:
        row = processed[sid]
        entry = rows[sid]
        token_ids = list(entry["token_ids"])
        n = len(token_ids)
        utility = combine_scores(entry["attention"], entry["semantic"], entry["gradient"],
                                 weights).float()
        coupling = torch.tensor(entry["coupling"], dtype=torch.float32)
        if args.coupling_variant != "raw":
            coupling = build_coupling_variant(args.coupling_variant, coupling,
                                              token_embeddings[sid])
        choices = list(row["choices"])
        truth = str(choices[int(row["answer"])])

        sets: dict[str, list[int]] = {}
        for budget in args.budgets:
            tag = f"b{int(budget)}"
            rng = random.Random(int(sid) * 1000003 + int(budget))
            level = scheduler_set(utility, coupling, budget, joint_eval, rng)
            sch = [i for i in range(n) if int(level[i]) > 0]
            sets[f"sched_{tag}"] = sch
            for k in range(args.random_per_budget):
                draw = random.Random(hash((sid, budget, k)) % (2 ** 31))
                sets[f"rand_{tag}_{k}"] = sorted(draw.sample(range(n), min(len(sch), n)))
            rng2 = random.Random(int(sid) * 1000003 + int(budget) + 7)
            level_top = scheduler_set(utility, coupling, budget, None, rng2)
            sets[f"topk_{tag}"] = [i for i in range(n) if int(level_top[i]) > 0]
        sets["full"] = list(range(n))

        pred_ok = int(entry.get("predicted_choice") is not None
                      and int(entry["predicted_choice"]) == int(row["answer"]))
        for name, kept in sets.items():
            kept = sorted(set(int(i) for i in kept))
            if not kept:
                continue
            text = received_text(token_ids, kept)
            additive = float(utility[kept].sum())
            proxy_set = float(set_utility(torch.tensor(kept), utility, coupling, args.beta))
            nll = rsu_nll(row, text, truth)
            # One forward pass gives both the answer and its confidence margin.
            # The margin is the length-robust quality measure: NLL drifts with how
            # much redundant context is present, the margin does not.
            letter = answer_letter_only(
                model=receiver, tokenizer=rx_tokenizer, row=row, received=text,
                answer_request=ANSWER_REQUEST, question_block=build_question_block(row))
            scores = letter["letter_scores"]
            picked = choices.index(letter["prediction"]) if letter["prediction"] in choices else -1
            margin = float("nan")
            if picked >= 0:
                others = [s for k, s in enumerate(scores) if k != picked]
                margin = float(scores[picked] - (max(others) if others else 0.0))
            records.append({
                "sample_id": sid, "set": name, "kept": len(kept), "tokens": n,
                "proxy_additive": additive, "proxy_set": proxy_set,
                "rsu_nll": nll, "margin": margin,
                "correct": int(picked == int(row["answer"])),
                "predicted_ok": pred_ok, "chars": len(text)})
            del text
        print(f"  {sid}: {len(sets)} sets   n_tokens={n}  pred_ok={pred_ok}")

    # ------------------------------------------------------------------ analysis
    import numpy as np
    from scipy.stats import spearmanr

    def corr(rows_):
        if len(rows_) < 4:
            return None
        add = np.array([r["proxy_additive"] for r in rows_])
        st = np.array([r["proxy_set"] for r in rows_])
        nll = np.array([r["rsu_nll"] for r in rows_])
        mar = np.array([r["margin"] for r in rows_])
        cor = np.array([r["correct"] for r in rows_])
        out = {}
        with np.errstate(invalid="ignore"):
            out["additive_vs_nll"] = float(spearmanr(add, nll).statistic)
            out["set_vs_nll"] = float(spearmanr(st, nll).statistic)
            if np.std(mar) > 1e-9:
                out["additive_vs_margin"] = float(spearmanr(add, mar).statistic)
                out["set_vs_margin"] = float(spearmanr(st, mar).statistic)
            if len(set(cor.tolist())) > 1:
                out["additive_vs_correct"] = float(spearmanr(add, cor).statistic)
                out["set_vs_correct"] = float(spearmanr(st, cor).statistic)
            out["additive_vs_set"] = float(spearmanr(add, st).statistic)
        return out

    per_sample = {}
    for sid in manifest:
        subset = [r for r in records if r["sample_id"] == sid]
        c = corr(subset)
        if c:
            per_sample[sid] = c

    def agg(key):
        vals = [v[key] for v in per_sample.values() if key in v and math.isfinite(v[key])]
        return (statistics.mean(vals), statistics.median(vals), len(vals)) if vals else None

    print("\n=== 代理 vs RSU 真实质量（逐样本 Spearman，再取均值）===")
    for key in ("additive_vs_margin", "set_vs_margin", "additive_vs_nll", "set_vs_nll",
                "additive_vs_correct", "set_vs_correct", "additive_vs_set"):
        a = agg(key)
        if a:
            print(f"  {key:24s} mean={a[0]:+.3f} median={a[1]:+.3f}  (n={a[2]} 样本)")

    print("\n=== 按端侧自预测是否正确分组 ===")
    for ok in (1, 0):
        sub = [r for r in records if r["predicted_ok"] == ok]
        print(f"  predicted_ok={ok}: {len(sub)} 个集合, "
              f"NLL 均值={statistics.mean([r['rsu_nll'] for r in sub]):.3f}, "
              f"正确率={statistics.mean([r['correct'] for r in sub])*100:.1f}%")

    print("\n=== 各集合族的平均质量（按 kept token 数归一）===")
    fams: dict[str, list] = {}
    for r in records:
        fam = r["set"].split("_b")[0]
        fams.setdefault(fam, []).append(r)
    for fam, rs in sorted(fams.items()):
        print(f"  {fam:8s} n={len(rs):4d}  kept={statistics.mean([r['kept'] for r in rs]):5.1f}  "
              f"NLL={statistics.mean([r['rsu_nll'] for r in rs]):6.3f}  "
              f"acc={statistics.mean([r['correct'] for r in rs])*100:5.1f}%  "
              f"proxy_set={statistics.mean([r['proxy_set'] for r in rs]):6.3f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cache": cache_path, "beta": args.beta,
                               "coupling_variant": args.coupling_variant,
                               "samples": len(manifest), "budgets": args.budgets,
                               "per_sample_correlation": per_sample,
                               "records": records}, indent=2), encoding="utf-8")
    print(f"\n[done] -> {out}")


if __name__ == "__main__":
    main()
