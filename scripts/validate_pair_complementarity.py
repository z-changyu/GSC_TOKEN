#!/usr/bin/env python
"""Group 1b -- does the coupling M_ij identify *complementary* token pairs?

Eq. (17) rewards co-receiving a pair with ``beta M_ij u'_i u'_j``. That is only the
right thing to do if ``M_ij`` is large exactly when the two tokens complete each
other, and small when they say the same thing twice. This script measures the real
interaction instead of assuming it:

For a token pair (i, j), holding **all other context fixed**, the report is sent in
four variants -- neither / i only / j only / both -- and the RSU's answer confidence
margin is measured for each. The true interaction is the difference in differences

    I_true(i,j) = [m(both) - m(i only)] - [m(j only) - m(neither)],

i.e. the extra value of j *given* i. ``I_true > 0`` means the pair is complementary
(j helps more when i is present); ``I_true < 0`` means j is redundant given i.

Pairs are mined from the reports in four families the paper cares about:
object-position, object-motion, negation-scope and verbatim repetition. Repetition
is the control: its true interaction should be ~0 or negative, and a coupling score
that rewards it is rewarding redundancy.

Ground truth is used only here, offline, for evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OBJECTS = (r"cars?|vehicles?|trucks?|vans?|buses?|bus|sedans?|taxis?|pedestrians?|"
           r"cyclists?|motorcycles?|bicycles?|traffic lights?|signs?|lanes?|roads?|"
           r"intersection|crosswalks?|barriers?|cones?|buildings?|trees?")
POSITIONS = (r"left|right|centre|center|ahead|behind|front|rear|side|near|far|"
             r"foreground|background|distant|distance")
MOTIONS = (r"moving|moves|stopped|stopping|parked|turning|turns|approaching|crossing|"
           r"waiting|standing|driving|passing|overtaking|traveling|travelling")
NEGATIONS = r"\b(no|not|without|none|absent|never)\b"


def token_spans(tokenizer, token_ids: list[int]):
    """Character span of every token in the decoded report."""
    full = tokenizer.decode(token_ids)
    spans, pos = [], 0
    for tid in token_ids:
        piece = tokenizer.decode([tid])
        idx = full.find(piece, pos) if piece else -1
        if idx < 0:
            idx = pos
        spans.append((idx, idx + max(len(piece), 0)))
        pos = max(idx + len(piece), pos)
    return full, spans


def token_at(spans, char_index: int) -> int:
    for i, (s, e) in enumerate(spans):
        if s <= char_index < e:
            return i
    return -1


def mine_pairs(text: str, spans, max_per_type: int, seed: int):
    """Candidate token-index pairs by family, in document order."""
    rng = random.Random(seed)

    def first_token(pattern: str, start: int = 0):
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            if m.start() < start:
                continue
            t = token_at(spans, m.start())
            if t >= 0:
                return t, m.start(), m.end()
        return None

    out = {}
    # object-position and object-motion: nearest partner inside the same sentence
    for name, partner in (("object_position", POSITIONS), ("object_motion", MOTIONS)):
        pairs = []
        for m in re.finditer(OBJECTS, text, flags=re.IGNORECASE):
            obj_tok = token_at(spans, m.start())
            if obj_tok < 0:
                continue
            window_end = m.end() + 120
            best = None
            for pm in re.finditer(partner, text[m.end():window_end], flags=re.IGNORECASE):
                ptok = token_at(spans, m.end() + pm.start())
                if ptok >= 0 and ptok != obj_tok:
                    best = ptok
                    break
            if best is not None:
                pairs.append(tuple(sorted((obj_tok, best))))
        out[name] = list(dict.fromkeys(pairs))
    # negation scope: the cue and the next object it governs
    neg = []
    for m in re.finditer(NEGATIONS, text, flags=re.IGNORECASE):
        cue = token_at(spans, m.start())
        after = re.search(OBJECTS, text[m.end():m.end() + 80], flags=re.IGNORECASE)
        if cue < 0 or not after:
            continue
        obj = token_at(spans, m.end() + after.start())
        if obj >= 0 and obj != cue:
            neg.append(tuple(sorted((cue, obj))))
    out["negation"] = list(dict.fromkeys(neg))
    # verbatim repetition: repeated word bigrams
    words = re.findall(r"[A-Za-z]+", text.lower())
    seen: dict[tuple[str, str], int] = {}
    rep = []
    for k in range(len(words) - 1):
        bigram = (words[k], words[k + 1])
        if bigram in seen:
            first = seen[bigram]
            a = [m.start() for m in re.finditer(re.escape(bigram[0]), text, re.IGNORECASE)]
            b_spans = [m.start() for m in re.finditer(re.escape(bigram[1]), text, re.IGNORECASE)]
            if a and b_spans:
                ta = token_at(spans, a[0]); tb = token_at(spans, b_spans[-1])
                if ta >= 0 and tb >= 0 and ta != tb:
                    rep.append(tuple(sorted((ta, tb))))
            if len(rep) >= max_per_type * 2:
                break
        else:
            seen[bigram] = k
    out["repetition"] = list(dict.fromkeys(rep))

    rng.shuffle(out.get("repetition", []))
    return {k: v[:max_per_type] for k, v in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/v2x_edge_edge_8b_rx8b_smoke.yaml")
    parser.add_argument("--score-cache", default=None)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--pairs-per-type", type=int, default=3)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--coupling-variant", default="raw", choices=["raw", "row", "A", "B"])
    parser.add_argument("--token-embeddings", default="outputs/cache/v2xqa_token_embeddings_8b.pt")
    parser.add_argument("--output", default="outputs/pair_complementarity.json")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import yaml
    from transformers import AutoModelForCausalLM, AutoProcessor

    from src.experiments.prompts import ANSWER_REQUEST
    from src.experiments.receiver_protocol import (answer_letter_only,
                                                   build_question_block)
    from src.experiments.scoring import combine_scores
    from src.pragmatic.coupling import build_coupling_variant
    from src.utils.io import read_jsonl

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    pragmatic = config["pragmatic"]
    weights = tuple(pragmatic.get("weights") or [
        pragmatic["lambda_attention"], pragmatic["lambda_semantic"], pragmatic["lambda_gradient"]])
    cache_path = args.score_cache or config["score_cache"]
    rows = {str(r["sample_id"]): r for r in
            (json.loads(line) for line in open(cache_path, encoding="utf-8"))}
    processed = {str(r["row_idx"]): r for r in read_jsonl(config["dataset"]["processed"])}
    manifest = [str(e["sample_id"]) for e in read_jsonl(config["dataset"]["manifest"])]
    manifest = [s for s in manifest if s in rows and s in processed][:args.num_samples]
    print(f"{len(manifest)} samples | beta={args.beta} | pairs/type={args.pairs_per_type} | "
          f"coupling={args.coupling_variant}")
    token_embeddings = {}
    if args.coupling_variant != "raw":
        if not Path(args.token_embeddings).exists():
            raise SystemExit(f"variant {args.coupling_variant} needs {args.token_embeddings}")
        token_embeddings = {str(k): v for k, v in
                            torch.load(args.token_embeddings, weights_only=False).items()}

    tx_tokenizer = AutoProcessor.from_pretrained(
        config["model"]["transmitter_path"], local_files_only=True).tokenizer
    rx_path = config["model"]["receiver_path"]
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

    def margin(row, kept) -> float:
        text = tx_tokenizer.decode([row["_ids"][i] for i in sorted(kept)],
                                   skip_special_tokens=True).strip()
        out = answer_letter_only(model=receiver, tokenizer=rx_tokenizer, row=row["_row"],
                                 received=text, answer_request=ANSWER_REQUEST,
                                 question_block=build_question_block(row["_row"]))
        scores = out["letter_scores"]
        picked = (row["_choices"].index(out["prediction"])
                  if out["prediction"] in row["_choices"] else -1)
        if picked < 0:
            return float("nan")
        others = [s for k, s in enumerate(scores) if k != picked]
        return float(scores[picked] - (max(others) if others else 0.0))

    records = []
    for sid in manifest:
        entry = rows[sid]; proc = processed[sid]
        token_ids = list(entry["token_ids"])
        text, spans = token_spans(tx_tokenizer, token_ids)
        utility = combine_scores(entry["attention"], entry["semantic"], entry["gradient"],
                                 weights).float()
        coupling = torch.tensor(entry["coupling"], dtype=torch.float32)
        if args.coupling_variant != "raw":
            coupling = build_coupling_variant(args.coupling_variant, coupling,
                                              token_embeddings[sid])
        ctx = {"_ids": token_ids, "_row": proc,
               "_choices": list(proc["choices"])}
        mined = mine_pairs(text, spans, args.pairs_per_type, int(sid))
        n = len(token_ids)
        for family, pairs in mined.items():
            for (i, j) in pairs:
                base = [t for t in range(n) if t not in (i, j)]
                m = {
                    "neither": margin(ctx, base),
                    "i_only": margin(ctx, base + [i]),
                    "j_only": margin(ctx, base + [j]),
                    "both": margin(ctx, base + [i, j]),
                }
                true_interaction = (m["both"] - m["i_only"]) - (m["j_only"] - m["neither"])
                m_ij = float(coupling[min(i, j), max(i, j)])
                proxy_interaction = args.beta * m_ij * float(utility[i]) * float(utility[j])
                records.append({
                    "sample_id": sid, "family": family, "i": i, "j": j,
                    "M_ij": m_ij, "u_i": float(utility[i]), "u_j": float(utility[j]),
                    "proxy_interaction": proxy_interaction,
                    "true_interaction": true_interaction, **{f"m_{k}": v for k, v in m.items()},
                })
                print(f"  {sid:>4s} {family:16s} ({i:3d},{j:3d}) M={m_ij:.4f} "
                      f"proxy={proxy_interaction:+.4f} true={true_interaction:+.3f}")

    import numpy as np
    from scipy.stats import spearmanr

    print("\n=== 按类型：真实交互 vs 代理交互 ===")
    print(f"{'类型':18s} {'n':>4s} {'真交互均值':>11s} {'代理均值':>10s} {'Spearman':>10s} {'互补对占比':>11s}")
    by_family = {}
    for fam in ("object_position", "object_motion", "negation", "repetition"):
        sub = [r for r in records if r["family"] == fam]
        if not sub:
            continue
        ti = np.array([r["true_interaction"] for r in sub])
        pi = np.array([r["proxy_interaction"] for r in sub])
        sp = (float(spearmanr(pi, ti).statistic)
              if np.std(pi) > 1e-12 and np.std(ti) > 1e-12 else float("nan"))
        by_family[fam] = {"n": len(sub), "true_mean": float(ti.mean()),
                          "proxy_mean": float(pi.mean()), "spearman": sp,
                          "positive_share": float((ti > 0).mean())}
        print(f"{fam:18s} {len(sub):4d} {ti.mean():+11.3f} {pi.mean():+10.4f} "
              f"{sp:10.3f} {(ti > 0).mean()*100:10.1f}%")

    all_ti = np.array([r["true_interaction"] for r in records])
    all_pi = np.array([r["proxy_interaction"] for r in records])
    pooled = (float(spearmanr(all_pi, all_ti).statistic)
              if np.std(all_pi) > 1e-12 and np.std(all_ti) > 1e-12 else float("nan"))
    print(f"\n合并全部 {len(records)} 对: 真交互均值={all_ti.mean():+.3f}  "
          f"代理均值={all_pi.mean():+.4f}  Spearman={pooled:.3f}")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"samples": len(manifest), "beta": args.beta,
                               "by_family": by_family, "pooled_spearman": pooled,
                               "records": records}, indent=2), encoding="utf-8")
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
