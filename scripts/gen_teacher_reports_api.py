#!/usr/bin/env python
"""Teacher reports from a hosted multimodal API, for distillation.

Same contract as ``scripts/gen_teacher_reports.py`` (the local Qwen3-VL-8B teacher):
one JSONL row per sample with the byte-identical V1 transmitter request, so the
downstream LoRA training and every evaluation driver need no changes.

Why a hosted teacher: the local 8B teacher is only 13.3pp better than the 4B student
in the controllable region, which turned out to be too small a gap to distil
(measured: 76.7% -> 65-78%, i.e. no reliable gain). A frontier multimodal teacher
raises the headroom itself.

The key is read from ``~/.config/gsc/deepseek_key`` (or ``$DEEPSEEK_API_KEY``) and is
never written into the repository.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE_URL = "https://api.deepseek.com"


def load_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key
    path = Path.home() / ".config" / "gsc" / "deepseek_key"
    if path.exists():
        return path.read_text().strip()
    raise SystemExit("no API key: set DEEPSEEK_API_KEY or write ~/.config/gsc/deepseek_key")


def encode_image(path: str, max_side: int) -> str:
    from PIL import Image

    image = Image.open(path).convert("RGB")
    if max_side and max(image.size) > max_side:
        scale = max_side / max(image.size)
        image = image.resize((max(1, int(image.width * scale)),
                              max(1, int(image.height * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/v2x_edge_edge_main_4b8b.yaml")
    parser.add_argument("--processed", default="data/processed/v2xqa_train.jsonl")
    parser.add_argument("--manifest", default="data/manifests/v2xqa_train.jsonl")
    parser.add_argument("--model", default="deepseek-flash")
    # deepseek-flash reasons by default and would spend the whole budget inside
    # `reasoning_content`, leaving `content` empty. {"type": "disabled"} is the
    # switch that turns the reasoning phase off entirely (verified: 1 completion
    # token instead of 35, content produced directly).
    parser.add_argument("--thinking", default="disabled", choices=["disabled", "auto"])
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--max-side", type=int, default=768, help="image downscale cap, px")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", default="outputs/distill/teacher_reports_api.jsonl")
    parser.add_argument("--device", default=None, help="unused; present for symmetry")
    args = parser.parse_args()

    import requests
    import yaml
    from src.experiments.prompts import (SYSTEM_PROMPT, build_choice_block,
                                         select_transmitter_request)
    from src.utils.io import read_jsonl

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    request_template = select_transmitter_request(config)
    processed = {str(r["row_idx"]): r for r in read_jsonl(args.processed)}
    manifest = [str(e["sample_id"]) for e in read_jsonl(args.manifest)
                if str(e["sample_id"]) in processed][:args.num_samples]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if out_path.exists():
        done = {str(r["sample_id"]): r for r in read_jsonl(out_path)}
    todo = [sid for sid in manifest if sid not in done]
    print(f"teacher={args.model}  samples={len(manifest)}  cached={len(done)}  to_do={len(todo)}")

    key = load_key()
    session = threading.local()

    def client() -> "requests.Session":
        s = getattr(session, "s", None)
        if s is None:
            s = requests.Session()
            s.headers.update({"Authorization": f"Bearer {key}",
                              "Content-Type": "application/json"})
            session.s = s
        return s

    lock = threading.Lock()
    stats = {"ok": 0, "fail": 0, "pt": 0, "ct": 0}

    def one(sid: str) -> dict:
        row = processed[sid]
        request = request_template.format(
            question=row["question"], choices=build_choice_block(row["choices"]))
        content = []
        if row.get("image"):
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + encode_image(row["image"], args.max_side)}})
        content.append({"type": "text", "text": request})
        payload = {"model": args.model,
                   "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": content}],
                   "max_tokens": args.max_tokens,
                   "temperature": 0}
        if args.thinking == "disabled":
            payload["thinking"] = {"type": "disabled"}
        last = ""
        for attempt in range(4):
            try:
                resp = client().post(f"{BASE_URL}/chat/completions", json=payload,
                                     timeout=args.timeout)
                if resp.status_code != 200:
                    last = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    time.sleep(2 * (attempt + 1))
                    continue
                data = resp.json()
                text = data["choices"][0]["message"]["content"].strip()
                usage = data.get("usage", {})
                return {"sample_id": sid, "report": text,
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens")}
            except Exception as exc:                 # noqa: BLE001
                last = f"{type(exc).__name__}: {exc}"
                time.sleep(2 * (attempt + 1))
        return {"sample_id": sid, "report": "", "error": last}

    def flush() -> None:
        with out_path.open("w", encoding="utf-8") as fh:
            for sid in manifest:
                rec = done.get(sid)
                if not rec:
                    continue
                row = processed[sid]
                fh.write(json.dumps({"sample_id": sid, "question": row["question"],
                                     "choices": row["choices"], "answer": row["answer"],
                                     "image": row.get("image"),
                                     "report": rec["report"],
                                     "teacher": args.model}, ensure_ascii=False) + "\n")

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(one, sid): sid for sid in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            with lock:
                if rec.get("report"):
                    done[rec["sample_id"]] = rec
                    stats["ok"] += 1
                    stats["pt"] += rec.get("prompt_tokens") or 0
                    stats["ct"] += rec.get("completion_tokens") or 0
                else:
                    stats["fail"] += 1
                if n % 25 == 0 or n == len(todo):
                    rate = n / (time.perf_counter() - t0)
                    print(f"  {n}/{len(todo)}  ok={stats['ok']} fail={stats['fail']} "
                          f"({rate:.2f}/s, tokens p={stats['pt']} c={stats['ct']})")
                    flush()
    flush()
    print(f"DONE ok={stats['ok']} fail={stats['fail']} -> {out_path}")
    print(f"tokens: prompt={stats['pt']} completion={stats['ct']}")
    print(f"elapsed {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
