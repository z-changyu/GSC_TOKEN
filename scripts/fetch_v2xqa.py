#!/usr/bin/env python
"""Build a GSC-Token ``processed`` manifest for V2X-QA from the local clone.

Mirrors ``scripts/fetch_scienceqa_pilot.py`` but reads the already-downloaded
V2X-QA annotations under ``datasets/V2X-QA/data/splits/{train,test}`` and resolves
each ``image_id_vs`` / ``image_id_is`` to the raw images in
``datasets/V2X-QA/data/raw_external``.

Output schema matches the ScienceQA processed file exactly, so the existing pilot /
channel / CV drivers work unchanged::

    row_idx, sample_id, image, question, choices, answer, task_id,
    question_id, image_id_vs, image_id_is, image_path_is, canonical_answer

``answer`` is the 0-based index of ``selected_option_id`` into ``choices``.
``sample_id`` equals ``str(row_idx)`` (a sequential integer) so the framework's
``int(sample_id)`` RNG-seed convention and ``processed[sample_id]`` lookups hold.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.io import write_jsonl  # noqa: E402


RAW_SIDE = {
    "vehicle": "V2X-Seq-SPD-vehicle-side-image",
    "infrastructure": "V2X-Seq-SPD-infrastructure-side-image",
}
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def resolve_image(root: Path, side: str, image_id) -> str | None:
    if not image_id:
        return None
    d = root / "data" / "raw_external" / RAW_SIDE[side]
    for ext in IMG_EXTS:
        p = d / f"{image_id}{ext}"
        if p.exists():
            return str(p)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["train", "test"])
    parser.add_argument("--data-root", default="datasets/V2X-QA")
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="Restrict to these task ids, e.g. VS1 VS2 CO3. Default: all.")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Deterministically sub-sample this many rows.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default=None, help="processed jsonl path")
    parser.add_argument("--manifest", default=None, help="manifest jsonl path")
    args = parser.parse_args()

    root = Path(args.data_root).resolve()
    split_dir = root / "data" / "splits" / args.split
    if not split_dir.exists():
        raise SystemExit(f"split dir not found: {split_dir}")

    files = sorted(split_dir.glob("*_annotations.jsonl"))
    rows = []
    for f in files:
        task = f.stem.replace("_annotations", "")
        if args.tasks and task not in args.tasks:
            continue
        with f.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append((task, json.loads(line)))

    if args.num_samples is not None and args.num_samples < len(rows):
        rows = random.Random(args.seed).sample(rows, args.num_samples)

    out_rows = []
    for idx, (task, row) in enumerate(rows):
        options = row.get("options", {})
        choices = [str(options.get(k, "")) for k in ("A", "B", "C", "D")]
        sel = row.get("selected_option_id")
        answer = ord(sel) - 65 if isinstance(sel, str) and "A" <= sel <= "D" else -1
        vs = row.get("image_id_vs")
        is_ = row.get("image_id_is")
        # Primary view (single-image pipeline): vehicle for VS/CO, infra for IS.
        primary = resolve_image(root, "vehicle", vs) or resolve_image(root, "infrastructure", is_)
        image_path_is = resolve_image(root, "infrastructure", is_)
        out_rows.append({
            "row_idx": idx,
            "sample_id": str(idx),
            "image": primary,
            "question": str(row.get("question", "")),
            "choices": choices,
            "answer": answer,
            "task_id": task,
            "question_id": row.get("question_id"),
            "image_id_vs": vs,
            "image_id_is": is_,
            "image_path_is": image_path_is,
            "canonical_answer": row.get("canonical_answer"),
        })

    out_path = Path(args.output or f"data/processed/v2xqa_{args.split}.jsonl")
    man_path = Path(args.manifest or f"data/manifests/v2xqa_{args.split}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    man_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, out_rows)
    write_jsonl(man_path, [{"sample_id": str(r["row_idx"]), "dataset": "v2xqa",
                            "split": args.split} for r in out_rows])
    missing = sum(1 for r in out_rows if not r["image"])
    print(json.dumps({"split": args.split, "samples": len(out_rows),
                      "missing_image": missing, "output": str(out_path),
                      "manifest": str(man_path)}, indent=2))


if __name__ == "__main__":
    main()
