#!/usr/bin/env python
"""Build a deterministic ScienceQA subset from the locally downloaded snapshot.

The pilot subset was originally fetched through the Hugging Face Dataset Viewer
API, which caps the sample count and requires network access. The full snapshot
under ``datasets/ScienceQA`` is already on disk, so this script rebuilds the same
processed schema offline and without that dependency.

Source layout
-------------
``datasets/ScienceQA/problems.json``   pid -> problem record (choices and answer
                                       are stored as string reprs)
``datasets/ScienceQA/pid_splits.json`` split name -> list of pids
``datasets/ScienceQA/<split>/<split>/<pid>/image.png``
"""
from __future__ import annotations

import argparse, ast, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io import write_jsonl


def load_problems(data_dir: Path) -> dict:
    return json.loads((data_dir / "problems.json").read_text(encoding="utf-8"))


def load_split(data_dir: Path, split: str) -> list[str]:
    splits = json.loads((data_dir / "pid_splits.json").read_text(encoding="utf-8"))
    if split not in splits:
        raise SystemExit(f"unknown split {split!r}; available: {sorted(splits)}")
    return list(splits[split])


def coerce_choices(value):
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return list(ast.literal_eval(value))


def build_row(pid: str, problem: dict, data_dir: Path, split: str) -> dict | None:
    image_name = problem.get("image")
    image_path = None
    if image_name:
        candidate = data_dir / split / split / str(pid) / Path(str(image_name)).name
        if not candidate.exists():
            return None
        image_path = str(candidate)
    choices = coerce_choices(problem.get("choices"))
    answer = problem.get("answer")
    answer = int(answer) if isinstance(answer, (int, str)) and str(answer).isdigit() else answer
    if not isinstance(answer, int) or not 0 <= answer < len(choices):
        return None
    return {
        "question": problem.get("question", ""),
        "choices": choices,
        "answer": answer,
        "hint": problem.get("hint", ""),
        "image": image_path,
        "lecture": problem.get("lecture", ""),
        "solution": problem.get("solution", ""),
        "task": problem.get("task", ""),
        "grade": problem.get("grade", ""),
        "subject": problem.get("subject", ""),
        "topic": problem.get("topic", ""),
        "category": problem.get("category", ""),
        "skill": problem.get("skill", ""),
        "row_idx": str(pid),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="datasets/ScienceQA")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--num-samples", type=int, default=150)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--require-image", action="store_true", default=True)
    parser.add_argument("--no-require-image", dest="require_image", action="store_false")
    parser.add_argument("--output", default="data/processed/scienceqa_validation_local.jsonl")
    parser.add_argument("--manifest", default="data/manifests/scienceqa_val_local.jsonl")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    problems = load_problems(data_dir)
    try:
        pids = load_split(data_dir, args.split)
    except SystemExit:
        # The snapshot uses "val" while the pilot manifests say "validation".
        pids = load_split(data_dir, "val")

    candidates = []
    for pid in pids:
        problem = problems.get(str(pid))
        if problem is None:
            continue
        row = build_row(str(pid), problem, data_dir, "val")
        if row is None:
            continue
        if args.require_image and row["image"] is None:
            continue
        if not row["question"] or not row["choices"]:
            continue
        candidates.append(row)

    if args.num_samples and args.num_samples < len(candidates):
        selected = sorted(random.Random(args.seed).sample(range(len(candidates)), args.num_samples))
        rows = [candidates[i] for i in selected]
    else:
        rows = candidates

    write_jsonl(args.output, rows)
    write_jsonl(args.manifest, [{"sample_id": row["row_idx"], "dataset": "scienceqa",
                                  "split": args.split} for row in rows])
    with_image = sum(row["image"] is not None for row in rows)
    print(json.dumps({"split": args.split, "candidates": len(candidates),
                      "samples": len(rows), "with_image": with_image,
                      "seed": args.seed, "output": args.output, "manifest": args.manifest},
                     indent=2))


if __name__ == "__main__":
    main()
