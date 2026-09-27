from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import DatasetAdapter, UnifiedSample, register_adapter

_RAW_SIDE = {
    "vehicle": "V2X-Seq-SPD-vehicle-side-image",
    "infrastructure": "V2X-Seq-SPD-infrastructure-side-image",
}
_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


@register_adapter("v2xqa", "v2x-qa")
class V2XQAAdapter(DatasetAdapter):
    """Adapter for the locally cloned V2X-QA annotations.

    Reads ``datasets/V2X-QA/data/splits/{split}/*_annotations.jsonl`` and resolves
    each ``image_id_vs`` / ``image_id_is`` to the raw images under
    ``datasets/V2X-QA/data/raw_external``. Mirrors the UnifiedSample contract of
    the other adapters so ``build_manifests --dataset v2xqa`` works unchanged.
    """

    name = "v2xqa"

    def __init__(self, split: str, data_dir=None, dataset=None) -> None:
        super().__init__(split, data_dir)
        root = Path(self.data_dir) if self.data_dir else Path("datasets/V2X-QA")
        self._root = root
        if dataset is not None:
            self._rows = dataset
            return
        split_dir = root / "data" / "splits" / split
        self._rows = []
        if split_dir.exists():
            for f in sorted(split_dir.glob("*_annotations.jsonl")):
                with f.open(encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if line:
                            self._rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self._rows)

    def _image(self, side: str, image_id) -> str | None:
        if not image_id:
            return None
        d = self._root / "data" / "raw_external" / _RAW_SIDE[side]
        for ext in _IMG_EXTS:
            p = d / f"{image_id}{ext}"
            if p.exists():
                return str(p)
        return None

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = dict(self._rows[index])
        options = row.get("options", {})
        choices = [str(options.get(k, "")) for k in ("A", "B", "C", "D")]
        sel = row.get("selected_option_id")
        answer = ord(sel) - 65 if isinstance(sel, str) and "A" <= sel <= "D" else -1
        vs, is_ = row.get("image_id_vs"), row.get("image_id_is")
        # Primary view used by the single-image pipeline: vehicle for VS/CO,
        # infrastructure for IS. The other view is preserved in metadata for R2.
        primary = self._image("vehicle", vs) or self._image("infrastructure", is_)
        sample = UnifiedSample(
            sample_id=str(index),
            dataset=self.name,
            split=self.split,
            image=primary,
            question=str(row.get("question", "")),
            choices=choices,
            ground_truth=str(choices[answer]) if 0 <= answer < len(choices) else "",
            answer_type="multiple_choice" if choices else "direct",
            metadata={
                "task_id": row.get("task_id"),
                "question_id": row.get("question_id"),
                "image_id_vs": vs,
                "image_id_is": is_,
                "image_path_is": self._image("infrastructure", is_),
                "canonical_answer": row.get("canonical_answer"),
                "original_sample_id": row.get("sample_id"),
            },
        ).to_dict()
        self.validate(sample)
        return sample
