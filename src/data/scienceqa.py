from __future__ import annotations

from typing import Any

from .base import DatasetAdapter, UnifiedSample, register_adapter


@register_adapter("scienceqa", "derek-thomas/scienceqa")
class ScienceQAAdapter(DatasetAdapter):
    name = "scienceqa"

    def __init__(self, split: str, data_dir=None, dataset=None) -> None:
        super().__init__(split, data_dir)
        if dataset is not None:
            self.data = dataset
        elif self.data_dir and self.data_dir.is_file() and self.data_dir.suffix == ".jsonl":
            import json
            with self.data_dir.open(encoding="utf-8") as handle:
                self.data = [json.loads(line) for line in handle if line.strip()]
        else:
            from datasets import load_dataset
            source = str(self.data_dir) if self.data_dir and self.data_dir.exists() else "derek-thomas/ScienceQA"
            self.data = load_dataset(source, split=split)

    def __len__(self) -> int: return len(self.data)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = dict(self.data[index])
        choices = list(row.get("choices") or []) or None
        answer = row.get("answer")
        if isinstance(answer, int) and choices and 0 <= answer < len(choices):
            ground_truth = choices[answer]
        else:
            ground_truth = str(answer or "")
        sample = UnifiedSample(
            sample_id=str(row.get("id", row.get("row_idx", index))), dataset=self.name, split=self.split,
            image=row.get("image"), question=str(row.get("question", "")), choices=choices,
            ground_truth=ground_truth, answer_type="multiple_choice" if choices else "direct",
            metadata={k: v for k, v in row.items() if k not in {"image", "question", "choices"}},
        ).to_dict()
        self.validate(sample)
        return sample
