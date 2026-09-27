from __future__ import annotations

from typing import Any

from .base import DatasetAdapter, UnifiedSample, register_adapter


@register_adapter("gqa", "vikhyatk/gqa-val")
class GQAAdapter(DatasetAdapter):
    name = "gqa"

    def __init__(self, split: str = "val", data_dir=None, dataset=None) -> None:
        super().__init__(split, data_dir)
        if dataset is not None:
            raw = dataset
        else:
            from datasets import load_dataset
            source = str(self.data_dir) if self.data_dir and self.data_dir.exists() else "vikhyatk/gqa-val"
            raw = load_dataset(source, split=split)
        self.data: list[dict[str, Any]] = []
        for image_index, row_ in enumerate(raw):
            row = dict(row_)
            # Repository schemas have appeared both flattened and grouped. Preserve either.
            qas = row.get("qa") or row.get("questions")
            if isinstance(qas, list):
                for qa_index, qa in enumerate(qas):
                    merged = dict(row)
                    merged.update(dict(qa))
                    merged["_image_index"] = image_index
                    merged["_qa_index"] = qa_index
                    self.data.append(merged)
            else:
                self.data.append(row)

    def __len__(self) -> int: return len(self.data)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.data[index]
        sid = row.get("question_id", row.get("id", index))
        sample = UnifiedSample(
            sample_id=str(sid), dataset=self.name, split=self.split, image=row.get("image"),
            question=str(row.get("question", "")), choices=None,
            ground_truth=str(row.get("answer", row.get("label", ""))), answer_type="direct",
            metadata={k: v for k, v in row.items() if k not in {"image", "question", "answer", "label"}},
        ).to_dict()
        self.validate(sample)
        return sample

