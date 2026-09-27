from __future__ import annotations

from typing import Any

from .base import DatasetAdapter, UnifiedSample, register_adapter


@register_adapter("aokvqa", "a-okvqa", "huggingfacem4/a-okvqa")
class AOKVQAAdapter(DatasetAdapter):
    name = "aokvqa"

    def __init__(self, split: str, data_dir=None, dataset=None) -> None:
        super().__init__(split, data_dir)
        if dataset is not None:
            self.data = dataset
        else:
            from datasets import load_dataset
            source = str(self.data_dir) if self.data_dir and self.data_dir.exists() else "HuggingFaceM4/A-OKVQA"
            self.data = load_dataset(source, split=split)

    def __len__(self) -> int: return len(self.data)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = dict(self.data[index])
        choices = list(row.get("choices") or []) or None
        answer = row.get("correct_choice_idx", row.get("answer"))
        if isinstance(answer, int) and choices and 0 <= answer < len(choices):
            ground_truth = choices[answer]
        else:
            direct = row.get("direct_answers") or answer or ""
            ground_truth = str(direct[0] if isinstance(direct, list) and direct else direct)
        sample = UnifiedSample(
            sample_id=str(row.get("question_id", row.get("id", index))), dataset=self.name,
            split=self.split, image=row.get("image"), question=str(row.get("question", "")),
            choices=choices, ground_truth=ground_truth,
            answer_type="multiple_choice" if choices else "direct",
            metadata={k: v for k, v in row.items() if k not in {"image", "question", "choices"}},
        ).to_dict()
        self.validate(sample)
        return sample

