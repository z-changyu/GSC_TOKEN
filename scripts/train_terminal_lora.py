#!/usr/bin/env python
"""Experiment B (step 2): LoRA-distil the teacher's reports into the small terminal.

Trains the small vision-language terminal to imitate the strong terminal's semantic
reports (``scripts/gen_teacher_reports.py`` output), with the loss applied **only to
the report tokens** — the question/choice prefix is masked, so the student is not
asked to relearn the prompt.

The adapter is merged into the base weights and exported as a standalone checkpoint,
so every downstream driver can use it via ``model.transmitter_path`` without code
changes (the pragmatic score cache must of course be rebuilt: the reports change).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class ReportDataset:
    """(image, question, choices) -> teacher report, as a chat sample."""

    def __init__(self, rows: list[dict], processor, max_seq_len: int,
                 request_template: str) -> None:
        self.rows = rows
        self.processor = processor
        self.max_seq_len = max_seq_len
        self.request_template = request_template

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        from PIL import Image

        row = self.rows[idx]
        choices = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(row["choices"]))
        # Byte-identical to the main experiment's transmitter prompt: the student is
        # later driven by that exact prompt, so it must be trained on it.
        request = self.request_template.format(question=row["question"], choices=choices)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": row["image"]},
            {"type": "text", "text": request},
        ]}]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        image = Image.open(row["image"]).convert("RGB")
        target = " " + row["report"].strip()

        prompt_inputs = self.processor(text=[prompt], images=[image],
                                       return_tensors="pt", truncation=False)
        prompt_len = int(prompt_inputs["input_ids"].shape[1])
        full = self.processor(text=[prompt + target], images=[image],
                              return_tensors="pt", truncation=False)
        input_ids = full["input_ids"][0]
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        if input_ids.shape[0] > self.max_seq_len:
            # Keep the HEAD: the image placeholder tokens live at the start of the
            # prompt, so cutting from the front would leave fewer image tokens than
            # the vision tower produced ("Image features and image tokens do not
            # match"). Trimming the tail only shortens the supervised report.
            input_ids = input_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
        item = {"input_ids": input_ids, "labels": labels,
                "attention_mask": input_ids.new_ones(input_ids.shape[0])}
        for key, value in full.items():
            if key in ("input_ids", "attention_mask"):
                continue
            tensor = value[0]
            if key == "pixel_values":
                continue                               # re-add in the collator
            item[key] = tensor
        item["pixel_values"] = full["pixel_values"][0]
        item["image_grid_thw"] = full["image_grid_thw"][0]
        return item


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", default="outputs/distill/teacher_reports_8b.jsonl")
    parser.add_argument("--student", default="models/Qwen3-VL-2B-Instruct")
    parser.add_argument("--output-dir", default="outputs/distill/lora_2b")
    parser.add_argument("--merged-dir", default="models/Qwen3-VL-2B-Distilled",
                        help="standalone checkpoint exported after training")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=640,
                        help="image tokens sit at the head of the sequence, so this "
                             "must exceed image + prompt + a useful report prefix")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--attn", default="eager", choices=["eager", "sdpa"],
                        help="attention implementation; training never reads attention "
                             "weights, so sdpa is faster and lighter than eager")
    parser.add_argument("--device", default="0")
    parser.add_argument("--config",
                        default="configs/experiments/v2x_edge_edge_main_2b4b_gpu1.yaml",
                        help="supplies the transmitter prompt version")
    parser.add_argument("--no-merge", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    import yaml
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoImageProcessor, AutoProcessor,
                              Qwen3VLForConditionalGeneration, Trainer, TrainingArguments)
    from src.experiments.prompts import select_transmitter_request
    from src.utils.io import read_jsonl

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    prompt_version = str(config.get("transmitter", {}).get("prompt_version", "v1"))
    request_template = select_transmitter_request(config)
    rows = [r for r in read_jsonl(args.reports) if r.get("report")]
    if args.max_samples:
        rows = rows[:args.max_samples]
    print(f"student={args.student}  train reports={len(rows)}  prompt={prompt_version}")
    if not rows:
        raise SystemExit("no teacher reports found")

    processor = AutoProcessor.from_pretrained(args.student, local_files_only=True)
    processor.image_processor = AutoImageProcessor.from_pretrained(
        args.student, min_pixels=25088, max_pixels=200704, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.student, dtype=torch.bfloat16, attn_implementation=args.attn,
        local_files_only=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    dataset = ReportDataset(rows, processor, args.max_seq_len, request_template)

    def collate(items: list[dict]) -> dict:
        max_len = max(int(i["input_ids"].shape[0]) for i in items)
        batch = {}
        for key in ("input_ids", "labels", "attention_mask"):
            padded = []
            for item in items:
                tensor = item[key]
                pad = max_len - tensor.shape[0]
                padded.append(torch.cat([tensor,
                                         torch.zeros(pad, dtype=tensor.dtype)]))
            batch[key] = torch.stack(padded)
        batch["pixel_values"] = torch.cat([i["pixel_values"] for i in items], dim=0)
        grid = [i["image_grid_thw"] for i in items]
        batch["image_grid_thw"] = (torch.stack(grid) if all(g.dim() == 1 for g in grid)
                                   else torch.cat([g.view(-1, 3) for g in grid], dim=0))
        return batch

    steps = max(1, int(len(dataset) * args.epochs / (args.batch_size * args.grad_accum)))
    training = TrainingArguments(
        output_dir=args.output_dir, per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr,
        num_train_epochs=args.epochs, warmup_ratio=0.03, lr_scheduler_type="cosine",
        bf16=True, logging_steps=max(1, steps // 20), save_strategy="no",
        report_to=[], gradient_checkpointing=True, dataloader_num_workers=2,
        remove_unused_columns=False, max_grad_norm=1.0)
    trainer = Trainer(model=model, args=training, train_dataset=dataset,
                      data_collator=collate)
    trainer.train()
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"adapter saved -> {args.output_dir}")

    if not args.no_merge:
        merged = model.merge_and_unload()
        merged.save_pretrained(args.merged_dir, safe_serialization=True)
        processor.save_pretrained(args.merged_dir)
        print(f"merged standalone checkpoint -> {args.merged_dir}")
    Path(args.output_dir, "train_meta.json").write_text(
        json.dumps({"reports": args.reports, "samples": len(rows), "epochs": args.epochs,
                    "lr": args.lr, "lora_r": args.lora_r, "max_seq_len": args.max_seq_len,
                    "student": args.student, "prompt_version": prompt_version},
                   indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
