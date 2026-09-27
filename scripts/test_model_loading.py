#!/usr/bin/env python
"""Single-GPU Qwen capability probe with auditable evidence per requirement."""
from __future__ import annotations

import argparse, gc, json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def record(results, name, fn):
    try: results[name] = {"status": "PASS", "evidence": fn()}
    except Exception as exc: results[name] = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transmitter", default="models/Qwen3-VL-2B-Instruct")
    parser.add_argument("--receiver", default="models/Qwen3-1.7B")
    parser.add_argument("--sample-jsonl", default="data/processed/scienceqa_validation_pilot.jsonl")
    parser.add_argument("--attention-implementation", default="eager", choices=["eager", "sdpa"])
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output", default="outputs/model_capabilities.json")
    args = parser.parse_args()
    if "," in args.gpu: raise SystemExit("Exactly one physical GPU ID is allowed")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import torch
    from PIL import Image
    from transformers import (AutoModelForCausalLM, AutoProcessor, AutoTokenizer,
                              Qwen3VLForConditionalGeneration, __version__ as transformers_version)
    results = {"runtime": {"physical_gpu": args.gpu, "visible_device_count": torch.cuda.device_count(),
                           "torch": torch.__version__, "transformers": transformers_version,
                           "attention_implementation": args.attention_implementation}}
    record(results, "cuda_available", lambda: {"name": torch.cuda.get_device_name(0), "count": torch.cuda.device_count()}
           if torch.cuda.is_available() and torch.cuda.device_count() == 1
           else (_ for _ in ()).throw(RuntimeError("single visible CUDA GPU required")))
    if results["cuda_available"]["status"] != "PASS":
        for name in ("transmitter_image_text_inference", "token_embeddings", "embedding_backward",
                     "attention_weights", "hidden_states", "bf16_or_fp16", "gpu_memory_fit", "receiver_loading"):
            results[name] = {"status": "BLOCKED", "error": "single-GPU CUDA prerequisite failed"}
    else:
        rows = [json.loads(line) for line in Path(args.sample_jsonl).read_text(encoding="utf-8").splitlines()]
        sample = next(row for row in rows if row.get("image"))
        processor = AutoProcessor.from_pretrained(args.transmitter, local_files_only=True)
        transmitter = Qwen3VLForConditionalGeneration.from_pretrained(
            args.transmitter, dtype=torch.bfloat16, attn_implementation=args.attention_implementation,
            local_files_only=True).to("cuda:0")
        transmitter.eval(); torch.cuda.reset_peak_memory_stats()
        messages = [{"role": "user", "content": [
            {"type": "image", "image": Image.open(sample["image"]).convert("RGB")},
            {"type": "text", "text": sample["question"]}]}]
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors="pt")
        inputs = {key: value.to("cuda:0") if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.no_grad():
            output = transmitter(**inputs, output_attentions=True, output_hidden_states=True, return_dict=True)
            generated = transmitter.generate(**inputs, max_new_tokens=8, do_sample=False)
        text = processor.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        record(results, "transmitter_image_text_inference", lambda: {"sample_id": sample["row_idx"],
               "input_shape": list(inputs["input_ids"].shape), "generated_text": text})
        record(results, "hidden_states", lambda: {"layers": len(output.hidden_states),
               "last_shape": list(output.hidden_states[-1].shape)} if output.hidden_states
               else (_ for _ in ()).throw(RuntimeError("no hidden states returned")))
        def attention_evidence():
            if not output.attentions or any(layer is None for layer in output.attentions):
                raise RuntimeError("genuine attention tensors were not returned")
            shapes = [list(layer.shape) for layer in output.attentions]
            if any(shape[-1] != inputs["input_ids"].shape[1] for shape in shapes):
                raise RuntimeError("attention/token alignment mismatch")
            return {"layers": len(shapes), "first_shape": shapes[0], "genuine_model_output": True}
        record(results, "attention_weights", attention_evidence)
        record(results, "bf16_or_fp16", lambda: {"model_dtype": str(next(transmitter.parameters()).dtype)})

        batch = processor(text="Question: What color is the sky? Answer: blue", return_tensors="pt")
        ids, mask = batch["input_ids"].to("cuda:0"), batch["attention_mask"].to("cuda:0")
        transmitter.train(); transmitter.zero_grad(set_to_none=True)
        embeddings = transmitter.get_input_embeddings()(ids).detach().requires_grad_(True)
        grad_output = transmitter(inputs_embeds=embeddings, attention_mask=mask, labels=ids, return_dict=True)
        grad_output.loss.backward(); gradient = embeddings.grad
        record(results, "token_embeddings", lambda: {"shape": list(embeddings.shape), "requires_grad": embeddings.requires_grad})
        record(results, "embedding_backward", lambda: {
            "probe_loss_type": "teacher_forced_prompt_self_nll_capability_only",
            "loss": float(grad_output.loss.detach()), "gradient_shape": list(gradient.shape),
            "finite": bool(torch.isfinite(gradient).all()), "nonzero": bool((gradient.norm(dim=-1) > 0).any())})
        transmitter_peak = torch.cuda.max_memory_allocated() / 2**30
        del output, generated, grad_output, gradient, embeddings, inputs, transmitter, processor
        gc.collect(); torch.cuda.empty_cache()

        tokenizer = AutoTokenizer.from_pretrained(args.receiver, local_files_only=True)
        receiver = AutoModelForCausalLM.from_pretrained(args.receiver, dtype=torch.bfloat16,
            attn_implementation=args.attention_implementation, local_files_only=True).to("cuda:0").eval()
        prompt = tokenizer.apply_chat_template([{"role": "user", "content":
            "Answer with only the option text. Question: What color is a clear daytime sky? Choices: red; blue; green"}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        receiver_inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        with torch.no_grad(): receiver_output = receiver.generate(**receiver_inputs, max_new_tokens=12, do_sample=False)
        prediction = tokenizer.decode(receiver_output[0, receiver_inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        record(results, "receiver_loading", lambda: {"prediction": prediction, "dtype": str(next(receiver.parameters()).dtype)})
        peak = max(transmitter_peak, torch.cuda.max_memory_allocated() / 2**30)
        record(results, "gpu_memory_fit", lambda: {"peak_allocated_gib": peak,
               "gpu_total_gib": torch.cuda.get_device_properties(0).total_memory / 2**30})

    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    if any(value.get("status") == "FAIL" for value in results.values() if isinstance(value, dict)):
        raise SystemExit(1)


if __name__ == "__main__": main()
