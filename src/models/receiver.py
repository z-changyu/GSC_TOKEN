class Receiver:
    def __init__(self, model, tokenizer) -> None: self.model, self.tokenizer = model, tokenizer

    def answer(self, prompt: str, **generation_kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, **generation_kwargs)
        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

