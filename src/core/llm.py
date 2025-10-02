"""LLM generation utilities."""
from __future__ import annotations

from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMGenerator:
    """Thin wrapper around a Hugging Face causal LM."""

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 0.9,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            # Some causal LMs lack an explicit pad token; reuse EOS.
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None,
        )
        self.generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0.0,
        }

    @torch.inference_mode()
    def generate(self, prompt: str) -> str:
        """Generate a response for ``prompt``."""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(**inputs, **self.generation_kwargs)
        # Remove prompt tokens
        generated = output_ids[0, inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def build_prompt(self, question: str, contexts: List[str]) -> str:
        joined_context = "\n\n".join(contexts)
        return (
            "You are a retrieval-augmented assistant. Use the supplied evidence to answer the question.\n"
            "Evidence:\n"
            f"{joined_context}\n\n"
            f"Question: {question}\n"
            "Answer concisely and cite the most relevant facts."
        )
