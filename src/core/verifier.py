"""Answer verification utilities using NLI-style classifiers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class VerificationResult:
    """Stores verifier scores for an answer vs. multiple passages."""

    score: float
    supporting_passages: List[str]


class AnswerVerifier:
    """Cross-encoder verifier to judge whether evidence supports an answer."""

    def __init__(self, model_id: str, device: str = "cuda", threshold: float = 0.5) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        self.threshold = threshold
        # Determine positive label index for NLI models
        self.entailment_index = self._detect_entailment_label()

    def _detect_entailment_label(self) -> int:
        if self.model.config.label2id:
            for label, idx in self.model.config.label2id.items():
                if label.lower() in {"entailment", "entails", "yes"}:
                    return int(idx)
        # Default to last label (often entailment)
        return int(self.model.config.num_labels - 1)

    @torch.inference_mode()
    def score_answer(self, question: str, answer: str, passages: Iterable[str]) -> VerificationResult:
        support_scores: List[float] = []
        for passage in passages:
            premise = passage
            if answer:
                hypothesis = (
                    f"For the question '{question}', the correct answer is {answer}."
                )
            else:
                hypothesis = question
            encoded = self.tokenizer(
                premise,
                hypothesis,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**encoded).logits[0]
            prob = torch.softmax(logits, dim=-1)[self.entailment_index].item()
            support_scores.append(prob)
        best_score = max(support_scores) if support_scores else 0.0
        supporting_passages = [passage for passage, score in zip(passages, support_scores) if score >= self.threshold]
        return VerificationResult(score=best_score, supporting_passages=supporting_passages)

    def is_supported(self, score: float) -> bool:
        return score >= self.threshold


def maybe_create_verifier(config: dict) -> Optional[AnswerVerifier]:
    verifier_cfg = config.get("verifier", {})
    if not verifier_cfg.get("enabled", False):
        return None
    return AnswerVerifier(
        model_id=verifier_cfg["model_id"],
        device=verifier_cfg.get("device", "cuda"),
        threshold=verifier_cfg.get("threshold", 0.5),
    )
