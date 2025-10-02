"""Failure taxonomy based on retrieval + verification signals."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass
class FailureTag:
    name: str
    reason: str


class FailureAnalyzer:
    """Apply rule-based tagging to categorize model errors."""

    def __init__(self, rules: Sequence[dict]):
        self.rules = rules

    def analyze(self, example: dict) -> List[FailureTag]:
        tags: List[FailureTag] = []
        for rule in self.rules:
            name = rule.get("name", "unknown")
            cond = rule.get("condition", "")
            if self._evaluate(cond, example):
                tags.append(FailureTag(name=name, reason=self._describe(cond, example)))
        return tags

    def _evaluate(self, condition: str, example: dict) -> bool:
        condition = condition.strip().lower()
        if condition == "verifier_below_threshold":
            score = example.get("verifier_score", 0.0)
            threshold = example.get("verifier_threshold", 0.5)
            return score < threshold
        if condition == "contains_gold_string_is_false":
            return not example.get("answer_contains_gold", False)
        if condition == "has_binary_choice & answer_not_in_context":
            return example.get("binary_choice", False) and not example.get("answer_in_context", False)
        return False

    def _describe(self, condition: str, example: dict) -> str:
        if condition == "verifier_below_threshold":
            return (
                f"Verifier score {example.get('verifier_score', 0.0):.2f} < "
                f"threshold {example.get('verifier_threshold', 0.5):.2f}"
            )
        if condition == "contains_gold_string_is_false":
            return "Answer string does not match gold span." \
                + (
                    " Gold substring absent." if not example.get("answer_contains_gold") else ""
                )
        if condition == "has_binary_choice & answer_not_in_context":
            return "Question forces a choice but selected entity not present in top documents."
        return condition


def detect_binary_choice(question: str) -> bool:
    return bool(re.search(r"\b or \b", question.lower()))


def answer_in_context(answer: str, documents: Sequence[dict]) -> bool:
    answer_norm = answer.lower().strip()
    if not answer_norm:
        return False
    for doc in documents:
        if answer_norm in doc.get("title", "").lower():
            return True
        if answer_norm in doc.get("text", "").lower():
            return True
    return False
