"""Answer refinement and augmentation strategies."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .llm import LLMGenerator


def build_refinement_prompt(dataset: str, question: str, contexts: List[str]) -> str:
    evidence = "\n\n".join(contexts[:3])
    if dataset == "boolq":
        return (
            "You are answering a yes/no question. Respond with a single word, 'yes' or 'no'.\n"
            "Evidence you must rely on:\n"
            f"{evidence}\n\n"
            f"Question: {question}\n"
            "Answer with 'yes' or 'no' only."
        )
    if dataset == "squad_v2":
        return (
            "Answer the question by copying an exact span from the evidence."
            " If the evidence does not contain an answer, respond with 'unanswerable'.\n"
            "Evidence:\n"
            f"{evidence}\n\n"
            f"Question: {question}\n"
            "Answer (exact span or 'unanswerable'):"
        )
    return (
        "Answer the question concisely using the evidence. Cite the key fact.\n"
        f"Evidence:\n{evidence}\n\nQuestion: {question}\nAnswer:"
    )


def refine_answer(
    dataset: str,
    question: str,
    contexts: List[str],
    llm: LLMGenerator,
    reason: str,
) -> Tuple[str, Dict[str, str]]:
    """Regenerate answer with stricter instructions."""
    prompt = build_refinement_prompt(dataset, question, contexts)
    regenerated = llm.generate(prompt)
    return regenerated, {"reason": reason, "prompt": prompt}
