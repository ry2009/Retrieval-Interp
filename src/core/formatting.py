"""Answer formatting utilities for dataset-specific templates."""
from __future__ import annotations

import difflib
from typing import List, Tuple


def format_boolq_answer(raw_answer: str) -> Tuple[str, dict]:
    """Return a strict yes/no answer inferred from free-form text."""
    text = raw_answer.strip().lower()
    tokens = text.replace(".", " ").split()

    metadata = {"strategy": "heuristic", "confident": False}

    if not text:
        return "", metadata

    yes_markers = {"yes", "yeah", "yep", "true", "certainly", "affirmative"}
    no_markers = {"no", "nope", "false", "cannot", "can't", "never"}

    first_tokens = tokens[:4]
    if any(tok in yes_markers for tok in first_tokens) and not any(
        tok in no_markers for tok in first_tokens
    ):
        metadata["confident"] = True
        return "yes", metadata
    if any(tok in no_markers for tok in first_tokens) and not any(
        tok in yes_markers for tok in first_tokens
    ):
        metadata["confident"] = True
        return "no", metadata

    # fallback: simple majority of markers in text
    yes_count = sum(tok in yes_markers for tok in tokens)
    no_count = sum(tok in no_markers for tok in tokens)
    if yes_count > no_count and yes_count:
        return "yes", metadata
    if no_count > yes_count and no_count:
        return "no", metadata

    return "", metadata


def format_squad_answer(raw_answer: str, contexts: List[str]) -> Tuple[str, dict]:
    """Map free-form text to an exact span present in contexts."""
    answer = raw_answer.strip()
    metadata = {"strategy": "exact"}
    if not answer:
        return "", metadata

    for ctx in contexts:
        if answer in ctx:
            return answer, metadata

    # Try longest matching substring using difflib
    best_match = ""
    best_ratio = 0.0
    for ctx in contexts:
        matcher = difflib.SequenceMatcher(a=ctx.lower(), b=answer.lower())
        match = matcher.find_longest_match(0, len(ctx), 0, len(answer))
        if match.size > 0:
            span = ctx[match.a : match.a + match.size]
            ratio = match.size / max(len(answer), 1)
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = span

    if best_match:
        metadata["strategy"] = "fuzzy"
        metadata["ratio"] = best_ratio
        return best_match.strip(), metadata

    metadata["strategy"] = "raw"
    return answer, metadata


def apply_formatting(dataset: str, raw_answer: str, contexts: List[str]) -> Tuple[str, dict]:
    """Dispatch to dataset-specific formatter."""
    if dataset == "boolq":
        return format_boolq_answer(raw_answer)
    if dataset == "squad_v2":
        return format_squad_answer(raw_answer, contexts)
    return raw_answer.strip(), {"strategy": "none"}
