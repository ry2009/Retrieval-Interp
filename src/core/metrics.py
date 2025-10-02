"""Evaluation metrics for retrieval and QA."""
from __future__ import annotations

import collections
import re
from typing import Iterable, Sequence

import numpy as np


_WORD_SPLIT = re.compile(r"\w+")


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace."""

    return " ".join(_WORD_SPLIT.findall(text.lower()))


def exact_match(prediction: str, ground_truth: str) -> float:
    """Binary exact-match score."""

    return float(normalize_text(prediction) == normalize_text(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1."""

    pred_tokens = _WORD_SPLIT.findall(normalize_text(prediction))
    gold_tokens = _WORD_SPLIT.findall(normalize_text(ground_truth))

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counter = collections.Counter(pred_tokens)
    gold_counter = collections.Counter(gold_tokens)
    shared = pred_counter & gold_counter

    precision = sum(shared.values()) / max(len(pred_tokens), 1)
    recall = sum(shared.values()) / max(len(gold_tokens), 1)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def retrieval_hit_rate(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Hit rate@k for supporting documents."""

    top_k = set(retrieved[:k])
    return float(any(doc_id in top_k for doc_id in gold))


def mean_reciprocal_rank(retrieved: Sequence[str], gold: Sequence[str]) -> float:
    """Compute MRR for the first relevant document."""

    rank_positions = [idx for idx, doc_id in enumerate(retrieved, start=1) if doc_id in gold]
    if not rank_positions:
        return 0.0
    return 1.0 / min(rank_positions)


def aggregate(values: Iterable[float]) -> dict[str, float]:
    """Return mean and std for a list of metric values."""

    arr = np.array(list(values), dtype=float)
    return {
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std(ddof=0)) if arr.size else 0.0,
    }
