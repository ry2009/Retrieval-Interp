"""Dataset loaders for retrieval experiments."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from datasets import load_dataset


@dataclass
class Document:
    """Container for a single retrieved document."""

    doc_id: str
    title: str
    text: str
    is_supporting: bool
    sample_id: str


@dataclass
class QAExample:
    """Question/answer pair with linked documents."""

    sample_id: str
    question: str
    answer: str
    supporting_doc_ids: Sequence[str]


def load_hotpotqa_subset(
    split: str,
    sample_size: int,
    max_contexts_per_question: int,
    seed: int,
) -> Tuple[List[QAExample], Dict[str, Document], Dict[str, List[str]]]:
    """Load a thin slice of HotpotQA distractor split.

    Parameters
    ----------
    split:
        Dataset split name (e.g., "train" or "validation").
    sample_size:
        Number of QA examples to sample.
    max_contexts_per_question:
        Upper bound on contexts kept per question (defaults to HotpotQA's 10).
    seed:
        RNG seed for reproducible sampling.

    Returns
    -------
    examples:
        List of QAExample entries.
    corpus:
        Mapping of ``doc_id`` to Document metadata and text.
    """

    dataset = load_dataset("hotpot_qa", "distractor", split=split)
    rng = random.Random(seed)

    # Ensure deterministic subset selection.
    all_indices = list(range(len(dataset)))
    rng.shuffle(all_indices)
    take_indices = all_indices[:sample_size]

    corpus: Dict[str, Document] = {}
    examples: List[QAExample] = []

    for idx in take_indices:
        row = dataset[idx]
        sample_id = str(row.get("_id") or row.get("id") or idx)
        supporting_titles = set(row["supporting_facts"].get("title", []))

        supporting_doc_ids: List[str] = []
        context_titles: Sequence[str] = row["context"].get("title", [])
        context_sentences: Sequence[Sequence[str]] = row["context"].get("sentences", [])

        for title, sentences in list(zip(context_titles, context_sentences))[:max_contexts_per_question]:
            text = " ".join(sentences)
            doc_id = f"{sample_id}::{title}"
            is_supporting = title in supporting_titles

            if doc_id not in corpus:
                corpus[doc_id] = Document(
                    doc_id=doc_id,
                    title=title,
                    text=text,
                    is_supporting=is_supporting,
                    sample_id=sample_id,
                )

            if is_supporting:
                supporting_doc_ids.append(doc_id)

        examples.append(
            QAExample(
                sample_id=sample_id,
                question=row["question"],
                answer=row["answer"],
                supporting_doc_ids=supporting_doc_ids,
            )
        )

    sample_to_doc_ids = {}
    for example in examples:
        doc_ids = [doc_id for doc_id in corpus if corpus[doc_id].sample_id == example.sample_id]
        sample_to_doc_ids[example.sample_id] = doc_ids

    return examples, corpus, sample_to_doc_ids


def load_squad_subset(
    split: str,
    sample_size: int,
    seed: int,
) -> Tuple[List[QAExample], Dict[str, Document], Dict[str, List[str]]]:
    dataset = load_dataset("squad_v2", split=split)
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    take_indices = indices[:sample_size]

    corpus: Dict[str, Document] = {}
    examples: List[QAExample] = []
    sample_to_doc_ids: Dict[str, List[str]] = {}

    for idx in take_indices:
        row = dataset[idx]
        sample_id = str(row["id"])
        context = row["context"]
        title = row.get("title") or "context"
        doc_id = f"{sample_id}::context"

        corpus[doc_id] = Document(
            doc_id=doc_id,
            title=title,
            text=context,
            is_supporting=True,
            sample_id=sample_id,
        )
        sample_to_doc_ids[sample_id] = [doc_id]

        answers = row.get("answers", {})
        answer_text = ""
        if answers and answers.get("text"):
            answer_text = answers["text"][0]

        examples.append(
            QAExample(
                sample_id=sample_id,
                question=row["question"],
                answer=answer_text,
                supporting_doc_ids=[doc_id] if answer_text else [doc_id],
            )
        )

    return examples, corpus, sample_to_doc_ids


def load_boolq_subset(
    split: str,
    sample_size: int,
    seed: int,
) -> Tuple[List[QAExample], Dict[str, Document], Dict[str, List[str]]]:
    dataset = load_dataset("boolq", split=split)
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    take_indices = indices[:sample_size]

    corpus: Dict[str, Document] = {}
    examples: List[QAExample] = []
    sample_to_doc_ids: Dict[str, List[str]] = {}

    for idx in take_indices:
        row = dataset[idx]
        sample_id = f"boolq-{idx}"
        doc_id = f"{sample_id}::passage"
        passage = row["passage"]

        corpus[doc_id] = Document(
            doc_id=doc_id,
            title="passage",
            text=passage,
            is_supporting=True,
            sample_id=sample_id,
        )
        sample_to_doc_ids[sample_id] = [doc_id]

        answer = "yes" if bool(row["answer"]) else "no"

        examples.append(
            QAExample(
                sample_id=sample_id,
                question=row["question"],
                answer=answer,
                supporting_doc_ids=[doc_id],
            )
        )

    return examples, corpus, sample_to_doc_ids


def load_dataset_subset(
    name: str,
    split: str,
    sample_size: int,
    seed: int,
    max_contexts_per_question: int = 10,
) -> Tuple[List[QAExample], Dict[str, Document], Dict[str, List[str]]]:
    if name == "hotpotqa":
        return load_hotpotqa_subset(split, sample_size, max_contexts_per_question, seed)
    if name == "squad_v2":
        return load_squad_subset(split, sample_size, seed)
    if name == "boolq":
        return load_boolq_subset(split, sample_size, seed)
    raise ValueError(f"Unsupported dataset: {name}")
