"""Experiment orchestration."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch

from . import analysis
from . import data
from . import metrics
from .llm import LLMGenerator
from .retrieval import DocEmbedding, LateInteractionRetriever
from .verifier import maybe_create_verifier
from .visualization import export_heatmaps
from .formatting import apply_formatting
from .augmentation import refine_answer


@dataclass
class RetrievalResult:
    sample_id: str
    question: str
    answer: str
    llm_answer: str
    supporting_doc_ids: List[str]
    top_docs: List[dict]
    em: float
    f1: float
    hit_at_k: float
    mrr: float


class ExperimentRunner:
    """Run the full retrieval + generation pipeline."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.output_dir = Path(config["evaluation"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        retriever_cfg = config["retriever"]
        self.retriever = LateInteractionRetriever(
            model_id=retriever_cfg["model_id"],
            device=retriever_cfg.get("device", "cuda"),
            max_length=retriever_cfg.get("max_length", 256),
            batch_size=retriever_cfg.get("batch_size", 16),
        )

        llm_cfg = config["llm"]
        self.llm = LLMGenerator(
            model_id=llm_cfg["model_id"],
            device=llm_cfg.get("device", "cuda"),
            max_new_tokens=llm_cfg.get("max_new_tokens", 128),
            temperature=llm_cfg.get("temperature", 0.0),
            top_p=llm_cfg.get("top_p", 0.9),
        )

        self.interp_cfg = config.get("interpretability", {})
        self.verifier = maybe_create_verifier(config)
        self.failure_analyzer = None
        if config.get("analysis", {}).get("enabled", False):
            rules = config["analysis"].get("failure_rules", [])
            self.failure_analyzer = analysis.FailureAnalyzer(rules)
        self.dataset_name = config["dataset"]["name"]
        self.augmentation_cfg = config.get("augmentation", {})

    def run(self) -> dict:
        cfg = self.config
        dataset_cfg = cfg["dataset"]

        start = time.time()
        examples, corpus, sample_to_doc_ids = data.load_dataset_subset(
            name=dataset_cfg["name"],
            split=dataset_cfg["split"],
            sample_size=dataset_cfg["sample_size"],
            seed=cfg.get("seed", 0),
            max_contexts_per_question=dataset_cfg.get("max_contexts_per_question", 10),
        )
        load_time = time.time() - start

        doc_embeddings = self.retriever.build_document_embeddings(corpus)

        results: List[RetrievalResult] = []
        top_k = cfg["evaluation"]["top_k"]
        include_matches = self.interp_cfg.get("store_token_matches", False)
        top_pairs = self.interp_cfg.get("top_token_pairs", 5)

        payload_results: List[dict] = []

        for example in examples:
            doc_ids_for_sample = sample_to_doc_ids.get(example.sample_id, [])
            candidate_docs = [doc_embeddings[doc_id] for doc_id in doc_ids_for_sample if doc_id in doc_embeddings]
            if not candidate_docs:
                # Skip examples where we could not encode any documents.
                continue
            ranked, token_matches, query_tokens = self.retriever.score(
                question=example.question,
                documents=candidate_docs,
                top_k=top_k,
                include_matches=include_matches,
                top_token_pairs=top_pairs,
            )
            ordered_docs = [doc_embeddings[doc_id] for doc_id, _ in ranked]
            context_texts = [doc.title + ": " + doc.text for doc in ordered_docs]

            prompt = self.llm.build_prompt(example.question, context_texts)
            raw_answer = self.llm.generate(prompt)
            formatted_answer, formatting_meta = apply_formatting(
                self.dataset_name, raw_answer, [doc.text for doc in ordered_docs]
            )
            final_answer = formatted_answer if formatted_answer else raw_answer.strip()

            retrieved_ids = [doc_id for doc_id, _ in ranked]
            hit_at_k = metrics.retrieval_hit_rate(retrieved_ids, example.supporting_doc_ids, top_k)
            mrr = metrics.mean_reciprocal_rank(retrieved_ids, example.supporting_doc_ids)
            em = metrics.exact_match(final_answer, example.answer)
            f1 = metrics.f1_score(final_answer, example.answer)

            contexts_for_verifier = context_texts
            verifier_score = None
            verifier_supported_docs: List[str] = []
            verifier_threshold = None
            initial_verifier_score = None
            if self.verifier:
                verifier_result = self.verifier.score_answer(
                    question=example.question,
                    answer=final_answer,
                    passages=contexts_for_verifier,
                )
                verifier_score = verifier_result.score
                initial_verifier_score = verifier_score
                verifier_supported_docs = verifier_result.supporting_passages
                verifier_threshold = self.verifier.threshold

            answer_contains_gold = example.answer.lower() in final_answer.lower()
            binary_choice = analysis.detect_binary_choice(example.question)
            answer_in_ctx = analysis.answer_in_context(final_answer, [
                {"title": doc.title, "text": doc.text} for doc in ordered_docs
            ])

            refinement_meta: Dict[str, str] = {}
            if self.augmentation_cfg.get("enabled", False):
                aug_threshold = self.augmentation_cfg.get(
                    "verifier_threshold",
                    self.verifier.threshold if self.verifier else 0.3,
                )
                needs_template = (
                    self.dataset_name == "boolq"
                    and final_answer not in {"yes", "no"}
                )
                verifier_low = (
                    self.verifier is not None
                    and initial_verifier_score is not None
                    and initial_verifier_score < aug_threshold
                )
                if needs_template or verifier_low:
                    refined_raw, refine_details = refine_answer(
                        self.dataset_name,
                        example.question,
                        [doc.text for doc in ordered_docs],
                        self.llm,
                        "template_fix" if needs_template else "verifier_low",
                    )
                    refined_formatted, refined_format_meta = apply_formatting(
                        self.dataset_name,
                        refined_raw,
                        [doc.text for doc in ordered_docs],
                    )
                    candidate_answer = (
                        refined_formatted if refined_formatted else refined_raw.strip()
                    )
                    if candidate_answer:
                        final_answer = candidate_answer
                        refinement_meta = {
                            **refine_details,
                            "formatting": refined_format_meta,
                            "initial_answer": raw_answer,
                        }
                        em = metrics.exact_match(final_answer, example.answer)
                        f1 = metrics.f1_score(final_answer, example.answer)
                        if self.verifier:
                            verifier_result = self.verifier.score_answer(
                                question=example.question,
                                answer=final_answer,
                                passages=contexts_for_verifier,
                            )
                            verifier_score = verifier_result.score
                            verifier_supported_docs = verifier_result.supporting_passages
                            verifier_threshold = self.verifier.threshold
                        answer_contains_gold = example.answer.lower() in final_answer.lower()
                        answer_in_ctx = analysis.answer_in_context(final_answer, [
                            {"title": doc.title, "text": doc.text} for doc in ordered_docs
                        ])
                        formatting_meta = refined_format_meta

            top_docs_payload = []
            for (doc_id, score), doc in zip(ranked, ordered_docs):
                payload = {
                    "doc_id": doc_id,
                    "score": score,
                    "title": doc.title,
                    "is_supporting": doc.is_supporting,
                    "text": doc.text,
                }
                if include_matches and doc_id in token_matches:
                    payload["token_matches"] = token_matches[doc_id]
                top_docs_payload.append(payload)

            result_dict = {
                "sample_id": example.sample_id,
                "question": example.question,
                "answer": example.answer,
                "initial_answer": raw_answer,
                "llm_answer": final_answer,
                "supporting_doc_ids": list(example.supporting_doc_ids),
                "top_docs": top_docs_payload,
                "em": em,
                "f1": f1,
                "hit_at_k": hit_at_k,
                "mrr": mrr,
                "verifier_score": verifier_score,
                "verifier_threshold": verifier_threshold,
                "verifier_supported_docs": verifier_supported_docs,
                "initial_verifier_score": initial_verifier_score,
                "answer_contains_gold": answer_contains_gold,
                "binary_choice": binary_choice,
                "answer_in_context": answer_in_ctx,
                "question_tokens": query_tokens,
                "formatting": formatting_meta,
                "refinement": refinement_meta,
            }

            if self.failure_analyzer:
                tags = self.failure_analyzer.analyze(result_dict)
                result_dict["failure_tags"] = [tag.__dict__ for tag in tags]

            results.append(
                RetrievalResult(
                    sample_id=example.sample_id,
                    question=example.question,
                    answer=example.answer,
                    llm_answer=final_answer,
                    supporting_doc_ids=list(example.supporting_doc_ids),
                    top_docs=top_docs_payload,
                    em=em,
                    f1=f1,
                    hit_at_k=hit_at_k,
                    mrr=mrr,
                )
            )
            payload_results.append(result_dict)

        metrics_summary = {
            "em": metrics.aggregate(result.em for result in results),
            "f1": metrics.aggregate(result.f1 for result in results),
            "hit_at_k": metrics.aggregate(result.hit_at_k for result in results),
            "mrr": metrics.aggregate(result.mrr for result in results),
        }

        payload = {
            "config": self.config,
            "load_time_sec": load_time,
            "num_examples": len(examples),
            "metrics": metrics_summary,
            "results": payload_results,
        }

        out_path = self.output_dir / "results.json"
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

        if self.interp_cfg.get("store_token_matches") and self.config.get("visualization", {}).get("enabled", False):
            viz_dir = Path(self.config["visualization"]["output_dir"])
            export_heatmaps(
                payload,
                viz_dir,
                max_examples=self.interp_cfg.get("heatmap_examples", 10),
            )

        return payload
