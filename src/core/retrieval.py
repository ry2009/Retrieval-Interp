"""Late interaction retriever implementation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .data import Document


@dataclass
class DocEmbedding:
    """Encoded document tensor and metadata stored on CPU."""

    doc_id: str
    embeddings: torch.Tensor  # (seq_len, hidden), float16
    mask: torch.Tensor  # (seq_len,)
    tokens: List[str]
    title: str
    text: str
    is_supporting: bool
    sample_id: str


class LateInteractionRetriever:
    """Simplified ColBERT-style late interaction retriever."""

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        max_length: int = 256,
        batch_size: int = 16,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def _encode_batch(self, texts: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokenized = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        outputs = self.model(**tokenized)
        embeddings = F.normalize(outputs.last_hidden_state, p=2, dim=-1)
        return embeddings, tokenized["attention_mask"].bool(), tokenized["input_ids"]

    @torch.inference_mode()
    def build_document_embeddings(self, corpus: Dict[str, Document]) -> Dict[str, DocEmbedding]:
        """Encode all documents and return CPU tensors for later scoring."""

        doc_ids = list(corpus.keys())
        embeddings: Dict[str, DocEmbedding] = {}

        for start in range(0, len(doc_ids), self.batch_size):
            chunk_ids = doc_ids[start : start + self.batch_size]
            texts = [corpus[doc_id].text for doc_id in chunk_ids]
            batch_embeddings, batch_masks, batch_input_ids = self._encode_batch(texts)

            batch_embeddings = batch_embeddings.half().cpu()
            batch_masks = batch_masks.cpu()
            batch_input_ids = batch_input_ids.cpu()

            for doc_id, emb, mask, token_ids in zip(
                chunk_ids, batch_embeddings, batch_masks, batch_input_ids
            ):
                tokens = self.tokenizer.convert_ids_to_tokens(token_ids.tolist())
                embeddings[doc_id] = DocEmbedding(
                    doc_id=doc_id,
                    embeddings=emb,
                    mask=mask,
                    tokens=tokens,
                    title=corpus[doc_id].title,
                    text=corpus[doc_id].text,
                    is_supporting=corpus[doc_id].is_supporting,
                    sample_id=corpus[doc_id].sample_id,
                )
        return embeddings

    @torch.inference_mode()
    def score(  # pylint: disable=too-many-locals
        self,
        question: str,
        documents: Sequence[DocEmbedding],
        top_k: int,
        include_matches: bool = False,
        top_token_pairs: int = 5,
    ) -> Tuple[List[Tuple[str, float]], Dict[str, List[dict]], List[str]]:
        """Score documents against the question using late interaction."""

        query_embeddings, query_mask, query_input_ids = self._encode_batch([question])
        query_embeddings = query_embeddings[0]
        query_mask = query_mask[0]
        query_tokens = self.tokenizer.convert_ids_to_tokens(query_input_ids[0].tolist())

        scores: List[Tuple[str, float]] = []
        token_matches: Dict[str, List[dict]] = {}

        q_emb = query_embeddings[query_mask]
        q_tokens = [tok for tok, keep in zip(query_tokens, query_mask.tolist()) if keep]

        for doc in documents:
            d_emb = doc.embeddings.to(self.device, dtype=query_embeddings.dtype)
            d_mask = doc.mask.to(self.device)

            sim = torch.matmul(q_emb, d_emb.T)  # (q_len, d_len)
            sim = sim.masked_fill(~d_mask.unsqueeze(0), float("-inf"))
            max_sim, max_idx = sim.max(dim=-1)
            score = float(max_sim.sum().item())
            scores.append((doc.doc_id, score))

            if include_matches:
                matches: List[dict] = []
                doc_tokens = doc.tokens
                for q_tok, s, idx in zip(q_tokens, max_sim.tolist(), max_idx.tolist()):
                    if q_tok in {"[CLS]", "[SEP]"}:
                        continue
                    doc_tok = doc_tokens[idx]
                    matches.append(
                        {
                            "query_token": q_tok,
                            "doc_token": doc_tok,
                            "similarity": s,
                        }
                    )
                matches.sort(key=lambda item: item["similarity"], reverse=True)
                token_matches[doc.doc_id] = matches[:top_token_pairs]

        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k], token_matches, q_tokens
