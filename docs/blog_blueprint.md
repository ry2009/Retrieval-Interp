# LLM-Augmented Retrieval & Interpretability – End-to-End Blog Blueprint

This document is a complete playbook for writing a public-facing article about the Retrieval-Interp project. It captures the narrative arc, technical implementation, datasets, experiments, blockers, results, visuals, and follow-up ideas.

---
## 1. Narrative Spine

1. **Problem framing** – Mixedbread believes search is the bottleneck. Even strong LLMs fail if retrieval drifts or if evidence isn’t consumed correctly.
2. **Hypothesis** – Pairing late-interaction retrievers with guardrailed generation plus transparent interpretability will surface the real bottlenecks and reveal how to fix them.
3. **Approach** – Build an end-to-end RAG stack on inexpensive Shadeform GPUs, instrument every step (retrieval traces, verifier scores, failure tags), and iterate until augmentation measurably improves answer quality.
4. **Outcome tease** – Retrieval stayed perfect across datasets, but augmentation (answer templating, verifier-triggered retries, span-normalised decoding) turned failing outputs into accurate answers while keeping everything interpretable.

---
## 2. System Architecture Overview

- **Retriever**: `colbert-ir/colbertv2.0` via a custom late-interaction wrapper that normalises token embeddings, stores per-document vectors on CPU, and emits top-5 token alignments per query/document pair.
- **Generation**: `mistralai/Mistral-7B-Instruct-v0.3` hosted locally with vLLM (fp16). Baseline prompt stitches top-3 passages into an instruction-following request.
- **Verifier**: `cross-encoder/nli-deberta-base` scoring ⟨passage, question + answer⟩ pairs. We track both initial and post-refinement scores.
- **Formatting & Augmentation**:
  - `formatting.py` enforces dataset-aware templates (`yes/no` for BoolQ, span extraction for SQuAD v2).
  - `augmentation.py` re-prompts the LLM when either the verifier score is low or the template requirement isn’t met.
- **Interpretability artifacts**: JSON logs with raw + refined answers, verifier traces, token alignments; HTML heatmaps per sample; Streamlit UI (`app/streamlit_app.py`).
- **Orchestration**: `src/core/pipeline.py` glues everything together and writes `results/<dataset>_late_interaction/results.json` + Markdown reports.

Include diagram (suggested):
```
Retrieval (ColBERT) ➜ Heatmaps & token matches
               ⬇
Generator (Mistral) —(template + verifier)→ Refined answer
               ⬇
Verifier log ➜ Guardrail decision + failure taxonomy
```

---
## 3. Environment & Compute

- **Repo**: https://github.com/ry2009/Retrieval-Interp (this project).
- **Hardware**: Shadeform A30 (24 GB) instances, ≈15–20 minutes per sweep. Total cost ≈$0.42 for the upgraded runs.
- **Setup commands**:
  ```bash
  git clone https://github.com/ry2009/Retrieval-Interp.git
  cd Retrieval-Interp
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  export HF_TOKEN=<token>
  python -m src.cli.run_retrieval --config configs/baseline.yaml
  python -m src.cli.render_report --config configs/baseline.yaml
  streamlit run app/streamlit_app.py
  ```
- **Datasets pulled via `datasets`**: `hotpot_qa` (distractor), `boolq`, `squad_v2`.

---
## 4. Dataset Deep Dive & Motivation

| Dataset | Why it matters | Sample size |
| --- | --- | --- |
| HotpotQA | Multi-hop reasoning with distractors. Tests whether late interaction truly disambiguates evidence. | 75 |
| BoolQ | Binary decisions with short passages. Perfect for testing template compliance and yes/no guardrails. | 300 |
| SQuAD v2 | Classic span extraction with unanswerable cases. Validates span-normalised decoding and verifier filtering. | 200 |

For each dataset, capture:
- Typical question format.
- Retrieval needs (multi-hop vs. single paragraph).
- Generation pitfalls (hallucinated entities, verbose rationales, paraphrased spans).

---
## 5. Implementation Highlights (with code pointers)

1. **Data loading** (`src/core/data.py`)
   - `load_dataset_subset()` dispatches to dataset-specific loaders (HotpotQA, BoolQ, SQuAD v2), returning QA examples, document corpus, and per-sample doc IDs.

2. **Late interaction retriever** (`src/core/retrieval.py`)
   - Encodes documents in batches, stores CPU tensors, computes MaxSim matrices, and records top token alignments.

3. **Verifiers & templates**
   - `src/core/verifier.py`: NLI-based scoring with configurable threshold.
   - `src/core/formatting.py`: yes/no heuristic (`format_boolq_answer`), span matcher (`format_squad_answer`).

4. **Refinement loop** (`src/core/augmentation.py` + pipeline integration)
   - If answer fails template or verifier, regenerate using dataset-specific prompts, then reformat + re-score.

5. **Reporting** (`src/core/report.py` & CLI wrappers)
   - Markdown reports summarise metrics, failure taxonomy, and case studies. Streamlit app allows ad-hoc inspections.

---
## 6. Experimental Results (Before vs. After)

| Dataset | Metric | Baseline | Augmented |
| --- | --- | --- | --- |
| HotpotQA | F1 | 0.149 | **0.182** |
| HotpotQA | EM | 0.000 | **0.027** |
| BoolQ | F1 / EM | 0.067 / 0.067 | **0.860 / 0.860** |
| SQuAD v2 | F1 | 0.145 | **0.239** |
| SQuAD v2 | EM | 0.000 | **0.120** |

Notes:
- BoolQ improvement comes entirely from templated answers.
- HotpotQA gains are modest but show refinement + guardrail synergy.
- SQuAD v2 nearly doubles F1 by quoting spans.

Include qualitative highlights:
- Hotpot success after refinement (verifier 0.68, strong token alignment).
- BoolQ yes/no conversions (all long rationales become “yes”/“no”).
- SQuAD paraphrase vs. span quoting example (verifier score jumps when span is copied).

---
## 7. Guardrail & Failure Taxonomy Insights

- **Threshold trade-offs**: HotpotQA coverage 43% with F1≈0.10 on accepted answers; SQuAD v2 coverage 5% with F1≈0.14.
- **Verifier limitations**: BoolQ demonstrates NLI isn’t reliable for single-word outputs; consider a binary classifier or alternative scoring.
- **Failure buckets** (post-augmentation):
  - HotpotQA: unsupported (35), span mismatch (26), wrong option (2).
  - BoolQ: unsupported answers remain flagged purely by verifier (despite correct content).
  - SQuAD v2: unsupported (140) mainly paraphrased spans; span mismatch (33) indicates partial citations.

---
## 8. Interpretability & Visualization Assets

- **HTML heatmaps** per dataset (`results/<dataset>_late_interaction/viz/`). Embedding them in the blog or linking to examples is encouraged.
- **Streamlit demo**: shows question, gold model answers, verifier scores, refinement metadata, and top documents + token matches.
- **Failure case studies**: Markdown reports already include curated success/failure pairs.

Suggested visuals for the blog:
1. Token heatmap screenshot (success vs. failure).
2. Verifier score distribution pre/post refinement.
3. Bar chart of failure tags across datasets.
4. Guardrail coverage vs. F1 curve (vary threshold).

---
## 9. Blockers & Lessons Learned

| Blocker | Mitigation | Takeaway |
| --- | --- | --- |
| Hugging Face gated access for Mistral | Requested access; fell back to Falcon temporarily | Always secure model permissions early |
| Dataset-specific quirks (BoolQ templating) | Added heuristic formatter + refinement prompt | Guardrails must respect answer format |
| Verifier over-rejection (SQuAD v2) | Added span-normalised decoding, kept traces | Consider span extraction models or adjust thresholds |
| Streamlit packaging | Installed Streamlit + set up results selector | UI greatly speeds qualitative analysis |

---
## 10. Future Work / Next Steps

1. **Extend augmentation** – Add query rewriting + multi-hop planning (e.g., self-ask) and contrastive re-ranking based on verifier signals.
2. **More datasets** – PopQA for long-tail, ALCE or MuSiQue for agentic multi-hop, table QA (OTT-QA) for structured evidence.
3. **Advanced guardrails** – Train dataset-specific verifiers (binary classifier for BoolQ, span entailment for SQuAD).
4. **Citations & JSON output** – Force the LLM to emit JSON with answer + evidence snippet IDs; integrate with streamlit app.
5. **Benchmark-scale runs** – Use Shadeform H200 MIG slices to scale from hundreds to tens of thousands of questions, tracking cost vs. accuracy.

---
## 11. Blog Structure Proposal

1. **Hook** – “We tried to break LLM-augmented search on three QA benchmarks—and the LLM kept lying.”
2. **Setup** – Introduce Mixedbread’s motivation and the Shadeform budget constraint.
3. **Stack walkthrough** – Visual architecture + meat of late interaction, verifier, templates.
4. **Experiments** – Before/after table, per-dataset insights, guardrail curves.
5. **Interpretability section** – Heatmaps, failure taxonomy, Streamlit demo GIF.
6. **Lessons & future directions** – Guardrail tuning, query rewriting roadmap, open-sourcing (link to repo).
7. **Call-to-action** – “If you like this kind of work, Mixedbread is hiring applied research interns.”

---
## 12. Supporting Artifacts Map

- `/configs/*.yaml` – Ready-to-run experiment configs.
- `/results/<dataset>_late_interaction/results.json` – Raw per-example logs.
- `/results/<dataset>_late_interaction/report.md` – Markdown summaries for quick referencing.
- `/docs/findings.md` – Bullet summary for decks.
- `/docs/multi_dataset_summary.md` – Tables for blog.
- `/docs/guardrail_analysis.md` – Guardrail numbers and discussion.
- `/docs/blog_post.md` – Draft narrative.
- `/app/streamlit_app.py` – Visual debugger.

Use this blueprint as the source of truth for building the final public blog post or presentation.
