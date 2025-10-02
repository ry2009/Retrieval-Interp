# Phase 2 Roadmap – LLM-Augmented Retrieval & Interpretability

_Last updated: October 2, 2025_

## Objectives
1. **Generator Upgrade:** Replace Falcon-7B with `mistralai/Mistral-7B-Instruct-v0.3` (access confirmed) running via vLLM to unlock higher throughput and consistent decoding controls.
2. **Retriever Depth:** Move from the synthetic late-interaction wrapper to ColBERTv2’s full ANN index (FAISS or ScaNN) and evaluate both vanilla and re-ranked outputs.
3. **Interpretability Expansion:** Produce HTML heatmaps, automated failure tagging, and an answer-verifier that flags unsupported spans.
4. **Cross-Benchmark Coverage:** Augment HotpotQA with PopQA (long-tail), ALCE (agentic multi-hop), and at least one multimodal or table-heavy dataset (e.g., OTT-QA subsets).
5. **Reporting Excellence:** Deliver a polished blog + slide deck with quantitative charts, qualitative case studies, and actionable recommendations for Mixedbread.

## Milestones & Tasks

### M0 – Environment Hardening (0.5 day)
- Spin a Shadeform H200 MIG 1g/2g slice or A100 80GB; install CUDA 12.4/vLLM 0.4+.
- Containerize the workflow (Dockerfile + `docker compose` for retriever + vLLM services).
- Cache ColBERT checkpoints, dataset shards, and tokenizer vocabularies on attached NVMe.

### M1 – Retriever Upgrade (1 day)
- Build ColBERT indexing pipeline using `colbert.indexer` for HotpotQA/PopQA subsets (≥5k queries, ≥50k passages).
- Implement residual compression (4-bit) for storage efficiency and measure recall vs. uncompressed.
- Add contrastive negative mining: re-score “false positives” where MaxSim is high but gold absent; log for later fine-tune.

### M2 – Generator & Verification (1 day)
- Launch Mistral-7B via vLLM with `--tensor-parallel-size 1` (fp16) and integrate in pipeline.
- Implement verifier head:
  * Option A: smaller NLI model (e.g., `microsoft/deberta-v3-large`) to judge entailment between evidence spans and answer.
  * Option B: self-check prompt where Mistral validates whether answer tokens appear in retrieved text.
- Instrument prompts to emit structured JSON (`answer`, `citations`, `confidence`).

### M3 – Interpretability Artifacts (1.5 days)
- Convert token-match logs into attention-style heatmaps using Plotly or Altair; export HTML per case.
- Build failure taxonomy auto-tagger: span mismatch, wrong option, unsupported hallucination, incomplete reasoning.
- Surface aggregate statistics (e.g., percentage of failures per bucket, average similarity differences) in a notebook.

### M4 – Multi-dataset Experiments (1 day)
- HotpotQA: 1k-question run; PopQA: 500 random samples (long-tail). ALCE: 250 agentic tasks.
- Record runtime, GPU utilization, and cost per dataset.
- Compare baseline vs. verifier-enforced answers.

### M5 – Deliverables (1 day)
- Blog post v2 with charts, heatmaps, and lessons for Mixedbread (focus on where retrieval excels vs. where LLM fails).
- Slide deck (10 slides) summarizing methodology, metrics, failure categories, and roadmap.
- Publish reproducibility pack (scripts, configs, Dockerfile, `README_phase2.md`).

## Indicative Timeline
- **Day 0:** Provision GPU, containerize environment, cache checkpoints.
- **Day 1:** Build ColBERT indexes; calibrate contrastive negatives.
- **Day 2:** Integrate Mistral + verifier guardrail; automate JSON tracing.
- **Day 3:** Generate heatmaps, failure dashboard, and PopQA benchmark run.
- **Day 4:** Expand to ALCE/table QA, compile comparative analytics.
- **Day 5:** Finalize blog + deck, polish reproducibility assets, handoff.

## Resource Estimates
- **Compute:** ~12 GPU-hours on Mistral (includes indexing + inference); ~3 GPU-hours for ColBERT indexing.
- **Storage:** ≈150 GB (indexes + dataset caches + heatmap artifacts).
- **Budget:** ~$120 on Shadeform if using H200 MIG slices; monitor wallet and schedule off-hours spots.

## Open Questions
- Should we fine-tune ColBERT on PopQA/ALCE, or stick to zero-shot? (Impact on timeline: +1.5 days)
- Do we need multimodal support now, or is text-only sufficient for phase 2?
- How will Mixedbread prefer visualizations embedded (inline blog vs. separate dashboard)?

## Next Actions
1. Approve compute budget & instance type (recommend H200 3g.71GB or A100 80GB).
2. Greenlight verifier approach (NLI vs. self-check) and visualization stack choice.
3. Once approved, spin up new instance and begin at Milestone M0.
