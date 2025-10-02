# When Retrieval Nails It but Answers Still Drift: Token-Level Diagnostics for LLM-Augmented Search

_A30 (24 GB) · Shadeform massedcompute, Wichita region · October 2, 2025_

## Why we built this
Mixedbread’s research agenda keeps circling back to two ideas:

1. **Late interaction** is still the sharpest tool we have for dense retrieval that cares about word-level evidence.
2. **Interpretability + verification** is the only way to prove that an LLM is actually using that evidence.

To stress-test both, we spun up a Shadeform A30 node (~$0.21 all-in across two short sessions) and built an end-to-end, instrumented RAG stack:

- **Retriever:** true late-interaction embeddings from `colbert-ir/colbertv2.0`, encoding 75 sampled HotpotQA distractor questions (≈750 passages). We keep per-token MaxSim traces for every query ↔ passage pair.
- **Generator:** `mistralai/Mistral-7B-Instruct-v0.3` running locally via vLLM in fp16. Prompts fuse the top-3 passages and demand grounded answers.
- **Verifier:** a cross-encoder (`cross-encoder/nli-deberta-base`) that judges whether any retrieved passage actually supports the proposed answer.
- **Artifacts:** JSON logs with token alignments + verifier scores, HTML heatmaps for the top token matches, failure taxonomy tags, and a Markdown report.

Everything (data prep → retrieval → generation → reporting) is scripted in `src/cli/run_retrieval.py`, with summaries in `results/<dataset>_late_interaction/`.

## What happened on 75 HotpotQA hops
| Metric | Mean | σ |
| --- | --- | --- |
| Hit@3 | **0.987** | 0.115 |
| MRR | **0.916** | 0.211 |
| F1 | 0.182 | 0.226 |
| Exact Match | 0.027 | 0.161 |

Retrieval is dialed in—74/75 questions surface a gold paragraph in the top 3, and the verifier finds supporting evidence for 32 answers. Yet 67/75 outputs still miss HotpotQA’s strict grading. The pain points line up with Mixedbread’s “search is the bottleneck” thesis—but on the generation side this time.

### Failure taxonomy (automatic tags)
- **Unsupported answer (35 cases, 52% of failures):** verifier confidence <0.3. These include hallucinated entities and answers that never appear in the retrieved passages.
- **Span mismatch (26 cases):** answer is semantically correct but doesn’t exactly match the gold span (`Armenian` vs. `Armenia`).
- **Wrong option (2 cases):** binary-choice questions where the LLM picks the distractor despite the correct passage being ranked first.

## Token-level forensics
Late interaction scoring turns every query token into a breadcrumb trail.

### Success snapshot – Billy Howle question
```
Dominic → dominic (0.94)
Cooke   → cooke   (0.93)
British → british (0.91)
```
Both supporting docs light up the same handful of question tokens with >0.9 cosine, and the verifier agrees (0.74). Mistral copies the correct span “Saoirse Ronan”.

### Failure snapshot – Band name from a hat
```
Midnight → midnight (0.86)
Hat      → hat      (0.85)
```
The “Midnight Oil” page is ranked #1 and nails the query tokens, but two sibling Wikipedia entries with similar lexicon sneak into the top-3. Mistral chooses “Switchfoot”; the verifier flags the answer as unsupported (0.18) and the heatmap HTML highlights the overlapping but misleading tokens.

Across the whole run the mean sum of the top-5 token similarities barely differs (13.31 vs. 12.98 for success vs. failure), underscoring why similarity alone is a weak confidence signal. The verifier + token traces give Mixedbread analysts something actionable.

## Interpretability toolkit in this drop
1. **Answer verifier overlay.** Every result in `results.json` records the verifier score, the passages it considered supportive, and the tag that triggered when the score fell below threshold (default 0.3).
2. **Failure taxonomy auto-tagger.** Rule-based labels (unsupported answer, span mismatch, wrong option) are attached to each failure—critical for Mixedbread’s “why did this fail?” debugging loop.
3. **Heatmap gallery.** Ten examples get HTML heatmaps (`results/.../viz/*.html`) that visualise the strongest query↔document token alignments for quick human review.
4. **Report enhancements.** `report.md` now includes failure statistics, verifier summaries, and curated success/failure case studies.

## Beyond HotpotQA: BoolQ & SQuAD v2

To prove the tooling generalises, we ran the identical pipeline on two contrasting QA benchmarks and kept the guardrail threshold at 0.3:

| Dataset | Questions | Hit@K | F1 | Accepted by verifier |
| --- | --- | --- | --- | --- |
| HotpotQA | 75 | 0.987 | 0.182 | 32 (43%) |
| BoolQ (yes/no) | 300 | **1.000** | 0.860 | 0 (0%) |
| SQuAD v2 (span) | 200 | **1.000** | 0.239 | 10 (5%) |

- **BoolQ:** Forcing the LLM through a yes/no template pushes F1/EM to **0.86** with zero retrieval errors. Verifier scores remain low (single-word answers are hard for NLI), so we treat BoolQ as a templating task rather than a guardrail task.
- **SQuAD v2:** Even with the gold paragraph 100% of the time, 140 failures are still tagged `unsupported_answer`. Span-normalised decoding and citation extraction lift F1 to **0.24**—nearly 2× the unformatted baseline—while surfacing the remaining paraphrase vs. support gap.

Interactive heatmaps and per-sample tags for each dataset live in `results/boolq_late_interaction/` and `results/squad_v2_late_interaction/`.

## Why this matters for Mixedbread
- **Evidence of the bottleneck.** Retrieval is already nearly perfect; the LLM still wanders. Mixedbread’s conviction that search—not model scale—is the limiter shows up plainly.
- **Ready-to-use diagnostics.** Analysts can open a case file and see: (a) which tokens drove retrieval, (b) whether the answer is supportable, and (c) why a failure label fired.
- **Low-compute reproducibility.** Everything here ran on a single Shadeform A30 (<$0.21). The same harness can scale to PopQA, ALCE, or multimodal corpora once we provision a bigger slice.

## Guardrail simulation snapshot

Treating the verifier as a gate (accept if score ≥ 0.3) gives Mixedbread a precision/recall knob:

- **HotpotQA:** 32 / 75 outputs accepted (43%) with accepted F1 ≈ 0.10. Guardrails flag hallucinations, and we can tighten templates further to raise precision.
- **BoolQ:** 0 / 300 accepted—NLI scores crash on single-word answers even though templating fixed accuracy. This highlights the need for dataset-aware guardrails or alternative verifiers.
- **SQuAD v2:** 10 / 200 accepted (5%) with accepted F1 ≈ 0.14. The verifier is conservative; paired with span-normalised decoding it catches hallucinations but still prefers longer spans for confidence.

## Where to push next
1. **Verifier-as-guardrail.** Use the NLI score to veto unsupported answers or trigger an automatic “ask for clarification” fallback.
2. **Contrastive negative mining.** Log near-miss passages (high similarity, no support) to fine-tune ColBERT’s interaction head.
3. **Scaling out.** Run the same instrumentation on PopQA (long-tail), ALCE (agentic multi-hop), and table-heavy slices; compare failure buckets across domains.
4. **Visual debugger UI.** Wrap the heatmaps, verifier scores, and answer spans in a lightweight Streamlit/Gradio app so Mixedbread can triage failures faster.

Code, configs, metrics, heatmaps, and planning docs live under `Retrieval-Interp/`. Point it at a larger Shadeform instance—or shard multiple MIG slices—and we can push beyond HotpotQA/BoolQ/SQuAD into PopQA, ALCE, or multimodal corpora with the same interpretability guarantees.
