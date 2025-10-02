# Interpretability Strategy

## Signals to Capture
- Token-level relevance scores from late interaction (MaxSim) heads.
- LLM-generated rationales explaining answer choice and citing retrieved passages.
- Cross-attention saliency from the generation model using gradient × input.

## Tooling Outline
1. **Retriever tracing** – modify ColBERT scoring to emit per-token similarity maps.
2. **LLM rationale generation** – prompt model for self-evaluation of retrieved context sufficiency.
3. **Visualization** – build lightweight HTML report with aggregated metrics and per-example heatmaps.

## Metrics
- Evidence coverage (fraction of gold sentences retrieved).
- Rationale faithfulness (LLM rationale tokens overlapping top retrieval tokens).
- Failure categorization (retrieval miss vs synthesis error).

## Integration Plan
- Implement hooks in `src/core/retrieval.py` to capture embeddings and Top-k docs.
- Use `rich` tables for CLI previews; Jinja2 templates for HTML summary.
- Store per-run artifacts in `results/<timestamp>/` for reproducibility.
