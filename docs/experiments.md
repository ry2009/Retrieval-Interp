# Experiment Roadmap

## Baseline 01 – HotpotQA
- Retriever: ColBERTv2, public checkpoints.
- LLM Re-ranker: Meta-Llama-3-8B-Instruct via vLLM on Shadeform A30.
- Metrics: EM, F1, answerability score (LLM self-check).
- Interpretability: token-level MaxSim heatmaps + LLM rationale highlighting evidence sentences.

## Baseline 02 – PopQA
- Retriever: Contriever-MS MARCO fine-tune.
- LLM: Mistral-Nemo-Instruct for answer synthesis.
- Focus: measuring long-tail entity coverage; correlate failure cases with retrieval depth.

## Stretch – Agentic Eval Set (ALCE)
- Compose multi-hop reasoning prompts; evaluate plan+act loop.
- Use LLM to introspect retrieval sufficiency before final answer.
