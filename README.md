# LLM-Augmented Retrieval & Interpretability Workspace

This workspace targets the Mixedbread research direction on LLM-augmented search with a focus on interpretability. It houses experiments, scripts, and documentation for coupling late-interaction retrievers with language-model reasoning layers and analyzing why retrieval succeeds or fails.

## Layout
- `src/` – Python modules for data loaders, retriever interfaces, LLM orchestration, and interpretability tooling.
- `configs/` – YAML configuration files describing experiment setups (datasets, retrievers, LLMs, metrics).
- `notebooks/` – Exploratory analyses and visualization notebooks.
- `docs/` – Narrative reports, blog outlines, and interpretability case studies.

## Getting Started
1. Provision a Shadeform GPU instance (see `docs/provisioning.md`).
2. Install dependencies via `scripts/bootstrap_env.sh` (to be added).
3. Run baseline retrieval with `python -m src.cli.run_retrieval --config configs/baseline.yaml`.
4. Launch interpretability report generation with `python -m src.cli.render_report --config configs/baseline.yaml`.

Detailed instructions will evolve alongside implementation.
