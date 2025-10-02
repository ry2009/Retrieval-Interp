# Immediate Next Steps

1. **Download SSH Key** – Retrieve the managed key (`9787758c-0dcf-420a-bb33-0a08e2e7ea3b`) from the Shadeform console, save as `shadeform_a30.pem`, and restrict permissions.
2. **Connect to Instance** – `ssh -i shadeform_a30.pem shadeform@209.137.198.203`; confirm GPU availability with `nvidia-smi` and install project requirements.
3. **Sync Repository** – Push this workspace to the remote instance (scp or git) and run `scripts/bootstrap_env.sh` inside `Retrieval-Interp/`.
4. **Implement Retrievers** – Build `src/core/retrieval.py` to wrap ColBERT inference and caching on the GPU.
5. **Integrate vLLM** – Stand up local vLLM server for Meta-Llama-3-8B and connect evaluator through OpenAI-compatible REST.
6. **Run Pilot Experiment** – Execute HotpotQA sample run, capture metrics, and log interpretability artifacts into `results/`.
7. **Tear Down Instance** – After experiments, delete the Shadeform instance (`curl -X POST https://api.shadeform.ai/v1/instances/163a0707-85d9-4764-a8c9-cfc5b86f0898/delete -H "X-API-KEY: $SHADEFORM_API_KEY"`).
8. **Budget Tracking** – Record runtime and remaining wallet balance before next launch.
