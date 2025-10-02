# Remote Environment Setup (Shadeform A30)

1. **Base packages**
   ```bash
   sudo apt update
   sudo apt install -y build-essential python3.10-venv python3-pip git jq
   ```
2. **Python env**
   ```bash
   git clone <repo_url> # TODO: populate once public
   cd Retrieval-Interp
   ./scripts/bootstrap_env.sh
   ```
3. **CUDA / Drivers**
   - `ubuntu22.04_cuda12.6_shade_os` ships with CUDA 12.6 drivers pre-installed.
   - Verify with `nvidia-smi` and `nvcc --version` (if needed install `cuda-toolkit-12-6`).
4. **Model Assets**
   ```bash
   export HF_TOKEN=...
   huggingface-cli download intfloat/colbertv2.0 --local-dir checkpoints/colbertv2
   huggingface-cli login --token $HF_TOKEN --add-to-git-credential
   ```
5. **Data**
   ```bash
   python -m src.data.prepare --dataset hotpotqa --split validation --sample 100
   ```
6. **vLLM Launch (local)**
   ```bash
   HF_TOKEN=... CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
     --host 0.0.0.0 --port 8000 --dtype float16 --max-model-len 4096 \
     --enforce-eager --tensor-parallel-size 1
   ```
7. **Run Baseline**
   ```bash
   python -m src.cli.run_retrieval --config configs/baseline.yaml \
     --port 8000 --retriever checkpoints/colbertv2 --dataset-cache data/hotpotqa
   ```

> Replace TODO placeholders once artifacts exist.
