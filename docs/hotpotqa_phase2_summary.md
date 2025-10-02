# HotpotQA Phase 2 Summary (Mistral + Verifier)

- **Generator:** `mistralai/Mistral-7B-Instruct-v0.3` on Shadeform A30 (vLLM fp16).
- **Retriever:** `colbert-ir/colbertv2.0` MaxSim with token-level traces (Top-3 passages per question).
- **Dataset:** 75 HotpotQA distractor questions, 10 contexts/question (≈750 passages).

## Metrics
| Metric | Mean | Std |
| ------ | ---- | ---- |
| Hit@3 | 0.987 | 0.115 |
| MRR | 0.916 | 0.211 |
| F1 | 0.182 | 0.226 |
| Exact Match | 0.027 | 0.161 |

## Verifier Outcomes (threshold = 0.3)
- Supported answers: **32 / 75** (43%).
- Unsupported answers (tagged): **35 / 75**.
- Average verifier score (supported): **0.51**; (unsupported): **0.18**.

## Failure Tags (out of 73 failures)
- `unsupported_answer`: 35 (52%).
- `span_mismatch`: 26 (39%).
- `wrong_option`: 2 (3%).

## Notable Cases
- **Success:** *“George Gershwin is an American composer and Judith Weir is a composer from which country?”* → Mistral answers “Judith Weir is a British composer.” Verifier score 0.67, heatmap shows dominant `British` alignment.
- **Unsupported:** *“Which gaming console was both Yakuza Kiwami and Yakuza 0 released on?”* → Mistral replies “PlayStation 4,” but retrieved passages discuss PS3/PS4 split; verifier score 0.29 flags lack of explicit support.
- **Wrong option:** *“Which rock band chose its name by drawing it out of a hat, Switchfoot or Midnight Oil?”* → Retrieval surfaces “Midnight Oil” first, yet Mistral copies the distractor; verifier score 0.14 and `wrong_option` tag.

## Artifacts
- `results/hotpotqa_late_interaction/results.json` – per-sample metrics, verifier outputs, tags.
- `results/hotpotqa_late_interaction/report.md` – narrative summary with tagged case studies.
- `results/hotpotqa_late_interaction/viz/` – 10 HTML token-similarity heatmaps.

## Next Dataset Targets
1. **PopQA (long-tail factual).** Larger entity spread to stress-test verifier calibration.
2. **ALCE (agentic multi-hop).** Observe failure distribution when answers require iterative planning.
3. **Table QA slice (e.g., OTT-QA).** Validate whether MaxSim + verifier handles structured evidence.
