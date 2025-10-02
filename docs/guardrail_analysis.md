# Verifier Guardrail Analysis (Threshold 0.3)

| Dataset | Questions | Accepted (>=0.3) | Coverage | Accepted F1 | Notes |
| --- | --- | --- | --- | --- | --- |
| HotpotQA | 75 | 32 | 43% | 0.10 | Guardrail catches hallucinations but still prefers concise spans. |
| BoolQ | 300 | 0 | 0% | 0.00 | NLI struggles with single-word answers despite high accuracy; adjust verifier or skip guardrail. |
| SQuAD v2 | 200 | 10 | 5% | 0.14 | Span-normalised decoding helps, but verifier remains conservative. |

Key observations:
- **Precision knob:** Higher thresholds still reduce hallucinations, but Hotpot/SQuAD show diminishing precision gains without template-aware re-generation.
- **Template gap:** BoolQ confirms that verification must be dataset-aware (either lower thresholds or alternate verifiers for binary answers).
- **Future work:** Pair verifier signals with structured decoding (JSON answers + citations) and automatic re-query when coverage is too low.
