# Multi-Dataset Summary (HotpotQA, BoolQ, SQuAD v2)

## Metrics Overview

| Dataset | Questions | Hit@K | MRR | F1 | EM | Verifier Pass (>=0.3) |
| --- | --- | --- | --- | --- | --- | --- |
| HotpotQA | 75 | 0.987 | 0.916 | 0.182 | 0.027 | 32 (43%) |
| BoolQ | 300 | 1.000 | 1.000 | 0.860 | 0.860 | 0 (0%) |
| SQuAD v2 | 200 | 1.000 | 1.000 | 0.239 | 0.120 | 10 (5%) |

## Failure Taxonomy Counts

| Dataset | Unsupported | Span Mismatch | Wrong Option |
| --- | --- | --- | --- |
| HotpotQA | 35 | 26 | 2 |
| BoolQ | 42 | 42 | 0 |
| SQuAD v2 | 140 | 33 | 0 |

## Representative Cases

- **HotpotQA:** “Who will Billy Howle be seen opposite…?” → Refinement keeps span accuracy, verifier 0.68.
- **BoolQ:** “Is EZ Pass the same as EZ Tag?” → Template forces **yes**, F1 = 1.0, verifier remains low (0.01) highlighting threshold tuning needs.
- **SQuAD v2:** “European Law is applied by…?” → Span-normalised decoding quotes the exact phrase, raising F1 while verifier still conservative (0.05).

## Guardrail Snapshot (Threshold 0.3)
- Coverage vs. precision trade-off: Hotpot 43%, BoolQ 0%, SQuAD 5% at threshold 0.3.
- Accepted subsets see modest F1 lifts (≈0.10, 0, 0.14) and motivate dataset-aware guardrail settings plus structured decoding.

See `docs/guardrail_analysis.md` for detailed numbers and follow-up ideas.
