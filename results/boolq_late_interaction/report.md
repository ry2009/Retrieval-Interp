# Experiment Report: boolq_late_interaction

## Aggregate Metrics
- **EM**: 0.860 ± 0.347
- **F1**: 0.860 ± 0.347
- **HIT_AT_K**: 1.000 ± 0.000
- **MRR**: 1.000 ± 0.000

## Failure Taxonomy
- **unsupported_answer**: 42 cases (100% of failures)
- **span_mismatch**: 42 cases (100% of failures)

## Case Studies
### Success: Sample boolq-2406
**Question:** is it illegal to buy organs in the us
**Gold Answer:** yes
**Model Answer:** yes
**Retrieval Hit@K:** 1.00 | **MRR:** 1.00
**Verifier Score:** 0.01 (threshold 0.30)
**Failure Tags:** unsupported_answer
Top Documents:
  - ✅ passage (score=8.48)
    - Token matches: organs→organs (0.92), illegal→illegal (0.84), us→states (0.81), it→the (0.75), is→illegal (0.75)
### Failure: Sample boolq-2071
**Question:** did tom hardy won an oscar for the revenant
**Gold Answer:** no
**Model Answer:** yes
**Retrieval Hit@K:** 1.00 | **MRR:** 1.00
**Verifier Score:** 0.01 (threshold 0.30)
**Failure Tags:** unsupported_answer, span_mismatch
Top Documents:
  - ✅ passage (score=10.94)
    - Token matches: ##ena→##ena (0.94), rev→rev (0.93), hardy→hardy (0.93), tom→tom (0.93), ##nt→##nt (0.92)