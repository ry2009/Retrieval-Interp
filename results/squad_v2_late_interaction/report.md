# Experiment Report: squad_v2_late_interaction

## Aggregate Metrics
- **EM**: 0.120 ± 0.325
- **F1**: 0.239 ± 0.369
- **HIT_AT_K**: 1.000 ± 0.000
- **MRR**: 1.000 ± 0.000

## Failure Taxonomy
- **unsupported_answer**: 140 cases (94% of failures)
- **span_mismatch**: 33 cases (22% of failures)

## Case Studies
### Success: Sample 572f64ccb2c2fd14005680b7
**Question:** Which century was there a program to straighten the Rhine? 
**Gold Answer:** 19th Century
**Model Answer:** 19th Century
**Retrieval Hit@K:** 1.00 | **MRR:** 1.00
**Verifier Score:** 0.01 (threshold 0.30)
**Failure Tags:** unsupported_answer
Top Documents:
  - ✅ Rhine (score=10.11)
    - Token matches: program→program (0.89), rhine→rhine (0.87), straighten→straightening (0.86), the→the (0.78), century→century (0.78)
### Failure: Sample 5a55157f134fea001a0e18ec
**Question:** BBN provided financing for what?
**Gold Answer:** 
**Model Answer:** provided
**Retrieval Hit@K:** 1.00 | **MRR:** 1.00
**Verifier Score:** 0.02 (threshold 0.30)
**Failure Tags:** unsupported_answer
Top Documents:
  - ✅ Packet_switching (score=5.95)
    - Token matches: ##n→##n (0.82), bb→bb (0.79), financing→financing (0.77), provided→provided (0.71), ?→provided (0.62)