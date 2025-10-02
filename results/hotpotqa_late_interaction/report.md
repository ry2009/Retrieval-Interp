# Experiment Report: hotpotqa_late_interaction

## Aggregate Metrics
- **EM**: 0.027 ± 0.161
- **F1**: 0.182 ± 0.226
- **HIT_AT_K**: 0.987 ± 0.115
- **MRR**: 0.916 ± 0.211

## Failure Taxonomy
- **unsupported_answer**: 35 cases (52% of failures)
- **span_mismatch**: 26 cases (39% of failures)
- **wrong_option**: 2 cases (3% of failures)

## Case Studies
### Success: Sample 5a8aa5835542996c9b8d5f4e
**Question:** Which rock band chose its name by drawing it out of a hat, Switchfoot or Midnight Oil?
**Gold Answer:** Midnight Oil
**Model Answer:** Midnight Oil.
**Retrieval Hit@K:** 1.00 | **MRR:** 1.00
**Verifier Score:** 0.03 (threshold 0.30)
**Failure Tags:** unsupported_answer, wrong_option
Top Documents:
  - ✅ Midnight Oil (score=15.30)
    - Token matches: oil→oil (0.86), midnight→midnight (0.84), band→band (0.84), hat→hat (0.82), name→name (0.82)
  - ⬜ Read About It (score=14.28)
    - Token matches: oil→oil (0.91), midnight→midnight (0.89), band→band (0.86), rock→rock (0.84), it→it (0.78)
  - ⬜ 20,000 Watt R.S.L. (score=13.94)
    - Token matches: oil→oil (0.90), midnight→midnight (0.89), band→band (0.83), rock→rock (0.81), name→name (0.76)
### Failure: Sample 5a7a567255429941d65f25bd
**Question:** What was Iqbal F. Qadir on when he participated in an attack on a radar station located on western shore of the Okhamandal Peninsula?
**Gold Answer:** flotilla
**Model Answer:** Iqbal F. Qadir participated in an attack on the Dwarka radar station, which was located on the western shore of the Okhamandal Peninsula in India.
**Retrieval Hit@K:** 1.00 | **MRR:** 1.00
**Verifier Score:** 0.04 (threshold 0.30)
**Failure Tags:** unsupported_answer, span_mismatch
Top Documents:
  - ✅ Iqbal F. Qadir (score=22.97)
    - Token matches: iqbal→iqbal (0.94), radar→radar (0.91), station→station (0.90), attack→attacked (0.84), participated→participation (0.82)
  - ⬜ Sevastopol Radar Station (score=21.04)
    - Token matches: located→located (0.89), station→station (0.86), radar→radar (0.85), attack→attack (0.83), on→in (0.81)
  - ⬜ Mukachevo Radar Station (score=20.75)
    - Token matches: located→located (0.90), radar→radar (0.86), station→station (0.86), attack→attack (0.84), on→in (0.81)