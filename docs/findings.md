# Findings – HotpotQA Late Interaction Study (Phase 2)

 - **HotpotQA (75 q):** Hit@3 = 0.987, F1 = 0.182, EM = 0.027. Verifier passes 32 answers (43%); failure tags now `unsupported_answer` 35, `span_mismatch` 26, `wrong_option` 2.
 - **BoolQ (300 q):** Hit@1 = 1.000, F1/EM = 0.860 thanks to templating; verifier rejects all answers (NLI struggles with single-word responses).
 - **SQuAD v2 (200 q):** Hit@3 = 1.000, F1 = 0.239, EM = 0.120. Verifier passes 10 answers (5%); remaining failures split across `unsupported_answer` 140 and `span_mismatch` 33.
 - **Token-similarity gap:** Across datasets the mean sum of top-5 query→doc similarities differs by <0.4 between successes and failures, reinforcing that similarity alone isn’t a reliable confidence signal.
 - **Guardrail simulation (threshold 0.3):** Accepted subsets achieve F1 of ~0.10 (HotpotQA), 0.00 (BoolQ), 0.14 (SQuAD v2) with coverage of 43%, 0%, and 5% respectively—guardrails catch egregious errors but require dataset-aware thresholds.
 - **Artifacts:** Heatmaps and per-sample tags live in `results/<dataset>_late_interaction/viz/`; Markdown reports include curated case studies and failure statistics.
 - **Cost footprint:** Combined Shadeform spend ≈$0.42 across the updated runs (A30, <25 minutes per sweep).
 - **Next targets:** PopQA (long-tail) and an agentic dataset (ALCE or MuSiQue) scheduled next; guardrail-driven re-asking and template-constrained decoding planned for BoolQ-style tasks.

Refer to `results/hotpotqa_late_interaction/results.json` for per-question traces and `report.md` for a narrative summary.
