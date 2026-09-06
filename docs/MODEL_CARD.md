# Model Card — AdalatAI Case-Type Classifier v1

| Field | Value |
|---|---|
| Task | Case narrative (Bangla/Banglish/English) → case-type flags (16 classes, multi-label) |
| Model | Multinomial Naive Bayes, one-vs-rest, word 1–2 grams, positive-evidence scoring (negatives clamped 0), frequency gate ≥3, strong-token cut ≥1.2 |
| Per-class thresholds | Calibrated on hand-written natural FIR sentences (both languages), recall-first |
| Training data | 4,600 template-generated bilingual case narratives (multi-offence, disclosed synthetic, seed 20260906) + background chatter |
| Held-out test | 1,200 naturalized narratives (noise-injected) |
| **Metrics** | **Precision 0.999 · Recall 0.994 · Micro-F1 0.997** (weakest class 0.99) |
| Retrieval | BM25 over 547 verbatim Penal Code sections (bdlaws.minlaw.gov.bd) — recall@5 1.0 through the full pipeline (classifier → mapped sections ∪ BM25) |
| Improve-loop | train → evaluate → critic (weakest classes) → remediate → retrain; traced per iteration in `case_model.json.loop_trace` |
| Scope guard | The classifier SUGGESTS only. Statute text is quoted verbatim. Punishment ranges are parsed from statute text. The human judge approves/edits/overrides and signs. AI never convicts or sentences. |

## Known limits (disclosed)
- Training narratives are template-generated (synthetic, disclosed; the competition FAQ accepts synthetic data). Real FIR language is more varied — a pilot with real (anonymized) FIRs is the roadmap.
- 10 of 557 section pages failed to extract and are listed in `failed_sec_ids`.
- Retrieval is lexical (BM25) — semantic/embedding retrieval is the upgrade path.
