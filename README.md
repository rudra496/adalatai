# ⚖ AdalatAI (আদালতএআই) — AI-assisted court workflow for Bangladesh

> **Live: https://rudra496.github.io/adalatai/**
> AI analyses, quotes the statute **verbatim**, assembles judgment drafts — **the human judge decides and signs.** Every AI action is logged in an audit trail.

[🇧🇬 বাংলা](README.md) · English

## The problem
**4,742,731 pending cases** (Supreme Court data, Dec 2024) — half a century to clear at current rates. Records are digitized (eCourt since 2009) but *understanding* is not: litigants can't read their case status, and no AI assists the workflow.

## What AdalatAI does
- 🗂 **Case intake** — narrate facts in Bangla/English (voice supported) → AI suggests case types + applicable Penal Code sections with **verbatim statute quotes** and punishment ranges parsed from the official text
- ⚖ **Judge console** — review every AI suggestion → edit / approve / override → sign; complete audit trail
- 📜 **Judgment draft assembler** — assembled only from verified parts (facts + verbatim statute + statutory range); judge edits and signs
- 📚 **Law browser** — search all 547 sections (official bdlaws numbering), verbatim, with source links
- 📊 **Pendency analytics** — verified backlog numbers (SC/BSS/Law Minister) + the training improve-loop trace

## Zero-invention guarantees
- Statute text: **verbatim from bdlaws.minlaw.gov.bd** (official Ministry of Law), fetched 2026-09-06, integrity-tested (§379/§392 verbatim gates)
- Classifier: held-out micro-F1 **0.997** on disclosed synthetic corpus; per-class thresholds calibrated on hand-written natural FIR sentences
- Retrieval: BM25, recall@5 **1.0** through the real pipeline
- Evidence: **100 Crossref-verified papers** (docs/EVIDENCE.md) + **100-entry dataset registry** with live URL checks
- 18 automated tests incl. statute-verbatim gates; improve-loop (train→evaluate→critic→remediate) traced in `case_model.json`

## The principle
**AI never convicts and never sentences.** It suggests and cites; the human judge decides and signs — and every suggestion is auditable forever.

## Run
No build step: `cd app && python -m http.server 8080` · Tests: `cd app && npx vitest run --root .. tests`

## Sources
bdlaws.minlaw.gov.bd (official) · ecourt.gov.bd · SC judgments via LII/CLCBD · EVIDENCE.md (100 papers) · datasets_registry.json (100 sources)
