# Evaluation results

Generated 2026-10-26T21:36:37+00:00 by `python scripts/evaluate.py`.
All data is synthetic (`data/synthetic_claims`, `data/synthetic_denials`). Numbers below are
computed by `src/claims_autopilot/evaluation.py`; re-run the script to reproduce them.

## 1. Claim extraction (field-level accuracy)

| Method | Claims | Fields | Accuracy | Values invented | Claims 100% correct |
|---|---|---|---|---|---|
| deterministic verifier only (no LLM) | 28 | 619 | 86.8% | 0 | 18 |

**LLM extraction was not run for this report.** Run `python scripts/evaluate.py --llm` with a
working OpenAI key to measure it.

Accuracy by input layout (parser baseline): superbill 100.0%, narrative 100.0%, soap 95.5%, export 100.0%, freetext 15.6%

The parser baseline is high on labeled layouts and low on free-text notes. That gap is why the
app uses the LLM for reading and keeps the parser only as a verifier.

## 2. Pre-submission check (seeded defects)

- Claims: 28; claims with at least one seeded defect: 18
- Seeded defects: 23 (17 are the kind a completeness/format rule can see; the rest are clinically wrong pairings, ID typos, or undocumented services)
- Defects caught: 17 (recall 73.9%)
- Claim error rate without precheck: 64.3%
- Claim error rate after fixing every flagged issue: 21.4%
- Relative reduction: **66.7%** (simulation; depends on the seeded defect mix)
- Clean or defect claims with unexpected ERROR/WARNING flags: 0

## 3. Denial codes and retrieval

- Denial texts: 10; codes expected: 19
- CARC/RARC extraction precision 100.0%, recall 100.0%
- Guidance retrieval recall@3: 100.0% (11 expected passages)

## Not measured

- Resubmission speed or efficiency (needs real workflow timing data).
- Quality of LLM explanations and letters (needs human review).
- Anything about real payers or real claims.
