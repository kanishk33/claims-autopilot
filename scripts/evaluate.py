"""Run the evaluation on the labeled synthetic set and write a report.

    python scripts/evaluate.py                 # rules + denials + parser baseline (no API calls)
    python scripts/evaluate.py --llm           # ALSO run live OpenAI extraction on all claims

Outputs:
    outputs/evaluation_results.json   full detail (git-ignored)
    docs/EVALUATION.md                human-readable summary (commit this)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from claims_autopilot.config import OUTPUTS_DIR, PROJECT_ROOT, get_settings, setup_logging
from claims_autopilot.evaluation import (
    evaluate_denials,
    evaluate_extraction,
    evaluate_precheck,
    llm_extract,
    parser_only_extract,
)


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="run live OpenAI extraction (costs a few cents)")
    args = ap.parse_args()
    setup_logging()

    results: dict = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    results["parser_baseline"] = evaluate_extraction(parser_only_extract, "deterministic verifier only (no LLM)")
    if args.llm:
        results["llm_extraction"] = evaluate_extraction(llm_extract(), f"OpenAI {get_settings().openai_model} + verifier")
    results["precheck"] = evaluate_precheck()
    results["denials"] = evaluate_denials()

    OUTPUTS_DIR.mkdir(exist_ok=True)
    (OUTPUTS_DIR / "evaluation_results.json").write_text(json.dumps(results, indent=2, default=str))

    pb, pc, dn = results["parser_baseline"], results["precheck"], results["denials"]
    lines = [
        "# Evaluation results",
        "",
        f"Generated {results['generated_at']} by `python scripts/evaluate.py{' --llm' if args.llm else ''}`.",
        "All data is synthetic (`data/synthetic_claims`, `data/synthetic_denials`). Numbers below are",
        "computed by `src/claims_autopilot/evaluation.py`; re-run the script to reproduce them.",
        "",
        "## 1. Claim extraction (field-level accuracy)",
        "",
        "| Method | Claims | Fields | Accuracy | Values invented | Claims 100% correct |",
        "|---|---|---|---|---|---|",
    ]
    rows = [pb] + ([results["llm_extraction"]] if "llm_extraction" in results else [])
    for r in rows:
        lines.append(f"| {r['method']} | {r['claims_evaluated']} | {r['fields_evaluated']} | {pct(r['field_accuracy'])} | "
                     f"{r['hallucinated_values']} | {r['claims_100pct_correct']} |")
    if "llm_extraction" not in results:
        lines += ["", "**LLM extraction was not run for this report.** Run `python scripts/evaluate.py --llm` with a",
                  "working OpenAI key to measure it."]
    lines += ["", "Accuracy by input layout (parser baseline): " +
              ", ".join(f"{k} {pct(v)}" for k, v in pb["accuracy_by_template"].items()), "",
              "The parser baseline is high on labeled layouts and low on free-text notes. That gap is why the",
              "app uses the LLM for reading and keeps the parser only as a verifier.", "",
              "## 2. Pre-submission check (seeded defects)", "",
              f"- Claims: {pc['claims']}; claims with at least one seeded defect: {pc['claims_with_seeded_defects']}",
              f"- Seeded defects: {pc['seeded_defects']} ({pc['defects_detectable_by_rule_design']} are the kind a completeness/format rule can see;"
              " the rest are clinically wrong pairings, ID typos, or undocumented services)",
              f"- Defects caught: {pc['defects_caught']} (recall {pct(pc['defect_recall'])})",
              f"- Claim error rate without precheck: {pct(pc['claim_error_rate_without_precheck'])}",
              f"- Claim error rate after fixing every flagged issue: {pct(pc['claim_error_rate_after_fixing_flags'])}",
              f"- Relative reduction: **{pct(pc['relative_error_rate_reduction'])}** (simulation; depends on the seeded defect mix)",
              f"- Clean or defect claims with unexpected ERROR/WARNING flags: {len(pc['claims_with_unexpected_flags'])}",
              "", "## 3. Denial codes and retrieval", "",
              f"- Denial texts: {dn['denials']}; codes expected: {dn['codes_expected']}",
              f"- CARC/RARC extraction precision {pct(dn['code_precision'])}, recall {pct(dn['code_recall'])}",
              f"- Guidance retrieval recall@{dn['k']}: {pct(dn['guidance_retrieval_recall_at_k'])} "
              f"({dn['guidance_passages_expected']} expected passages)",
              "", "## Not measured", "",
              "- Resubmission speed or efficiency (needs real workflow timing data).",
              "- Quality of LLM explanations and letters (needs human review).",
              "- Anything about real payers or real claims."]
    report = PROJECT_ROOT / "docs" / "EVALUATION.md"
    report.parent.mkdir(exist_ok=True)
    report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
