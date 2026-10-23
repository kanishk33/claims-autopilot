"""Measure the pipeline on the labeled synthetic set.

Three evaluations:
1. Extraction accuracy: field-by-field comparison with ground truth.
   mode="llm"    -> real OpenAI extraction + verifier (needs API key)
   mode="parser" -> deterministic verifier only (no LLM). This is a BASELINE,
                    not the app's extraction method.
2. Pre-submission check: how many seeded defects the rules catch, and the
   claim error rate before vs. after fixing the flagged issues (simulation).
3. Denials: CARC/RARC code extraction and retrieval recall.

Every number is computed from data/synthetic_*; nothing is hardcoded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable

from .config import SYNTHETIC_CLAIMS_DIR, SYNTHETIC_DENIALS_DIR
from .denial import extract_codes
from .extractor import get_path, verify_against_source
from .retrieval import _index
from .schemas import ClaimData
from .source_parser import normalize_date, parse_explicit, value_in_text
from .validator import validate_claim

EVAL_AS_OF = date(2026, 10, 23)  # fixed so results are reproducible

SCALAR_EVAL_FIELDS = {
    "patient.full_name": "name", "patient.date_of_birth": "date", "patient.sex": "exact",
    "insurance.payer_name": "name", "insurance.member_id": "id",
    "billing_provider.name": "name", "billing_provider.npi": "id",
    "rendering_provider.name": "name", "rendering_provider.npi": "id",
    "ordering_provider.name": "name", "ordering_provider.npi": "id",
    "service_date": "date", "place_of_service": "exact", "prior_authorization_number": "id",
}


def load_claim_truth() -> list[dict[str, Any]]:
    return json.loads((SYNTHETIC_CLAIMS_DIR / "ground_truth.json").read_text())


def load_denial_truth() -> list[dict[str, Any]]:
    return json.loads((SYNTHETIC_DENIALS_DIR / "ground_truth.json").read_text())


def _eq(expected: Any, predicted: Any, kind: str) -> bool:
    if expected in (None, "", []) or predicted in (None, "", []):
        return expected in (None, "", []) and predicted in (None, "", [])
    e, p = str(expected), str(predicted)
    if kind == "date":
        return normalize_date(e) == normalize_date(p)
    if kind == "name":
        # tolerate titles/suffixes: "Dr. Alex Example" == "Alex Example, MD"
        return value_in_text(p, e, "name") or value_in_text(e, p, "name")
    if kind == "id":
        return "".join(ch for ch in e.upper() if ch.isalnum()) == "".join(ch for ch in p.upper() if ch.isalnum())
    return e.strip().upper() == p.strip().upper()


@dataclass
class FieldResult:
    claim_id: str
    field: str
    expected: Any
    predicted: Any
    correct: bool
    hallucinated: bool = False  # predicted a value where none exists


def compare_claim(claim_id: str, pred: ClaimData, truth: dict[str, Any]) -> list[FieldResult]:
    t = ClaimData.model_validate(truth)
    out: list[FieldResult] = []
    for path, kind in SCALAR_EVAL_FIELDS.items():
        e, p = get_path(t, path), get_path(pred, path)
        out.append(FieldResult(claim_id, path, e, p, _eq(e, p, kind), hallucinated=e is None and p is not None))

    pred_dx = {d.pointer: d.code.replace(".", "") for d in pred.diagnoses}
    for d in t.diagnoses:
        got = pred_dx.get(d.pointer)
        out.append(FieldResult(claim_id, f"diagnosis[{d.pointer}]", d.code, got, got == d.code.replace(".", "")))
    true_dx = {d.code.replace(".", "") for d in t.diagnoses}
    for d in pred.diagnoses:
        if d.code.replace(".", "") not in true_dx:
            out.append(FieldResult(claim_id, "diagnosis[extra]", None, d.code, False, hallucinated=True))

    remaining = list(pred.service_lines)
    for line in t.service_lines:
        match = next((l for l in remaining if l.procedure_code == line.procedure_code), None)
        tag = f"line[{line.line_number}:{line.procedure_code}]"
        if match:
            remaining.remove(match)
        out.append(FieldResult(claim_id, f"{tag}.code", line.procedure_code, match and match.procedure_code, match is not None))
        out.append(FieldResult(claim_id, f"{tag}.units", line.units, match and match.units, bool(match) and match.units == line.units))
        out.append(FieldResult(claim_id, f"{tag}.modifiers", line.modifiers, match and match.modifiers,
                               bool(match) and sorted(match.modifiers) == sorted(line.modifiers)))
        out.append(FieldResult(claim_id, f"{tag}.diagnosis_pointers", line.diagnosis_pointers,
                               match and match.diagnosis_pointers,
                               bool(match) and match.diagnosis_pointers == line.diagnosis_pointers))
    for extra in remaining:
        out.append(FieldResult(claim_id, "line[extra]", None, extra.procedure_code, False, hallucinated=True))
    return out


def evaluate_extraction(extract: Callable[[str], ClaimData], label: str) -> dict[str, Any]:
    results: list[FieldResult] = []
    per_claim = []
    failures = []
    for row in load_claim_truth():
        text = (SYNTHETIC_CLAIMS_DIR / row["file"]).read_text()
        try:
            pred = extract(text)
        except Exception as exc:  # report, never hide
            failures.append({"claim_id": row["claim_id"], "error": str(exc)})
            continue
        r = compare_claim(row["claim_id"], pred, row["expected_claim"])
        results.extend(r)
        per_claim.append({"claim_id": row["claim_id"], "template": row["template"],
                          "fields": len(r), "correct": sum(x.correct for x in r),
                          "accuracy": round(sum(x.correct for x in r) / len(r), 4)})
    by_template: dict[str, list[int]] = {}
    for c in per_claim:
        agg = by_template.setdefault(c["template"], [0, 0])
        agg[0] += c["correct"]
        agg[1] += c["fields"]
    total = len(results)
    correct = sum(r.correct for r in results)
    return {
        "method": label,
        "claims_evaluated": len(per_claim),
        "claims_failed": failures,
        "fields_evaluated": total,
        "fields_correct": correct,
        "field_accuracy": round(correct / total, 4) if total else None,
        "hallucinated_values": sum(r.hallucinated for r in results),
        "claims_100pct_correct": sum(1 for c in per_claim if c["correct"] == c["fields"]),
        "accuracy_by_template": {k: round(v[0] / v[1], 4) for k, v in by_template.items()},
        "per_claim": per_claim,
        "errors": [r.__dict__ for r in results if not r.correct],
    }


def parser_only_extract(text: str) -> ClaimData:
    """BASELINE: deterministic verifier with an empty LLM result."""
    return verify_against_source(ClaimData(), text)


def llm_extract(client: Any | None = None) -> Callable[[str], ClaimData]:
    from .extractor import extract_claim

    def run(text: str) -> ClaimData:
        return extract_claim(text, client=client).packet

    return run


def evaluate_precheck(claim_source: str = "ground_truth") -> dict[str, Any]:
    """claim_source="ground_truth": validate the expected claim (isolates the
    rules from extraction quality)."""
    rows = load_claim_truth()
    n = len(rows)
    claims_with_defect = 0
    claims_with_residual = 0
    seeded = caught = detectable = 0
    false_positive_claims = []
    detail = []
    for row in rows:
        claim = ClaimData.model_validate(row["expected_claim"])
        report = validate_claim(claim, as_of=EVAL_AS_OF)
        fired = {i.rule_id for i in report.issues}
        defects = row["seeded_defects"]
        if defects:
            claims_with_defect += 1
        residual = False
        for d in defects:
            seeded += 1
            detectable += d["detectable_by_rules"]
            hit = d["expected_rule"] in fired
            caught += hit
            if not hit:
                residual = True
        claims_with_residual += residual
        actionable = {i.rule_id for i in report.issues if i.severity.value in ("ERROR", "WARNING")}
        expected_rules = {d["expected_rule"] for d in defects if d["expected_rule"]}
        unexpected = sorted(actionable - expected_rules)
        if unexpected:
            false_positive_claims.append({"claim_id": row["claim_id"], "unexpected_rules": unexpected})
        detail.append({"claim_id": row["claim_id"], "defects": [d["defect"] for d in defects],
                       "rules_fired": sorted(fired), "risk": report.risk_level.value})
    pre = claims_with_defect / n
    post = claims_with_residual / n
    return {
        "claims": n,
        "claims_with_seeded_defects": claims_with_defect,
        "seeded_defects": seeded,
        "defects_detectable_by_rule_design": detectable,
        "defects_caught": caught,
        "defect_recall": round(caught / seeded, 4) if seeded else None,
        "claim_error_rate_without_precheck": round(pre, 4),
        "claim_error_rate_after_fixing_flags": round(post, 4),
        "relative_error_rate_reduction": round((pre - post) / pre, 4) if pre else None,
        "claims_with_unexpected_flags": false_positive_claims,
        "detail": detail,
    }


def evaluate_denials(k: int = 3) -> dict[str, Any]:
    rows = load_denial_truth()
    tp = fp = fn = 0
    retrieval_hits = retrieval_total = 0
    detail = []
    index = _index()
    for row in rows:
        text = (SYNTHETIC_DENIALS_DIR / row["file"]).read_text()
        got = {(c.code_type, c.code) for c in extract_codes(text)}
        exp = {(c["code_type"], c["code"]) for c in row["expected_codes"]}
        tp += len(got & exp)
        fp += len(got - exp)
        fn += len(exp - got)
        # Retrieval: does the app's query surface the right guidance passage?
        # Exact code passages are always included by lookup, so only the
        # ranked knowledge-base passages (KB-*) are scored here.
        codes = extract_codes(text)
        query = text + " " + " ".join(f"{c.code} {c.description or ''}" for c in codes)
        ranked = [p.id for p, _ in index.score(query) if p.id.startswith("KB-")][:k]
        for kb in row.get("expected_guidance", []):
            retrieval_total += 1
            retrieval_hits += kb in ranked
        detail.append({"denial_id": row["denial_id"], "expected": sorted(exp), "extracted": sorted(got),
                       "top_k_guidance": ranked, "expected_guidance": row.get("expected_guidance", [])})
    return {
        "denials": len(rows),
        "code_precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "code_recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "codes_expected": tp + fn,
        "guidance_passages_expected": retrieval_total,
        "guidance_retrieval_recall_at_k": round(retrieval_hits / retrieval_total, 4) if retrieval_total else None,
        "k": k,
        "detail": detail,
    }
