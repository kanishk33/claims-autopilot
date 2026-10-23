"""Feature 2: Pre-Submission Denial Prevention (deterministic rules).

All rules live in data/validation_rules.yml so they can be changed without
touching code. The validator never calls the LLM: the same claim always gets
the same result, and every issue says which rule produced it.
"""

from __future__ import annotations

import re
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .config import RULES_PATH
from .extractor import get_path
from .schemas import (
    ClaimData,
    Issue,
    IssueCategory,
    RiskLevel,
    Severity,
    ValidationReport,
)
from .source_parser import normalize_date


@lru_cache(maxsize=4)
def load_rules(path: str | Path = RULES_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def npi_check_digit_ok(npi: str) -> bool:
    """CMS NPI check digit: Luhn algorithm over '80840' + the 10 digits."""
    if not re.fullmatch(r"\d{10}", npi or ""):
        return False
    total = 0
    for i, ch in enumerate(reversed("80840" + npi)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _fmt(template: str, **kw: Any) -> str:
    try:
        return template.format(**kw)
    except (KeyError, IndexError):
        return template


class _Collector:
    def __init__(self, points: dict[str, int]):
        self.issues: list[Issue] = []
        self.points = points

    def add(self, rule_id: str, category: IssueCategory, severity: Severity, field: str, message: str,
            why: str, question: str | None = None, demo: bool = False, line: int | None = None) -> None:
        self.issues.append(Issue(
            rule_id=rule_id, category=category, severity=severity, field=field, message=message,
            why_it_matters=why, question=question, demo_assumption=demo, line_number=line,
            points=self.points.get(severity.value, 0),
        ))


def validate_claim(claim: ClaimData, *, rules: dict[str, Any] | None = None, as_of: date | None = None) -> ValidationReport:
    rules = rules or load_rules()
    as_of = as_of or date.today()
    fmt = rules["format_rules"]
    c = _Collector(rules["risk_scoring"]["points"])
    VE, DR, INFO = IssueCategory.VALIDATION_ERROR, IssueCategory.DENIAL_RISK, IssueCategory.INFO

    # 1. Required fields --------------------------------------------------
    for r in rules["required_fields"]:
        if get_path(claim, r["field"]) in (None, ""):
            sev = Severity(r["severity"])
            c.add(r["id"], VE if sev == Severity.ERROR else DR, sev, r["field"], r["message"],
                  r["why_it_matters"], r.get("question"), r.get("demo_assumption", False))
    if not claim.diagnoses:
        c.add("REQ_DIAGNOSIS", VE, Severity.ERROR, "diagnoses", "No diagnosis codes on the claim.",
              "Every professional claim line must point to at least one ICD-10-CM diagnosis.",
              "What diagnosis (ICD-10-CM) supports the services billed?")
    if not claim.service_lines:
        c.add("REQ_SERVICE_LINE", VE, Severity.ERROR, "service_lines", "No procedure (CPT/HCPCS) lines on the claim.",
              "A claim must contain at least one billable service line.",
              "Which services/procedures were performed and should be billed?")

    # 2. Formats ------------------------------------------------------------
    for role in ("billing_provider", "rendering_provider", "ordering_provider"):
        npi = get_path(claim, f"{role}.npi")
        if not npi:
            continue
        label = role.replace("_", " ")
        if not re.fullmatch(r"\d{10}", npi):
            c.add("FMT_NPI_LENGTH", VE, Severity.ERROR, f"{role}.npi", f"The {label} NPI '{npi}' is not 10 digits.",
                  "An NPI is always exactly 10 digits; a malformed NPI is rejected before adjudication.",
                  f"Please confirm the {label}'s 10-digit NPI.")
        elif fmt.get("npi_check_digit") and not npi_check_digit_ok(npi):
            c.add("FMT_NPI_CHECK_DIGIT", VE, Severity.ERROR, f"{role}.npi",
                  f"The {label} NPI '{npi}' fails the NPI check-digit test.",
                  "CMS NPIs end in a check digit (Luhn formula). A failed check almost always means a typo or placeholder number.",
                  f"Please re-check the {label}'s NPI against NPPES.")

    dob = normalize_date(claim.patient.date_of_birth) if claim.patient.date_of_birth else None
    dos = normalize_date(claim.service_date) if claim.service_date else None
    if claim.patient.date_of_birth and not dob:
        c.add("FMT_DOB", VE, Severity.ERROR, "patient.date_of_birth", f"Date of birth '{claim.patient.date_of_birth}' is not a valid date.",
              "Invalid dates fail front-end claim edits.", "What is the patient's date of birth (YYYY-MM-DD)?")
    if claim.service_date and not dos:
        c.add("FMT_DOS", VE, Severity.ERROR, "service_date", f"Date of service '{claim.service_date}' is not a valid date.",
              "Invalid dates fail front-end claim edits.", "On what date was the service performed (YYYY-MM-DD)?")
    if dob and dos and dob > dos:
        c.add("FMT_DOB_AFTER_DOS", VE, Severity.ERROR, "patient.date_of_birth", "Date of birth is after the date of service.",
              "This is logically impossible and indicates a data-entry error.", "Please confirm both dates.")
    if dos:
        dos_d = date.fromisoformat(dos)
        if dos_d > as_of:
            c.add("FMT_DOS_FUTURE", VE, Severity.ERROR, "service_date", f"Date of service {dos} is in the future.",
                  "Claims cannot be billed for services that have not happened yet.", "Please confirm the date of service.")
        elif (as_of - dos_d).days > fmt.get("timely_filing_days", 365):
            c.add("TIMELY_FILING", DR, Severity.WARNING, "service_date",
                  f"Date of service is more than {fmt['timely_filing_days']} days ago.",
                  "Payers set filing deadlines (CARC 29 is used when the limit has expired). Limits vary by payer and contract.",
                  "Is there proof of timely original submission, or does this payer allow a longer filing window?", demo=True)

    pos_codes = fmt.get("place_of_service", {})
    if claim.place_of_service and claim.place_of_service not in pos_codes:
        c.add("FMT_POS", VE, Severity.ERROR, "place_of_service",
              f"Place of service '{claim.place_of_service}' is not in the configured POS list.",
              "An invalid place of service is a common front-end rejection (RARC M77).",
              "Where was the service performed?")

    member, payer = claim.insurance.member_id, claim.insurance.payer_name
    if member and payer and (payer.lower() in member.lower() or " " in member.strip()):
        c.add("FMT_MEMBER_ID_COMBINED", VE, Severity.ERROR, "insurance.member_id",
              "The member ID appears to contain the payer name or spaces.",
              "If the member ID does not exactly match the payer's records, the patient cannot be identified (CARC 31).",
              "What is the member ID exactly as printed on the card?")

    icd_re = re.compile(fmt["icd10_pattern"])
    if len(claim.diagnoses) > fmt.get("max_diagnoses", 12):
        c.add("FMT_TOO_MANY_DX", VE, Severity.ERROR, "diagnoses", "More than 12 diagnoses on one claim.",
              "The professional claim format holds at most 12 diagnosis codes.", "Which diagnoses are most relevant to the services?")
    for dx in claim.diagnoses:
        if not icd_re.match(dx.code):
            c.add("FMT_ICD10", VE, Severity.ERROR, "diagnoses", f"'{dx.code}' is not a valid ICD-10-CM code format.",
                  "Invalid diagnosis codes are rejected (RARC M76).", f"What is the correct ICD-10-CM code for diagnosis {dx.pointer}?")

    proc_re, mod_re = re.compile(fmt["procedure_pattern"]), re.compile(fmt["modifier_pattern"])
    dx_pointers = {d.pointer for d in claim.diagnoses}
    for line in claim.service_lines:
        n, code = line.line_number, line.procedure_code
        if not proc_re.match(code):
            c.add("FMT_PROCEDURE_CODE", VE, Severity.ERROR, "service_lines", f"Line {n}: '{code}' is not a valid CPT/HCPCS format.",
                  "Invalid procedure codes are rejected (RARC M51).", f"What is the correct procedure code for line {n}?", line=n)
        for m in line.modifiers:
            if not mod_re.match(m):
                c.add("FMT_MODIFIER", VE, Severity.ERROR, "service_lines", f"Line {n}: modifier '{m}' is not a 2-character modifier.",
                      "Modifiers are exactly two characters.", f"Which modifier was intended on line {n}?", line=n)
        if line.units is None:
            c.add("REQ_UNITS", VE, Severity.ERROR, "service_lines", f"Line {n} ({code}): units are missing.",
                  "Units determine payment; a line without units cannot be priced.", f"How many units of {code} were performed?", line=n)
        elif line.units < 1:
            c.add("FMT_UNITS", VE, Severity.ERROR, "service_lines", f"Line {n} ({code}): units must be at least 1.",
                  "Zero or negative units are invalid.", f"How many units of {code} were performed?", line=n)
        if not line.diagnosis_pointers:
            c.add("REQ_DX_POINTER", VE, Severity.ERROR, "service_lines", f"Line {n} ({code}): no diagnosis pointer.",
                  "Each service line must point to the diagnosis that supports it (box 24E).",
                  f"Which diagnosis supports {code}?", line=n)
        else:
            bad = [p for p in line.diagnosis_pointers if p not in dx_pointers]
            if bad:
                c.add("FMT_DX_POINTER_INVALID", VE, Severity.ERROR, "service_lines",
                      f"Line {n} ({code}): pointer(s) {bad} do not match any listed diagnosis.",
                      "A pointer to a diagnosis that is not on the claim is a front-end error.",
                      f"Which listed diagnosis supports {code}?", line=n)
            if len(line.diagnosis_pointers) > fmt.get("max_pointers_per_line", 4):
                c.add("FMT_TOO_MANY_POINTERS", VE, Severity.ERROR, "service_lines",
                      f"Line {n} ({code}): more than 4 diagnosis pointers.",
                      "The professional claim format allows at most 4 pointers per line.",
                      f"Which (up to 4) diagnoses best support {code}?", line=n)

    seen: dict[tuple, int] = {}
    for line in claim.service_lines:
        key = (line.procedure_code, tuple(sorted(line.modifiers)))
        if key in seen:
            c.add("DUPLICATE_LINE", DR, Severity.WARNING, "service_lines",
                  f"Line {line.line_number} repeats {line.procedure_code} from line {seen[key]} with the same modifiers.",
                  "Identical lines on the same date are often denied as duplicates (CARC 18) unless a modifier explains them.",
                  f"Was {line.procedure_code} really performed more than once? If so, which modifier (e.g. 76, 91, 59) applies?",
                  demo=True, line=line.line_number)
        else:
            seen[key] = line.line_number

    # 3. Code-specific rules (configurable demo assumptions) ---------------
    codes = [l.procedure_code for l in claim.service_lines]
    for r in rules.get("code_rules", []):
        cat, sev, demo = IssueCategory(r["category"]), Severity(r["severity"]), r.get("demo_assumption", False)
        rid = r["id"]
        if "requires" in r:
            for line in claim.service_lines:
                if line.procedure_code in r["codes"]:
                    missing = [f for f in r["requires"] if get_path(claim, f) in (None, "")]
                    if missing:
                        k = {"code": line.procedure_code}
                        c.add(rid, cat, sev, ", ".join(missing), _fmt(r["message"], **k), r["why_it_matters"],
                              _fmt(r["question"], **k), demo, line.line_number)
        elif "procedure_codes" in r:
            if any(code in r["procedure_codes"] for code in codes):
                for line in claim.service_lines:
                    if line.procedure_code in r["em_codes"] and r["required_modifier"] not in line.modifiers:
                        c.add(rid, cat, sev, "service_lines", _fmt(r["message"], code=line.procedure_code),
                              r["why_it_matters"], r["question"], demo, line.line_number)
        elif "place_of_service" in r:
            if claim.place_of_service in r["place_of_service"]:
                for line in claim.service_lines:
                    if line.procedure_code in r["em_codes"] and r["required_modifier"] not in line.modifiers:
                        c.add(rid, cat, sev, "service_lines", _fmt(r["message"], code=line.procedure_code),
                              r["why_it_matters"], r["question"], demo, line.line_number)
        elif "max_units" in r:
            for line in claim.service_lines:
                mx = r["max_units"].get(line.procedure_code)
                if mx is not None and line.units is not None and line.units > mx:
                    k = {"code": line.procedure_code, "units": line.units, "max": mx}
                    c.add(rid, cat, sev, "service_lines", _fmt(r["message"], **k), r["why_it_matters"],
                          _fmt(r["question"], **k), demo, line.line_number)
        elif "required_modifier" in r and "codes" in r:
            for line in claim.service_lines:
                if line.procedure_code in r["codes"] and r["required_modifier"] not in line.modifiers:
                    k = {"code": line.procedure_code}
                    c.add(rid, cat, sev, "service_lines", _fmt(r["message"], **k), r["why_it_matters"],
                          _fmt(r["question"], **k), demo, line.line_number)

    return _score(c.issues, rules)


def _score(issues: list[Issue], rules: dict[str, Any]) -> ValidationReport:
    th = rules["risk_scoring"]["thresholds"]
    score = sum(i.points for i in issues)
    level = RiskLevel.HIGH if score >= th["HIGH"] else RiskLevel.MEDIUM if score >= th["MEDIUM"] else RiskLevel.LOW
    order = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
    issues = sorted(issues, key=lambda i: (order[i.severity], i.line_number or 0))
    n_err = sum(1 for i in issues if i.severity == Severity.ERROR)
    n_warn = sum(1 for i in issues if i.severity == Severity.WARNING)
    explanation = [
        f"Score {score} = {n_err} error(s) x {rules['risk_scoring']['points']['ERROR']} "
        f"+ {n_warn} warning(s) x {rules['risk_scoring']['points']['WARNING']}.",
        f"Thresholds: HIGH >= {th['HIGH']}, MEDIUM >= {th['MEDIUM']}, otherwise LOW.",
    ]
    if n_err:
        explanation.append("HIGH because at least one validation error must be fixed before submission.")
    elif n_warn:
        explanation.append("No hard errors, but some payer-dependent denial risks should be reviewed.")
    else:
        explanation.append("No errors or warnings under the configured rules (this is not a guarantee of payment).")
    return ValidationReport(issues=issues, risk_level=level, risk_score=score, risk_explanation=explanation)
