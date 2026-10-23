"""Feature 3: Denial Resolution Copilot.

1. Find CARC/RARC codes in remittance text (deterministic regex).
2. Look up each code in the local reference table (never the LLM's memory).
3. Retrieve related guidance passages (BM25, see retrieval.py).
4. Ask the LLM to explain the denial and plan the fix, using only steps 1-3
   and the claim data as context (RAG).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from .extractor import get_path
from .llm import LLMResult, parse_structured
from .retrieval import Passage, lookup_code, retrieve
from .schemas import ClaimData, DenialAnalysis, DenialCode

log = logging.getLogger(__name__)

GROUP_CODES = ("CO", "PR", "OA", "PI", "CR")
CARC_WITH_GROUP = re.compile(r"\b(CO|PR|OA|PI|CR)\s*[-:]?\s*([A-Z]?\d{1,3})\b")
CARC_LABELED = re.compile(r"\b(?:CARC|reason code|adjustment reason(?: code)?)s?\s*[:#]?\s*((?:[A-Z]?\d{1,3}(?:\s*,\s*|\s+and\s+|\s+)?)+)",
                          re.IGNORECASE)
RARC_RE = re.compile(r"(?<![A-Z0-9-])(MA\d{2,3}|M\d{1,3}|N\d{1,3})(?![A-Z0-9])")

# Which claim field each code usually points at (used for the cross-check).
CODE_TO_FIELD = {
    "N264": "ordering_provider.name",
    "N265": "ordering_provider.npi",
    "N290": "rendering_provider.npi",
    "N257": "billing_provider.npi",
    "M77": "place_of_service",
    "M76": "diagnoses",
    "M51": "service_lines",
    "M20": "service_lines",
    "31": "insurance.member_id",
    "197": "prior_authorization_number",
}


def extract_codes(text: str) -> list[DenialCode]:
    """All CARC (with group code when present) and RARC codes, in order, no duplicates."""
    found: list[tuple[int, DenialCode]] = []
    seen: set[tuple[str, str]] = set()

    def add(pos: int, code_type: str, code: str, group: str | None = None) -> None:
        key = (code_type, code)
        if key in seen:
            return
        seen.add(key)
        ref = lookup_code(code_type, code)
        found.append((pos, DenialCode(
            code_type=code_type, code=code, group_code=group,
            description=ref["official_description"] if ref else None,
            source=ref["source_url"] if ref else None,
            found_in_reference=ref is not None,
        )))

    for m in CARC_WITH_GROUP.finditer(text):
        add(m.start(), "CARC", m.group(2).lstrip("0") or "0", m.group(1))
    for m in CARC_LABELED.finditer(text):
        for c in re.findall(r"[A-Z]?\d{1,3}", m.group(1)):
            add(m.start(), "CARC", c.lstrip("0") or "0")
    for m in RARC_RE.finditer(text):
        add(m.start(), "RARC", m.group(1))
    return [c for _, c in sorted(found, key=lambda x: x[0])]


def extract_claim_reference(text: str) -> str | None:
    m = re.search(r"claim\s*(?:#|number|no\.?|id)\s*[:#]?\s*([A-Z0-9][A-Z0-9\-]{3,})", text, re.IGNORECASE)
    return m.group(1) if m else None


def claim_crosscheck(codes: list[DenialCode], claim: ClaimData | None) -> list[str]:
    """Compare denial codes with the claim currently loaded in the app."""
    if claim is None:
        return []
    notes = []
    for c in codes:
        path = CODE_TO_FIELD.get(c.code)
        if not path:
            continue
        value = get_path(claim, path)
        if value in (None, "", []):
            notes.append(f"{c.code_type} {c.code}: the claim still has no value for `{path}`. This must be obtained before resubmitting.")
        else:
            shown = value if isinstance(value, str) else f"{len(value)} item(s)"
            notes.append(f"{c.code_type} {c.code}: the claim now has `{path}` = {shown}. Confirm it is correct and include it on the corrected claim.")
    return notes


def deterministic_steps(codes: list[DenialCode]) -> list[dict[str, str]]:
    """Reference-table guidance for each code. Shown even if the LLM is unavailable."""
    rows = []
    for c in codes:
        ref = lookup_code(c.code_type, c.code)
        if ref:
            rows.append({"code": f"{c.group_code + '-' if c.group_code else ''}{c.code}",
                         "type": c.code_type,
                         "official_description": ref["official_description"],
                         "suggested_action (project guidance)": ref["suggested_actions_project_guidance"],
                         "typical_path": ref["typical_path"],
                         "source": ref["source_url"]})
        else:
            rows.append({"code": c.code, "type": c.code_type,
                         "official_description": "Not in the local reference subset. Look it up at x12.org/codes.",
                         "suggested_action (project guidance)": "", "typical_path": "needs_human_review", "source": ""})
    return rows


ANALYSIS_PROMPT = """You are a revenue-cycle denial analyst helping a human billing specialist.
You receive SYNTHETIC remittance text, the denial codes with their OFFICIAL
descriptions from a local reference file, retrieved guidance passages (each
with an id), and the claim data currently on file.

Rules:
- Use ONLY the information provided. Do not use outside code definitions.
- Never invent evidence, NPIs, authorization numbers, dates, or clinical facts.
  If something is needed, put it in information_needed.
- Pick recommended_path from: corrected_claim, appeal, provide_information,
  patient_responsibility, needs_human_review. Data-error denials (e.g. CARC 16
  with N-codes, MA130) are normally corrected_claim, not appeal.
- cited_source_ids must be ids of passages you actually used.
- Write for a busy billing specialist: plain, specific, short."""


@dataclass
class DenialResult:
    source_text: str
    claim_reference: str | None
    codes: list[DenialCode]
    passages: list[Passage]
    reference_steps: list[dict[str, str]]
    crosscheck: list[str]
    analysis: DenialAnalysis | None = None
    llm: LLMResult | None = None
    warnings: list[str] = field(default_factory=list)


def build_context(text: str, codes: list[DenialCode], passages: list[Passage], claim: ClaimData | None) -> str:
    parts = ["## Remittance text (synthetic)", text.strip(), "", "## Denial codes (official descriptions from local reference)"]
    for c in codes:
        parts.append(f"- {c.code_type} {(c.group_code + '-') if c.group_code else ''}{c.code}: "
                     f"{c.description or 'NOT IN LOCAL REFERENCE'}")
    parts += ["", "## Retrieved passages"]
    for p in passages:
        parts.append(f"[{p.id}] {p.title}: {p.text}")
    if claim is not None:
        parts += ["", "## Claim data on file (JSON)", claim.model_dump_json(exclude_none=True)]
    return "\n".join(parts)


def prepare_denial(text: str, claim: ClaimData | None = None, k: int = 4) -> DenialResult:
    """Everything except the LLM call (fast, deterministic, testable)."""
    if not text or not text.strip():
        raise ValueError("The denial text is empty.")
    codes = extract_codes(text)
    ids = [f"{c.code_type}-{c.code}" for c in codes] + [f"GROUP-{c.group_code}" for c in codes if c.group_code]
    query = text + " " + " ".join(f"{c.code} {c.description or ''}" for c in codes)
    passages = retrieve(query, k=k, must_include=list(dict.fromkeys(ids)))
    result = DenialResult(
        source_text=text, claim_reference=extract_claim_reference(text), codes=codes, passages=passages,
        reference_steps=deterministic_steps(codes), crosscheck=claim_crosscheck(codes, claim),
    )
    if not codes:
        result.warnings.append("No CARC or RARC codes were found in the text.")
    unknown = [c.code for c in codes if not c.found_in_reference]
    if unknown:
        result.warnings.append(f"Not in the local reference subset (meaning not shown): {', '.join(unknown)}.")
    return result


def analyze_denial(text: str, claim: ClaimData | None = None, *, client: Any | None = None) -> DenialResult:
    """Deterministic preparation + LLM explanation. Raises LLMError on API failure."""
    result = prepare_denial(text, claim)
    context = build_context(text, result.codes, result.passages, claim)
    llm = parse_structured(ANALYSIS_PROMPT, context, DenialAnalysis, client=client)
    analysis: DenialAnalysis = llm.parsed
    valid_ids = {p.id for p in result.passages}
    bad = [i for i in analysis.cited_source_ids if i not in valid_ids]
    if bad:
        result.warnings.append(f"The model cited sources that were not retrieved and were removed: {', '.join(bad)}.")
        analysis.cited_source_ids = [i for i in analysis.cited_source_ids if i in valid_ids]
    result.analysis, result.llm = analysis, llm
    log.info("Denial analyzed: %d codes, path=%s", len(result.codes), analysis.recommended_path)
    return result
