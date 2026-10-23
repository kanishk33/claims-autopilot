"""Drafts a correction cover letter or appeal letter for HUMAN REVIEW.

The draft is never submitted anywhere. After the LLM writes it, we scan the
letter for identifiers (NPIs, codes, dollar amounts, dates) that were not in
the input and warn about them, as a guard against invented evidence.
"""

from __future__ import annotations

import re
from typing import Any

from .denial import DenialResult, build_context
from .llm import LLMResult, parse_structured
from .schemas import AppealLetter, ClaimData

LETTER_PROMPT = """You draft letters for a medical billing specialist to REVIEW and edit.
Write a {kind} for the denial described below.

Rules:
- Use only facts present in the context. Never invent clinical findings,
  dates, NPIs, authorization numbers, or evidence.
- Where a fact or document is needed but not provided, write a placeholder in
  square brackets, e.g. [ORDERING PROVIDER NPI], [ATTACH: signed order].
- Do not claim documents are enclosed; list them in attachments_to_gather.
- Start the body with "DRAFT - FOR HUMAN REVIEW - SYNTHETIC DATA".
- Keep it under 300 words, professional and specific."""

_ID_PATTERNS = [
    re.compile(r"(?<!\d)\d{10}(?!\d)"),                       # NPIs
    re.compile(r"\b[A-TV-Z][0-9][0-9AB]\.[0-9A-TV-Z]{1,4}\b"),  # ICD-10 with dot
    re.compile(r"(?<![\d.])\d{4}[0-9FTU](?![\d.])"),          # CPT-like
    re.compile(r"\$\s?[\d,]+(?:\.\d{2})?"),                   # dollar amounts
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),                     # ISO dates
]


def unsupported_identifiers(letter_text: str, context: str) -> list[str]:
    """Identifiers in the letter that do not appear anywhere in the context."""
    flat_ctx = context.replace(",", "")
    out = []
    for pat in _ID_PATTERNS:
        for tok in pat.findall(letter_text):
            if tok.replace(",", "").replace(" ", "") not in flat_ctx.replace(" ", ""):
                out.append(tok)
    return sorted(set(out))


def draft_letter(denial: DenialResult, claim: ClaimData | None = None, *, client: Any | None = None) -> tuple[AppealLetter, LLMResult, list[str]]:
    path = denial.analysis.recommended_path if denial.analysis else "needs_human_review"
    kind = ("cover letter to accompany a corrected claim" if path in ("corrected_claim", "provide_information")
            else "first-level appeal letter")
    context = build_context(denial.source_text, denial.codes, denial.passages, claim)
    if denial.analysis:
        context += "\n\n## Analyst plan\n" + denial.analysis.model_dump_json()
    if denial.claim_reference:
        context += f"\n\nClaim reference: {denial.claim_reference}"
    llm = parse_structured(LETTER_PROMPT.format(kind=kind), context, AppealLetter, client=client)
    letter: AppealLetter = llm.parsed
    warnings = unsupported_identifiers(letter.body, context)
    return letter, llm, warnings
