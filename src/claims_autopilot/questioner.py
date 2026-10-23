"""Turns validation issues into a short list of targeted follow-up questions
and a pre-submission checklist.

The questions themselves come from the rules file (deterministic). An
optional LLM step rewrites them into one polite message to the provider's
office; it may only rephrase the questions it is given.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .llm import LLMResult, parse_structured
from .schemas import Issue, IssueCategory, Severity, ValidationReport


def follow_up_questions(report: ValidationReport) -> list[dict[str, str]]:
    """Unique questions, most important first."""
    seen: set[str] = set()
    out = []
    for issue in report.issues:
        if not issue.question or issue.question in seen:
            continue
        seen.add(issue.question)
        out.append({"priority": issue.severity.value, "question": issue.question, "rule_id": issue.rule_id,
                    "field": issue.field})
    return out


def checklist(report: ValidationReport) -> list[dict[str, Any]]:
    """Rows for the pre-submission checklist table."""
    label = {
        IssueCategory.VALIDATION_ERROR: "Must fix",
        IssueCategory.DENIAL_RISK: "Review (denial risk)",
        IssueCategory.INFO: "FYI",
    }
    return [
        {
            "Status": "☐",
            "Type": label[i.category],
            "Severity": i.severity.value,
            "Line": i.line_number or "",
            "Issue": i.message,
            "Why it matters": i.why_it_matters,
            "Rule": i.rule_id + (" (demo assumption)" if i.demo_assumption else ""),
        }
        for i in report.issues
    ]


class ProviderQuery(BaseModel):
    subject: str
    message: str = Field(..., description="Polite message to the provider office containing ONLY the given questions.")


QUERY_PROMPT = """You write short, polite messages from a medical billing team to a
provider's office. Combine the given questions into one message. Do not add
new questions, facts, codes, or patient details. Do not include any patient
identifiers other than the claim reference given."""


def draft_provider_query(issues: list[Issue], claim_ref: str, *, client: Any | None = None) -> LLMResult:
    questions = [i.question for i in issues if i.question and i.severity != Severity.INFO]
    questions = list(dict.fromkeys(questions))
    if not questions:
        raise ValueError("There are no open questions to send.")
    content = f"Claim reference: {claim_ref}\nQuestions:\n" + "\n".join(f"- {q}" for q in questions)
    return parse_structured(QUERY_PROMPT, content, ProviderQuery, client=client)
