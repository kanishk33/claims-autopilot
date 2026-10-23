"""Pydantic data models (the "shape" of a claim, an issue, and a denial).

`ClaimData` is exactly what we ask the LLM to return through OpenAI
Structured Outputs. `ClaimPacket` adds the fields our own code computes
afterwards (field status, verification results) and is what the app shows.

The "before" validators make the models forgiving about *shape* (for example
an NPI sent as a number, or insurance sent as a single string) without ever
inventing *values*.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

POINTER_LETTERS = "ABCDEFGHIJKL"  # CMS-1500 box 21 labels diagnoses A-L


def _to_str_or_none(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return str(int(v)) if float(v).is_integer() else str(v)
    if isinstance(v, str):
        v = v.strip()
        return v or None
    return v


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore")


# ---------------------------------------------------------------------------
# Claim parts
# ---------------------------------------------------------------------------
class Patient(_Base):
    full_name: Optional[str] = Field(None, description="Patient name exactly as written.")
    date_of_birth: Optional[str] = Field(None, description="YYYY-MM-DD if a date of birth is written, else null.")
    sex: Optional[str] = Field(None, description="Sex/gender exactly as written (e.g. F, M), else null.")

    _clean = field_validator("full_name", "date_of_birth", "sex", mode="before")(_to_str_or_none)


class Insurance(_Base):
    payer_name: Optional[str] = Field(None, description="Insurance company / plan name ONLY. Never include the member ID.")
    plan_type: Optional[str] = Field(None, description="Plan type if written (PPO, HMO, Medicare Advantage...), else null.")
    member_id: Optional[str] = Field(None, description="Member/subscriber ID ONLY, exactly as written, e.g. EP-78451233.")
    group_number: Optional[str] = Field(None, description="Group number if written, else null.")

    _clean = field_validator("payer_name", "plan_type", "member_id", "group_number", mode="before")(_to_str_or_none)


class Provider(_Base):
    name: Optional[str] = Field(None, description="Provider or organization name exactly as written.")
    npi: Optional[str] = Field(None, description="10-digit NPI as a string, exactly as written, else null.")

    _clean = field_validator("name", "npi", mode="before")(_to_str_or_none)


class Diagnosis(_Base):
    pointer: int = Field(..., description="1 for the first diagnosis listed, 2 for the second, and so on.")
    code: str = Field(..., description="ICD-10-CM code exactly as written, e.g. J02.9.")
    description: Optional[str] = Field(None, description="Description if written, else null.")

    @field_validator("pointer", mode="before")
    @classmethod
    def _letter_pointer(cls, v: Any) -> Any:
        if isinstance(v, str) and v.strip().upper() in POINTER_LETTERS and v.strip():
            return POINTER_LETTERS.index(v.strip().upper()) + 1
        return v

    @field_validator("code", mode="before")
    @classmethod
    def _code(cls, v: Any) -> Any:
        return v.strip().upper() if isinstance(v, str) else v


class ServiceLine(_Base):
    line_number: int = Field(..., description="1 for the first procedure line, 2 for the second, ...")
    procedure_code: str = Field(..., description="CPT or HCPCS code exactly as written, e.g. 99213.")
    description: Optional[str] = Field(None, description="Description if written, else null.")
    modifiers: list[str] = Field(default_factory=list, description="Modifiers such as 25, 59, QW. Empty list if none written.")
    units: Optional[int] = Field(None, description="Units/days as an integer if written, else null.")
    diagnosis_pointers: list[int] = Field(default_factory=list, description="Pointers to diagnoses (1 = first diagnosis). Empty if none written.")
    charge: Optional[float] = Field(None, description="Charge in dollars if written, else null.")

    @field_validator("procedure_code", mode="before")
    @classmethod
    def _code(cls, v: Any) -> Any:
        v = _to_str_or_none(v)
        return v.upper() if isinstance(v, str) else v

    @field_validator("modifiers", mode="before")
    @classmethod
    def _modifiers(cls, v: Any) -> Any:
        if v is None:
            return []
        if isinstance(v, str):
            v = re.split(r"[\s,;/-]+", v)
        return [str(m).strip().upper() for m in v if str(m).strip()]

    @field_validator("diagnosis_pointers", mode="before")
    @classmethod
    def _pointers(cls, v: Any) -> Any:
        if v is None:
            return []
        if isinstance(v, (int, str)):
            v = re.split(r"[\s,;]+", str(v))
        out = []
        for p in v:
            p = str(p).strip().upper()
            if not p:
                continue
            out.append(POINTER_LETTERS.index(p) + 1 if p in POINTER_LETTERS else int(p))
        return out

    @field_validator("units", mode="before")
    @classmethod
    def _units(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return None
            return int(float(v))
        if isinstance(v, float) and v.is_integer():
            return int(v)
        return v

    @field_validator("charge", mode="before")
    @classmethod
    def _charge(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = v.replace("$", "").replace(",", "").strip()
            return float(v) if v else None
        return v


class VerificationNote(_Base):
    field: str = Field(..., description="Dotted field path, e.g. insurance.member_id")
    reason: str = Field(..., description="Why this needs a human to verify it.")


class ClaimData(_Base):
    """What the LLM returns. Every field is nullable so the model can say 'not
    present' instead of guessing."""

    patient: Patient = Field(default_factory=Patient)
    insurance: Insurance = Field(default_factory=Insurance)
    billing_provider: Provider = Field(default_factory=Provider)
    rendering_provider: Provider = Field(default_factory=Provider)
    ordering_provider: Provider = Field(default_factory=Provider)
    service_date: Optional[str] = Field(None, description="Date of service as YYYY-MM-DD, else null.")
    place_of_service: Optional[str] = Field(None, description="2-digit CMS place of service code, e.g. 11, else null.")
    prior_authorization_number: Optional[str] = Field(None, description="Prior authorization / referral number if written, else null.")
    diagnoses: list[Diagnosis] = Field(default_factory=list)
    service_lines: list[ServiceLine] = Field(default_factory=list)
    needs_verification: list[VerificationNote] = Field(
        default_factory=list, description="Values that are present but ambiguous, illegible, or contradictory."
    )

    _clean = field_validator("service_date", "place_of_service", "prior_authorization_number", mode="before")(
        _to_str_or_none
    )

    @field_validator("insurance", mode="before")
    @classmethod
    def _insurance_string(cls, v: Any) -> Any:
        # A plain string is treated as the payer name only. We never try to
        # split a member ID out of it here; the source verifier handles that.
        if isinstance(v, str):
            return {"payer_name": v}
        return v

    @field_validator("place_of_service", mode="after")
    @classmethod
    def _pos(cls, v: Optional[str]) -> Optional[str]:
        if v and v.isdigit() and len(v) == 1:
            return v.zfill(2)
        return v


# ---------------------------------------------------------------------------
# Final packet shown in the app
# ---------------------------------------------------------------------------
class FieldStatus(str, Enum):
    PROVIDED = "provided"
    MISSING = "missing"
    NEEDS_VERIFICATION = "needs_verification"


class VerifierAction(_Base):
    field: str
    action: str  # "removed_unsupported" | "recovered_from_source" | "conflict"
    llm_value: Optional[str] = None
    source_value: Optional[str] = None
    detail: str = ""


class ClaimPacket(ClaimData):
    synthetic_data: bool = True
    extraction_method: str = "openai_structured_output+source_verification"
    model: Optional[str] = None
    field_status: dict[str, FieldStatus] = Field(default_factory=dict)
    verifier_actions: list[VerifierAction] = Field(default_factory=list)

    @property
    def missing_fields(self) -> list[str]:
        return [f for f, s in self.field_status.items() if s == FieldStatus.MISSING]

    @property
    def fields_to_verify(self) -> list[str]:
        return [f for f, s in self.field_status.items() if s == FieldStatus.NEEDS_VERIFICATION]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
class IssueCategory(str, Enum):
    VALIDATION_ERROR = "validation_error"  # definitely wrong/missing per our rules
    DENIAL_RISK = "denial_risk"  # may cause a denial depending on payer/policy
    INFO = "info"  # advisory only


class Severity(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Issue(_Base):
    rule_id: str
    category: IssueCategory
    severity: Severity
    field: str
    message: str
    why_it_matters: str
    question: Optional[str] = None
    demo_assumption: bool = False
    line_number: Optional[int] = None
    points: int = 0


class ValidationReport(_Base):
    issues: list[Issue]
    risk_level: RiskLevel
    risk_score: int
    risk_explanation: list[str]

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.category == IssueCategory.VALIDATION_ERROR]

    @property
    def risks(self) -> list[Issue]:
        return [i for i in self.issues if i.category == IssueCategory.DENIAL_RISK]

    @property
    def ready_to_submit(self) -> bool:
        return not self.errors


# ---------------------------------------------------------------------------
# Denials
# ---------------------------------------------------------------------------
class DenialCode(_Base):
    code_type: str  # "CARC" or "RARC"
    code: str
    group_code: Optional[str] = None  # CO, PR, OA, PI, CR (CARC only)
    description: Optional[str] = None  # from the local reference file only
    source: Optional[str] = None
    found_in_reference: bool = False


class ClaimLineDenial(_Base):
    """One adjudicated line found in remittance text."""

    procedure_code: Optional[str] = None
    billed: Optional[float] = None
    paid: Optional[float] = None


class DenialAnalysis(_Base):
    """What the LLM returns for the denial copilot (Structured Outputs)."""

    plain_language_summary: str = Field(..., description="2-4 sentences a billing specialist can read quickly.")
    likely_root_cause: str = Field(..., description="Most likely cause, grounded ONLY in the denial text, codes, and claim data given.")
    recommended_path: str = Field(
        ..., description="One of: corrected_claim, appeal, provide_information, patient_responsibility, needs_human_review"
    )
    correction_steps: list[str] = Field(..., description="Ordered, concrete steps. Reference codes/sources by id.")
    information_needed: list[str] = Field(..., description="Facts or documents a human must supply. Never invent them.")
    cited_source_ids: list[str] = Field(..., description="Ids of the retrieved reference passages you relied on.")


class AppealLetter(_Base):
    subject: str
    body: str = Field(..., description="Letter text. Use [BRACKETED PLACEHOLDERS] for anything not provided.")
    placeholders: list[str] = Field(..., description="Every [PLACEHOLDER] used in the body.")
    attachments_to_gather: list[str] = Field(..., description="Documents a human must locate. Never claim they are attached.")

    @model_validator(mode="after")
    def _placeholders_listed(self) -> "AppealLetter":
        found = sorted(set(re.findall(r"\[[A-Z0-9 _/#\-\.]+\]", self.body)))
        missing = [p for p in found if p not in self.placeholders]
        if missing:
            self.placeholders = self.placeholders + missing
        return self
