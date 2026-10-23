"""Feature 1: Intelligent Claim Builder.

Pipeline:
    visit text --(OpenAI Structured Outputs)--> ClaimData
               --(verify_against_source)-----> ClaimPacket with field status

The LLM does the reading. The verifier makes sure the LLM did not invent
values and did not drop values that are clearly labeled in the source.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .llm import LLMResult, parse_structured
from .schemas import (
    ClaimData,
    ClaimPacket,
    Diagnosis,
    FieldStatus,
    ServiceLine,
    VerifierAction,
)
from .source_parser import normalize_date, parse_explicit, value_in_text

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a careful medical billing assistant that converts a SYNTHETIC
visit summary or superbill into structured professional-claim data.

Rules:
1. Copy values exactly as written. NEVER guess or invent a value. If a field is
   not written in the text, return null (or an empty list).
2. Keep fields separate: insurance.payer_name is only the payer/plan name;
   insurance.member_id is only the member ID. Never combine them.
3. NPIs are 10-digit strings. Billing, rendering, and ordering providers are
   different roles: only fill a role if the text names that role (if the text
   says one provider is both billing and rendering, fill both).
4. Extract EVERY procedure line (CPT/HCPCS) with its units, modifiers, and
   diagnosis pointers. Pointer 1 = first diagnosis listed, 2 = second, etc.
   Letters A-L mean 1-12. Do not add codes that are not written.
5. Diagnoses: list ICD-10-CM codes in the order written, pointer starting at 1.
   Do not assign codes from the narrative if no code is written.
6. Dates as YYYY-MM-DD. Place of service as the 2-digit code if written.
7. Put anything ambiguous, illegible, or contradictory in needs_verification
   with a short reason instead of choosing a value."""

# field path -> how to check it against the source text
SCALAR_FIELDS: dict[str, str] = {
    "patient.full_name": "name",
    "patient.date_of_birth": "date",
    "patient.sex": "text",
    "insurance.payer_name": "name",
    "insurance.plan_type": "text",
    "insurance.member_id": "id",
    "insurance.group_number": "id",
    "billing_provider.name": "name",
    "billing_provider.npi": "id",
    "rendering_provider.name": "name",
    "rendering_provider.npi": "id",
    "ordering_provider.name": "name",
    "ordering_provider.npi": "id",
    "service_date": "date",
    "place_of_service": "code",
    "prior_authorization_number": "id",
}
# Values that can reasonably be *derived* from wording (e.g. "office visit" ->
# POS 11). If unsupported we keep them but require human verification.
INFERABLE_FIELDS = {"place_of_service", "patient.sex", "insurance.plan_type"}


def get_path(obj: Any, path: str) -> Any:
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def set_path(obj: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def _same(a: str | None, b: str | None, kind: str) -> bool:
    if a is None or b is None:
        return a == b
    if kind == "date":
        return normalize_date(a) == normalize_date(b)
    if kind == "name":
        return value_in_text(a, b, "name") or value_in_text(b, a, "name")
    return "".join(ch for ch in a.lower() if ch.isalnum()) == "".join(ch for ch in b.lower() if ch.isalnum())


def verify_against_source(llm_data: ClaimData, source_text: str, model: str | None = None) -> ClaimPacket:
    """Cross-check LLM output with the source text. Never adds a value that is
    not written in the source."""
    packet = ClaimPacket.model_validate(llm_data.model_dump())
    packet.model = model
    explicit = parse_explicit(source_text)
    actions: list[VerifierAction] = []
    status: dict[str, FieldStatus] = {}

    # Member ID glued to the payer name is a known LLM failure mode.
    payer, member = packet.insurance.payer_name, packet.insurance.member_id
    if payer and member and member in payer and payer != member:
        packet.insurance.payer_name = payer.replace(member, "").strip(" ,;-#") or None
        actions.append(VerifierAction(field="insurance.payer_name", action="split_combined_value",
                                      llm_value=payer, detail="Removed the member ID from the payer name."))
    if payer and not member and explicit.insurance.member_id and explicit.insurance.member_id in payer:
        packet.insurance.payer_name = payer.replace(explicit.insurance.member_id, "").strip(" ,;-#") or None

    for path, kind in SCALAR_FIELDS.items():
        llm_value = get_path(packet, path)
        src_value = get_path(explicit, path)
        supported = value_in_text(llm_value, source_text, kind)

        if llm_value is not None and not supported:
            if src_value is not None:
                set_path(packet, path, src_value)
                actions.append(VerifierAction(field=path, action="conflict", llm_value=str(llm_value),
                                              source_value=src_value,
                                              detail="LLM value not found in source; used the labeled source value."))
                status[path] = FieldStatus.NEEDS_VERIFICATION
            elif path in INFERABLE_FIELDS:
                actions.append(VerifierAction(field=path, action="inferred", llm_value=str(llm_value),
                                              detail="Not written verbatim; the model inferred it. Confirm before submitting."))
                status[path] = FieldStatus.NEEDS_VERIFICATION
            else:
                set_path(packet, path, None)
                actions.append(VerifierAction(field=path, action="removed_unsupported", llm_value=str(llm_value),
                                              detail="Value does not appear in the source text, so it was removed."))
                status[path] = FieldStatus.MISSING
            continue

        if llm_value is None and src_value is not None:
            set_path(packet, path, src_value)
            actions.append(VerifierAction(field=path, action="recovered_from_source", source_value=src_value,
                                          detail="LLM left this empty but the source labels it explicitly."))
            status[path] = FieldStatus.PROVIDED
            continue

        if llm_value is not None and src_value is not None and not _same(str(llm_value), src_value, kind):
            if kind in ("id", "date"):
                # Labeled identifiers/dates are copied verbatim, so prefer them.
                set_path(packet, path, src_value)
            actions.append(VerifierAction(field=path, action="conflict", llm_value=str(llm_value), source_value=src_value,
                                          detail="LLM and labeled source value differ."))
            status[path] = FieldStatus.NEEDS_VERIFICATION
            continue

        status[path] = FieldStatus.PROVIDED if get_path(packet, path) is not None else FieldStatus.MISSING

    _verify_diagnoses(packet, explicit, source_text, actions, status)
    _verify_service_lines(packet, explicit, source_text, actions, status)

    for note in packet.needs_verification:
        status[note.field] = FieldStatus.NEEDS_VERIFICATION

    packet.field_status = status
    packet.verifier_actions = actions
    return packet


def _verify_diagnoses(packet: ClaimPacket, explicit: ClaimData, text: str,
                      actions: list[VerifierAction], status: dict[str, FieldStatus]) -> None:
    kept: list[Diagnosis] = []
    for dx in packet.diagnoses:
        if value_in_text(dx.code, text, "code"):
            kept.append(dx)
        else:
            actions.append(VerifierAction(field="diagnoses", action="removed_unsupported", llm_value=dx.code,
                                          detail="Diagnosis code not written in the source text."))
    have = {d.code.replace(".", "") for d in kept}
    for dx in explicit.diagnoses:
        if dx.code.replace(".", "") not in have:
            kept.append(Diagnosis(pointer=len(kept) + 1, code=dx.code, description=None))
            actions.append(VerifierAction(field="diagnoses", action="recovered_from_source", source_value=dx.code,
                                          detail="Diagnosis written in source but missing from LLM output."))
    packet.diagnoses = kept
    status["diagnoses"] = FieldStatus.PROVIDED if kept else FieldStatus.MISSING


def _verify_service_lines(packet: ClaimPacket, explicit: ClaimData, text: str,
                          actions: list[VerifierAction], status: dict[str, FieldStatus]) -> None:
    kept: list[ServiceLine] = []
    for line in packet.service_lines:
        if value_in_text(line.procedure_code, text, "code"):
            kept.append(line)
        else:
            actions.append(VerifierAction(field="service_lines", action="removed_unsupported",
                                          llm_value=line.procedure_code,
                                          detail="Procedure code not written in the source text."))

    unmatched = list(kept)
    for src in explicit.service_lines:
        match = next((l for l in unmatched if l.procedure_code == src.procedure_code), None)
        if match is None:
            kept.append(src.model_copy())
            actions.append(VerifierAction(field="service_lines", action="recovered_from_source",
                                          source_value=src.procedure_code,
                                          detail="Procedure line written in source but missing from LLM output."))
            continue
        unmatched.remove(match)
        prefix = f"service_lines[{src.procedure_code}]"
        for attr in ("units", "modifiers", "diagnosis_pointers"):
            src_val, llm_val = getattr(src, attr), getattr(match, attr)
            src_has = src_val not in (None, [])
            if src_has and src_val != llm_val:
                setattr(match, attr, src_val)
                actions.append(VerifierAction(field=f"{prefix}.{attr}",
                                              action="recovered_from_source" if llm_val in (None, []) else "conflict",
                                              llm_value=str(llm_val), source_value=str(src_val),
                                              detail=f"Used the explicitly labeled {attr.replace('_', ' ')} from the source."))
                if llm_val not in (None, []):
                    status[f"{prefix}.{attr}"] = FieldStatus.NEEDS_VERIFICATION
        if match.charge is None and src.charge is not None:
            match.charge = src.charge

    for i, line in enumerate(kept, start=1):
        line.line_number = i
        prefix = f"service_lines[{line.procedure_code}]"
        status.setdefault(f"{prefix}.units", FieldStatus.PROVIDED if line.units else FieldStatus.MISSING)
        status.setdefault(f"{prefix}.diagnosis_pointers",
                          FieldStatus.PROVIDED if line.diagnosis_pointers else FieldStatus.MISSING)
    packet.service_lines = kept
    status["service_lines"] = FieldStatus.PROVIDED if kept else FieldStatus.MISSING


@dataclass
class ExtractionResult:
    packet: ClaimPacket
    llm: LLMResult


def extract_claim(source_text: str, *, client: Any | None = None, model: str | None = None) -> ExtractionResult:
    """Run the LLM extraction and the source verifier. Raises `LLMError` if the
    OpenAI call fails (it never fabricates a result)."""
    if not source_text or not source_text.strip():
        raise ValueError("The visit summary is empty.")
    result = parse_structured(SYSTEM_PROMPT, source_text, ClaimData, client=client, model=model)
    packet = verify_against_source(result.parsed, source_text, model=result.model)
    log.info("Claim extracted: %d dx, %d lines, %d verifier actions",
             len(packet.diagnoses), len(packet.service_lines), len(packet.verifier_actions))
    return ExtractionResult(packet=packet, llm=result)
