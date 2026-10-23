"""Deterministic reading of *explicitly labeled* values in claim text.

This is NOT a replacement for the LLM. Free-text visit summaries are too
varied for regular expressions. It has two jobs:

1. Give the verifier a second opinion on values that are clearly labeled
   ("Member ID: EP-78451233", "Billing Provider NPI: 1234567890", "CPT 99213").
2. Check whether a value the LLM returned actually appears in the source text.
"""

from __future__ import annotations

import re
from datetime import date, datetime

from .schemas import ClaimData, Diagnosis, Insurance, Patient, Provider, ServiceLine

NULL_WORDS = re.compile(
    r"^(none|n/?a|unknown|not (documented|recorded|provided|listed|available|on file)|missing|pending|tbd|-+)\b",
    re.IGNORECASE,
)

MONTHS = {m.lower(): i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"], start=1)}

_DATE_PATTERNS = [
    (re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"), "ymd"),
    (re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"), "mdy"),
    (re.compile(r"\b(\d{1,2})-(\d{1,2})-(\d{4})\b"), "mdy"),
    (re.compile(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+(\d{4})\b"), "Mdy"),
    (re.compile(r"\b(\d{1,2})\s+([A-Za-z]{3,9})\.?\s+(\d{4})\b"), "dMy"),
]

ICD10_RE = re.compile(r"\b([A-TV-Z][0-9][0-9AB](?:\.[0-9A-TV-Z]{1,4})?)\b")
PROC_RE = re.compile(r"\b(\d{4}[0-9FTU]|[A-V]\d{4})\b")
NPI_RE = re.compile(r"(?<![\d-])(\d{10})(?![\d-])")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip().strip(".,;:").strip()
    if not value or NULL_WORDS.match(value):
        return None
    return value


def _month(word: str) -> int | None:
    word = word.lower().rstrip(".")
    for name, num in MONTHS.items():
        if len(word) >= 3 and name.startswith(word):
            return num
    return None


def _to_iso(groups: tuple[str, str, str], kind: str) -> str | None:
    try:
        if kind == "ymd":
            y, m, d = int(groups[0]), int(groups[1]), int(groups[2])
        elif kind == "mdy":
            m, d, y = int(groups[0]), int(groups[1]), int(groups[2])
        elif kind == "Mdy":
            m = _month(groups[0])
            if m is None:
                return None
            d, y = int(groups[1]), int(groups[2])
        else:  # dMy
            m = _month(groups[1])
            if m is None:
                return None
            d, y = int(groups[0]), int(groups[2])
        return date(y, m, d).isoformat()
    except ValueError:
        return None


def find_dates(text: str) -> list[str]:
    """All dates in the text, normalized to YYYY-MM-DD, in order of appearance."""
    found: list[tuple[int, str]] = []
    for pattern, kind in _DATE_PATTERNS:
        for m in pattern.finditer(text):
            iso = _to_iso(m.groups(), kind)
            if iso:
                found.append((m.start(), iso))
    return [iso for _, iso in sorted(found)]


def normalize_date(value: str | None) -> str | None:
    if not value:
        return None
    dates = find_dates(value)
    if dates:
        return dates[0]
    try:
        return datetime.fromisoformat(value).date().isoformat()
    except ValueError:
        return None


def _first_date_after(label: str, text: str) -> str | None:
    m = re.search(label + r"\s*[:\-]?\s*(.{0,40})", text, re.IGNORECASE)
    if not m:
        return None
    return normalize_date(m.group(1))


def _labeled(label: str, text: str, value: str = r"[^\n]+") -> str | None:
    m = re.search(r"(?:^|[ \t|,;])" + label + r"[ \t]*[:#][ \t]*(" + value + ")", text, re.IGNORECASE | re.MULTILINE)
    return _clean(m.group(1)) if m else None


def _cut_name(raw: str | None) -> str | None:
    """Trim a provider/payer name at the next field on the same line."""
    if not raw:
        return None
    raw = re.split(r"\s{2,}|\(|,\s*NPI|\bNPI\b|\|", raw, maxsplit=1)[0]
    return _clean(raw)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
ROLE_WORDS = {
    "billing_provider": r"billing",
    "rendering_provider": r"rendering|performing|attending",
    "ordering_provider": r"ordering|referring",
}


def _provider_for_role(role_regex: str, lines: list[str]) -> Provider:
    name = npi = None
    for line in lines:
        if not re.search(r"\b(" + role_regex + r")\b", line, re.IGNORECASE):
            continue
        # A line may mention several roles ("Billing/Rendering NPI"); only use
        # the part of the line that starts at this role's keyword.
        start = re.search(r"\b(" + role_regex + r")\b", line, re.IGNORECASE).start()
        segment = line[start:]
        other = [r for r in ROLE_WORDS.values() if r != role_regex]
        nxt = re.search(r"\b(" + "|".join(other) + r")\b", segment[1:], re.IGNORECASE)
        if nxt:
            segment = segment[: nxt.start() + 1]
        if npi is None:
            m = re.search(r"NPI\s*[:#]?\s*(\d{10})\b", segment, re.IGNORECASE) or (
                NPI_RE.search(segment) if "npi" in segment.lower() else None
            )
            if m:
                npi = m.group(1)
        if name is None:
            m = re.search(r"(?:provider|physician|clinician)(?:\s+name)?\s*:\s*(.+)", segment, re.IGNORECASE)
            if m and not re.match(r"\s*NPI\b", m.group(1), re.IGNORECASE):
                name = _cut_name(m.group(1))
    return Provider(name=name, npi=npi)


def _diagnoses(lines: list[str]) -> list[Diagnosis]:
    codes: list[str] = []
    in_section = False
    for line in lines:
        low = line.lower()
        if re.search(r"diagnos|\bdx\b|assessment|icd", low):
            in_section = True
        elif re.search(r"procedure|cpt|hcpcs|plan:|services", low) and not ICD10_RE.search(line):
            in_section = False
        if in_section or re.search(r"diagnos|\bdx\b|assessment|icd", low):
            for code in ICD10_RE.findall(line):
                # Skip things that only look like ICD codes inside procedure lines
                if re.search(r"\b(cpt|hcpcs)\b", low) and not re.search(r"diagnos|\bdx\b|icd", low):
                    continue
                if code not in codes:
                    codes.append(code)
    return [Diagnosis(pointer=i, code=c, description=None) for i, c in enumerate(codes, start=1)]


def _service_lines(lines: list[str]) -> list[ServiceLine]:
    out: list[ServiceLine] = []
    for line in lines:
        if not re.search(r"\b(cpt|hcpcs|procedure code|proc)\b", line, re.IGNORECASE):
            continue
        for m in re.finditer(r"\b(?:CPT|HCPCS|proc(?:edure)?(?: code)?)\s*[:#]?\s*(?:code\s*)?" + PROC_RE.pattern,
                             line, re.IGNORECASE):
            code = m.group(1).upper()
            # Text belonging to this code runs until the next code on the line.
            rest = line[m.end():]
            nxt = re.search(r"\b(?:CPT|HCPCS)\b", rest, re.IGNORECASE)
            seg = rest[: nxt.start()] if nxt else rest
            mods: list[str] = []
            inline = re.match(r"-([0-9A-Z]{2}(?:-[0-9A-Z]{2})*)\b", rest)
            if inline:
                mods = inline.group(1).split("-")
            mm = re.search(r"(?i:modifiers?|mod)\s*[:=]?\s*((?:[0-9A-Z]{2}\b[\s,/&]*)+)", seg)
            if mm:
                mods = [x for x in re.split(r"[\s,/&]+", mm.group(1)) if x]
            units = None
            um = re.search(r"\b(?:units?|qty|quantity)\s*[:=]?\s*(\d+)\b|\b(\d+)\s+units?\b|\bx\s?(\d+)\b", seg, re.IGNORECASE)
            if um:
                units = int(next(g for g in um.groups() if g))
            ptrs: list[int] = []
            pm = re.search(
                r"(?:pointers?|dx ptr|linked to diagnos[ie]s|diagnosis)\s*[:=#]?\s*((?:[1-9A-L]\b(?:\s*(?:,|&|and|/)\s*)?)+)",
                seg, re.IGNORECASE,
            )
            if pm:
                for p in re.split(r"\s*(?:,|&|and|/)\s*|\s+", pm.group(1).strip(), flags=re.IGNORECASE):
                    p = p.strip().upper()
                    if p.isdigit():
                        ptrs.append(int(p))
                    elif p and p in "ABCDEFGHIJKL":
                        ptrs.append("ABCDEFGHIJKL".index(p) + 1)
            charge = None
            cm = re.search(r"\$\s?([\d,]+\.?\d*)", seg)
            if cm:
                charge = float(cm.group(1).replace(",", ""))
            out.append(ServiceLine(line_number=len(out) + 1, procedure_code=code, modifiers=mods,
                                   units=units, diagnosis_pointers=ptrs, charge=charge))
    return out


def parse_explicit(text: str) -> ClaimData:
    """Return only values that are explicitly labeled in the text."""
    lines = text.splitlines()

    patient = Patient(
        full_name=_cut_name(_labeled(r"patient(?:\s+name)?", text)),
        date_of_birth=_first_date_after(r"\b(?:DOB|D\.O\.B\.?|date of birth|birth ?date)", text),
        sex=_labeled(r"(?:sex|gender)", text, r"[A-Za-z]+"),
    )
    payer_raw = _labeled(r"(?:insurance|payer|insurer|health plan)(?:\s+name)?", text)
    payer = _clean(re.split(r"\s{2,}|,|;|\||\bmember\b|\bsubscriber\b|\bID\b", payer_raw, flags=re.IGNORECASE)[0]) if payer_raw else None
    plan_type = None
    if payer:
        pm = re.search(r"\b(PPO|HMO|EPO|POS plan|HDHP|Medicare Advantage|Medicaid|Medicare)\b", payer, re.IGNORECASE)
        plan_type = pm.group(1) if pm else None
    insurance = Insurance(
        payer_name=payer,
        plan_type=plan_type,
        member_id=_labeled(r"(?:member|subscriber|policy)\s*(?:id|#|number|no\.?)", text, r"[A-Z0-9][A-Z0-9\-]{3,}")
        or _labeled(r"(?:member|subscriber)", text, r"[A-Z0-9][A-Z0-9\-]{3,}"),
        group_number=_labeled(r"group\s*(?:#|number|no\.?)?", text, r"[A-Z0-9][A-Z0-9\-]{2,}"),
    )
    pos = None
    m = re.search(r"\b(?:place of service|POS)\b\s*(?:code)?\s*[:#\-]?\s*(\d{1,2})\b", text, re.IGNORECASE)
    if m:
        pos = m.group(1).zfill(2)
    prior_auth = _labeled(r"(?:prior auth(?:orization)?|authorization|auth)\s*(?:#|number|no\.?)?", text,
                          r"[A-Z0-9][A-Z0-9\-]{3,}")

    return ClaimData(
        patient=patient,
        insurance=insurance,
        billing_provider=_provider_for_role(ROLE_WORDS["billing_provider"], lines),
        rendering_provider=_provider_for_role(ROLE_WORDS["rendering_provider"], lines),
        ordering_provider=_provider_for_role(ROLE_WORDS["ordering_provider"], lines),
        service_date=_first_date_after(r"\b(?:date of service|DOS|service date|visit date|encounter date)", text),
        place_of_service=pos,
        prior_authorization_number=prior_auth,
        diagnoses=_diagnoses(lines),
        service_lines=_service_lines(lines),
    )


# ---------------------------------------------------------------------------
# "Does this value appear in the source?" checks
# ---------------------------------------------------------------------------
def _squash(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def value_in_text(value: str | None, text: str, kind: str = "text") -> bool:
    """True if `value` is supported by the source text.

    kind: "date" (any date format), "code" (ignore dots/case), "id"
    (ignore spaces/hyphens), "name" (all words present), "text".
    """
    if value is None or value == "":
        return True
    if kind == "date":
        iso = normalize_date(value)
        return iso is not None and iso in find_dates(text)
    if kind in ("code", "id"):
        sq = _squash(value)
        if not sq:
            return False
        tokens = {_squash(t) for t in re.split(r"[\s,;:()|/\-]+", text)}
        tokens |= {_squash(t) for t in re.split(r"[\s,;:()|/]+", text)}
        return sq in tokens or (kind == "id" and len(sq) >= 8 and sq in _squash(text))
    if kind == "name":
        words = [w for w in re.findall(r"[a-z0-9]+", value.lower()) if w not in {"dr", "md", "do", "np", "pa"}]
        low = text.lower()
        return all(re.search(r"\b" + re.escape(w) + r"\b", low) for w in words)
    return _squash(value) in _squash(text)
