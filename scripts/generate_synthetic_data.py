"""Generate the labeled SYNTHETIC evaluation set.

Writes:
  data/synthetic_claims/claim_XX.txt     visit summaries / superbills
  data/synthetic_claims/ground_truth.json expected extraction + seeded defects
  data/synthetic_denials/denial_XX.txt   remittance texts
  data/synthetic_denials/ground_truth.json expected CARC/RARC codes

Everything is fictional: patient names are "SYNTHETIC PATIENT X", payers are
invented, NPIs are computed numbers that pass the check digit but are not
assigned to anyone we know of. Codes are real public code-set values used
only as examples.

The set is deterministic (no randomness) so results are reproducible:
    python scripts/generate_synthetic_data.py
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLAIMS_DIR = ROOT / "data" / "synthetic_claims"
DENIALS_DIR = ROOT / "data" / "synthetic_denials"
BANNER = "*** SYNTHETIC DATA - FOR DEMONSTRATION ONLY - NOT A REAL PATIENT OR PROVIDER ***"


def npi(base9: str) -> str:
    """Append the CMS NPI check digit (Luhn over '80840' + 9 digits)."""
    total = 0
    for i, ch in enumerate(reversed("80840" + base9)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return base9 + str((10 - total % 10) % 10)


PAYERS = [
    ("ExamplePayer PPO", "PPO", "EP"),
    ("DemoHealth HMO", "HMO", "DH"),
    ("SampleCare Medicare Advantage", "Medicare Advantage", "SC"),
    ("TestShield EPO", "EPO", "TS"),
]
CLINICS = [
    ("Sample Family Medicine Group", npi("124531960")),
    ("Example Internal Medicine Associates", npi("198765432")),
    ("Demo Urgent & Primary Care", npi("156789012")),
]
CLINICIANS = [
    ("Dr. Alex Example", npi("172839405")),
    ("Dr. Jordan Sample", npi("143256789")),
    ("Dr. Riley Demo", npi("161234500")),
    ("Dr. Casey Placeholder", npi("188877766")),
]
ORDERERS = [("Dr. Taylor Orderly", npi("135792468")), ("Dr. Sam Referral", npi("124681357"))]

# name: (diagnoses [(code, description)], lines [(code, description, modifiers, pointers, charge)], needs)
SCENARIOS = {
    "strep": ([("J02.9", "Acute pharyngitis, unspecified"), ("R50.9", "Fever, unspecified")],
              [("99213", "Office visit, established patient", [], [1, 2], 145.00),
               ("87880", "Rapid strep A antigen test", [], [1], 35.00)], {}),
    "cough_xray": ([("R05.9", "Cough, unspecified")],
                   [("99214", "Office visit, established patient, moderate", [], [1], 190.00),
                    ("71046", "Chest X-ray, 2 views", [], [1], 120.00)], {"ordering": True}),
    "knee_injection": ([("M17.11", "Unilateral primary osteoarthritis, right knee")],
                       [("99213", "Office visit, established patient", ["25"], [1], 145.00),
                        ("20610", "Major joint injection/aspiration, right knee", ["RT"], [1], 160.00)], {}),
    "diabetes": ([("E11.9", "Type 2 diabetes mellitus without complications"), ("I10", "Essential (primary) hypertension")],
                 [("99214", "Office visit, established patient, moderate", [], [1, 2], 190.00),
                  ("83036", "Hemoglobin A1c test", [], [1], 28.00)], {"ordering": True}),
    "hypertension": ([("I10", "Essential (primary) hypertension")],
                     [("99213", "Office visit, established patient", [], [1], 145.00)], {}),
    "headache_mri": ([("R51.9", "Headache, unspecified")],
                     [("70553", "MRI brain without and with contrast", [], [1], 1450.00)],
                     {"ordering": True, "prior_auth": True}),
    "skin_biopsy": ([("D48.5", "Neoplasm of uncertain behavior of skin")],
                    [("99213", "Office visit, established patient", ["25"], [1], 145.00),
                     ("11102", "Tangential biopsy of skin, single lesion", [], [1], 210.00)], {}),
    "flu_vaccine": ([("Z23", "Encounter for immunization")],
                    [("90686", "Influenza vaccine, quadrivalent, preservative free", [], [1], 40.00),
                     ("G0008", "Administration of influenza vaccine", [], [1], 25.00)], {}),
    "uti": ([("N39.0", "Urinary tract infection, site not specified")],
            [("99212", "Office visit, established patient, straightforward", [], [1], 95.00),
             ("81002", "Urinalysis, non-automated, without microscopy", [], [1], 12.00)], {"ordering": True}),
    "ankle_xray": ([("S93.401A", "Sprain of unspecified ligament of right ankle, initial encounter")],
                   [("99203", "Office visit, new patient, low", [], [1], 160.00),
                    ("73610", "X-ray ankle, 3 views", ["RT"], [1], 95.00)], {"ordering": True}),
    "back_pain": ([("M54.50", "Low back pain, unspecified")],
                  [("99203", "Office visit, new patient, low", [], [1], 160.00)], {}),
    "telehealth_anxiety": ([("F41.1", "Generalized anxiety disorder")],
                           [("99213", "Video visit, established patient", ["95"], [1], 145.00)], {"pos": "10"}),
}

# Each defect: how it changes the claim, and which rule (if any) should catch it.
DEFECT_RULES = {
    "missing_member_id": "REQ_MEMBER_ID",
    "missing_dob": "REQ_PATIENT_DOB",
    "missing_rendering_npi": "REQ_RENDERING_NPI",
    "bad_npi_check_digit": "FMT_NPI_CHECK_DIGIT",
    "missing_pos": "REQ_PLACE_OF_SERVICE",
    "missing_ordering_provider": "ORDERING_PROVIDER_REQUIRED",
    "missing_dx_pointer": "REQ_DX_POINTER",
    "invalid_dx_pointer": "FMT_DX_POINTER_INVALID",
    "missing_prior_auth": "PRIOR_AUTH_REQUIRED",
    "missing_modifier_25": "MODIFIER_25_EM_WITH_PROCEDURE",
    "excess_units": "MAX_UNITS",
    "telehealth_missing_95": "TELEHEALTH_MODIFIER",
    # Realistic defects that format/completeness rules CANNOT see:
    "dx_does_not_support_procedure": None,   # valid code + valid pointer, clinically wrong pairing
    "member_id_typo": None,                  # well-formed ID that does not match the payer's records
    "service_not_documented": None,          # billed line with no support in the note
}

# (claim number, scenario, template, payer idx, clinic idx, clinician idx, defects)
PLAN = [
    (1, "strep", "superbill", 0, 0, 0, []),            # = Example A layout
    (2, "cough_xray", "narrative", 1, 2, 3, ["missing_rendering_npi", "missing_pos", "missing_ordering_provider"]),  # ~ Example B
    (3, "hypertension", "superbill", 2, 1, 1, []),
    (4, "diabetes", "soap", 0, 1, 1, []),
    (5, "knee_injection", "export", 3, 0, 2, []),
    (6, "flu_vaccine", "superbill", 2, 2, 3, []),
    (7, "back_pain", "narrative", 1, 0, 0, []),
    (8, "telehealth_anxiety", "soap", 0, 1, 2, []),
    (9, "strep", "export", 1, 2, 3, ["missing_member_id"]),
    (10, "uti", "superbill", 3, 2, 0, ["missing_ordering_provider"]),
    (11, "headache_mri", "narrative", 0, 1, 1, ["missing_prior_auth"]),
    (12, "skin_biopsy", "soap", 2, 0, 2, ["missing_modifier_25"]),
    (13, "hypertension", "export", 3, 1, 1, ["bad_npi_check_digit"]),
    (14, "ankle_xray", "superbill", 1, 2, 3, ["missing_pos", "missing_dob"]),
    (15, "diabetes", "narrative", 2, 1, 0, ["missing_dx_pointer"]),
    (16, "back_pain", "soap", 0, 0, 2, ["excess_units"]),
    (17, "telehealth_anxiety", "export", 1, 1, 1, ["telehealth_missing_95"]),
    (18, "strep", "soap", 3, 0, 0, ["invalid_dx_pointer"]),
    (19, "cough_xray", "superbill", 2, 2, 2, ["dx_does_not_support_procedure"]),
    (20, "knee_injection", "narrative", 0, 0, 3, ["member_id_typo"]),
    (21, "flu_vaccine", "export", 1, 2, 1, ["service_not_documented"]),
    (22, "uti", "soap", 0, 2, 3, ["missing_rendering_npi", "member_id_typo"]),
    (23, "headache_mri", "superbill", 3, 1, 0, []),
    (24, "skin_biopsy", "narrative", 1, 0, 1, ["missing_member_id", "dx_does_not_support_procedure"]),
    # Unlabeled free-text notes: values are embedded in prose (hardest for regex).
    (25, "strep", "freetext", 1, 2, 2, []),
    (26, "diabetes", "freetext", 3, 1, 3, []),
    (27, "ankle_xray", "freetext", 0, 2, 1, ["missing_ordering_provider"]),
    (28, "hypertension", "freetext", 2, 0, 0, ["member_id_typo"]),
]

DOBS = ["1987-04-12", "1975-04-03", "1990-11-23", "1962-07-08", "1948-01-30", "2001-09-15", "1983-03-02", "1956-12-19"]
SEXES = ["F", "M"]
MONTH = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
         "November", "December"]


def build_claim(n, scen, payer_i, clinic_i, clin_i) -> dict:
    dxs, lines, needs = SCENARIOS[scen]
    payer, plan, prefix = PAYERS[payer_i]
    clinic, clinic_npi = CLINICS[clinic_i]
    clin, clin_npi = CLINICIANS[clin_i]
    month = 1 + (n % 9)
    day = 3 + (n * 7) % 24
    claim = {
        "patient": {"full_name": f"SYNTHETIC PATIENT {chr(64 + n) if n <= 26 else n}",
                    "date_of_birth": DOBS[n % len(DOBS)], "sex": SEXES[n % 2]},
        "insurance": {"payer_name": payer, "plan_type": plan, "member_id": f"{prefix}-{(48151623 + n * 7919) % 100000000:08d}",
                      "group_number": None},
        "billing_provider": {"name": clinic, "npi": clinic_npi},
        "rendering_provider": {"name": clin, "npi": clin_npi},
        "ordering_provider": {"name": None, "npi": None},
        "service_date": f"2026-{month:02d}-{day:02d}",
        "place_of_service": needs.get("pos", "11"),
        "prior_authorization_number": f"PA-2026-{1000 + n}" if needs.get("prior_auth") else None,
        "diagnoses": [{"pointer": i, "code": c, "description": d} for i, (c, d) in enumerate(dxs, start=1)],
        "service_lines": [
            {"line_number": i, "procedure_code": c, "description": d, "modifiers": list(m), "units": 1,
             "diagnosis_pointers": list(p), "charge": ch}
            for i, (c, d, m, p, ch) in enumerate(lines, start=1)
        ],
        "needs_verification": [],
    }
    if needs.get("ordering"):
        o = ORDERERS[n % 2]
        claim["ordering_provider"] = {"name": o[0], "npi": o[1]}
    if n == 1:  # Example A exact values from the project brief
        claim["patient"].update(full_name="SYNTHETIC PATIENT A", date_of_birth="1987-04-12", sex="F")
        claim["insurance"]["member_id"] = "EP-78451233"
        claim["billing_provider"]["npi"] = "1234567893"
        claim["rendering_provider"]["npi"] = "1234567893"
        claim["service_date"] = "2026-02-12"
    return claim


def apply_defects(claim: dict, defects: list[str]) -> tuple[dict, list[str]]:
    c = copy.deepcopy(claim)
    notes = []
    for d in defects:
        if d == "missing_member_id":
            c["insurance"]["member_id"] = None
        elif d == "missing_dob":
            c["patient"]["date_of_birth"] = None
        elif d == "missing_rendering_npi":
            c["rendering_provider"]["npi"] = None
        elif d == "bad_npi_check_digit":
            good = c["billing_provider"]["npi"]
            c["billing_provider"]["npi"] = good[:-1] + str((int(good[-1]) + 1) % 10)
        elif d == "missing_pos":
            c["place_of_service"] = None
        elif d == "missing_ordering_provider":
            c["ordering_provider"] = {"name": None, "npi": None}
        elif d == "missing_dx_pointer":
            c["service_lines"][-1]["diagnosis_pointers"] = []
        elif d == "invalid_dx_pointer":
            c["service_lines"][-1]["diagnosis_pointers"] = [3]
        elif d == "missing_prior_auth":
            c["prior_authorization_number"] = None
        elif d == "missing_modifier_25":
            c["service_lines"][0]["modifiers"] = []
        elif d == "excess_units":
            c["service_lines"][0]["units"] = 2
        elif d == "telehealth_missing_95":
            c["service_lines"][0]["modifiers"] = []
        elif d == "dx_does_not_support_procedure":
            c["diagnoses"] = [{"pointer": 1, "code": "Z23", "description": "Encounter for immunization"}]
            notes.append("Diagnosis Z23 does not clinically support the billed services.")
        elif d == "member_id_typo":
            mid = c["insurance"]["member_id"]
            c["insurance"]["member_id"] = mid[:-2] + mid[-1] + mid[-2]
            notes.append("Member ID digits transposed vs. the payer's eligibility record.")
        elif d == "service_not_documented":
            c["service_lines"].append({"line_number": len(c["service_lines"]) + 1, "procedure_code": "99213",
                                       "description": "Office visit, established patient", "modifiers": [],
                                       "units": 1, "diagnosis_pointers": [1], "charge": 145.00})
            notes.append("E/M line billed but the note documents only a nurse vaccine visit.")
    return c, notes


# ---------------------------------------------------------------------------
# Text templates (four different layouts so the LLM sees varied input)
# ---------------------------------------------------------------------------
def _date_long(iso: str) -> str:
    y, m, d = iso.split("-")
    return f"{MONTH[int(m) - 1]} {int(d)}, {y}"


def _date_us(iso: str) -> str:
    y, m, d = iso.split("-")
    return f"{m}/{d}/{y}"


def _mods(l):
    return ", ".join(l["modifiers"]) if l["modifiers"] else "none"


def _ptrs(l):
    return ", ".join(map(str, l["diagnosis_pointers"])) if l["diagnosis_pointers"] else "(not recorded)"


def t_superbill(c: dict, note: str) -> str:
    p, ins, b, r, o = c["patient"], c["insurance"], c["billing_provider"], c["rendering_provider"], c["ordering_provider"]
    out = [BANNER, "", "SUPERBILL / ENCOUNTER FORM", "", "PROVIDERS",
           f"Billing Provider: {b['name']}", f"Billing Provider NPI: {b['npi'] or 'not recorded'}",
           f"Rendering Provider: {r['name']}", f"Rendering Provider NPI: {r['npi'] or 'not recorded'}"]
    if o["name"]:
        out += [f"Ordering Provider: {o['name']}", f"Ordering Provider NPI: {o['npi']}"]
    out += ["", "PATIENT", f"Patient Name: {p['full_name']}", f"DOB: {p['date_of_birth'] or 'not recorded'}",
            f"Sex: {p['sex']}", f"Insurance: {ins['payer_name']}", f"Member ID: {ins['member_id'] or 'not recorded'}"]
    if c["prior_authorization_number"]:
        out.append(f"Prior Authorization #: {c['prior_authorization_number']}")
    out += ["", "ENCOUNTER", f"Date of Service: {c['service_date']}",
            f"Place of Service: {c['place_of_service'] or 'not recorded'}", "", "DIAGNOSES (ICD-10-CM)"]
    out += [f"{d['pointer']}. {d['code']} - {d['description']}" for d in c["diagnoses"]]
    out += ["", "PROCEDURES"]
    for l in c["service_lines"]:
        out.append(f"Line {l['line_number']}: CPT {l['procedure_code']} | {l['description']} | Units: {l['units']} | "
                   f"Modifiers: {_mods(l)} | Dx Pointers: {_ptrs(l)} | Charge: ${l['charge']:.2f}")
    out += ["", "CLINICAL NOTE", note]
    return "\n".join(out) + "\n"


def t_narrative(c: dict, note: str) -> str:
    p, ins, b, r, o = c["patient"], c["insurance"], c["billing_provider"], c["rendering_provider"], c["ordering_provider"]
    out = [BANNER, "", f"VISIT SUMMARY - {b['name']}", "",
           f"Patient: {p['full_name']}"]
    if p["date_of_birth"]:
        out.append(f"DOB: {_date_us(p['date_of_birth'])}")
    out += [f"Sex: {p['sex']}", f"Insurance: {ins['payer_name']}"]
    if ins["member_id"]:
        out.append(f"Member ID: {ins['member_id']}")
    out += ["", f"Billing Provider: {b['name']}, NPI {b['npi']}"]
    out.append(f"Rendering Provider: {r['name']}" + (f", NPI {r['npi']}" if r["npi"] else " (NPI not recorded on this summary)"))
    out.append(f"Visit date: {_date_long(c['service_date'])}")
    if c["place_of_service"]:
        out.append(f"Place of service: {c['place_of_service']}")
    if c["prior_authorization_number"]:
        out.append(f"Authorization number: {c['prior_authorization_number']}")
    out += ["", f"History and exam: {note}", ""]
    dx = "; ".join(f"{d['description']} ({d['code']})" for d in c["diagnoses"])
    out.append(f"Assessment: {dx}.")
    plan = []
    for l in c["service_lines"]:
        mod = f" with modifier {', '.join(l['modifiers'])}" if l["modifiers"] else ""
        ptr = (f", linked to diagnosis {', '.join(map(str, l['diagnosis_pointers']))}" if l["diagnosis_pointers"] else "")
        plan.append(f"{l['description']} - CPT {l['procedure_code']}{mod}, {l['units']} unit{'s' if l['units'] != 1 else ''}{ptr}. Charge ${l['charge']:.2f}.")
    out.append("Plan / services: " + " ".join(plan))
    out.append(f"Ordering provider: {o['name']}, NPI {o['npi']}" if o["name"] else "Ordering provider: not documented.")
    return "\n".join(out) + "\n"


def t_soap(c: dict, note: str) -> str:
    p, ins, b, r, o = c["patient"], c["insurance"], c["billing_provider"], c["rendering_provider"], c["ordering_provider"]
    out = [BANNER, "", "PROGRESS NOTE (SOAP)", f"Clinic: {b['name']} | Billing NPI: {b['npi']}",
           f"Rendering provider: {r['name']} | Rendering NPI: {r['npi'] or 'not on file'}",
           f"Patient: {p['full_name']} | DOB: {p['date_of_birth'] or 'unknown'} | Sex: {p['sex']}",
           f"Payer: {ins['payer_name']} | Member ID: {ins['member_id'] or 'not on file'}",
           f"DOS: {_date_us(c['service_date'])} | POS: {c['place_of_service'] or 'not recorded'}"]
    if o["name"]:
        out.append(f"Ordering provider: {o['name']} | Ordering NPI: {o['npi']}")
    if c["prior_authorization_number"]:
        out.append(f"Prior auth #: {c['prior_authorization_number']}")
    out += ["", f"S/O: {note}", "", "A (diagnoses):"]
    out += [f"  Dx {d['pointer']}: {d['code']} {d['description']}" for d in c["diagnoses"]]
    out += ["", "P (services billed):"]
    for l in c["service_lines"]:
        code = l["procedure_code"] + ("-" + "-".join(l["modifiers"]) if l["modifiers"] else "")
        out.append(f"  CPT {code} {l['description']}, units {l['units']}, dx pointer {_ptrs(l)}, ${l['charge']:.2f}")
    return "\n".join(out) + "\n"


def t_export(c: dict, note: str) -> str:
    p, ins, b, r, o = c["patient"], c["insurance"], c["billing_provider"], c["rendering_provider"], c["ordering_provider"]
    kv = [("PATIENT NAME", p["full_name"]), ("DOB", p["date_of_birth"]), ("SEX", p["sex"]),
          ("PAYER", ins["payer_name"]), ("MEMBER ID", ins["member_id"]),
          ("BILLING PROVIDER", b["name"]), ("BILLING NPI", b["npi"]),
          ("RENDERING PROVIDER", r["name"]), ("RENDERING NPI", r["npi"]),
          ("ORDERING PROVIDER", o["name"]), ("ORDERING NPI", o["npi"]),
          ("DATE OF SERVICE", c["service_date"]), ("PLACE OF SERVICE", c["place_of_service"]),
          ("PRIOR AUTH", c["prior_authorization_number"])]
    out = [BANNER, "", "EHR CHARGE EXPORT (synthetic)"]
    out += [f"{k}: {v if v else ''}" for k, v in kv]
    out += ["", "DX# | ICD-10-CM | DESCRIPTION"]
    out += [f"{d['pointer']} | {d['code']} | {d['description']}" for d in c["diagnoses"]]
    out += ["", "LINE | CPT/HCPCS | MOD | UNITS | DX PTR | CHARGE"]
    for l in c["service_lines"]:
        out.append(f"{l['line_number']} | HCPCS {l['procedure_code']} | mod: {' '.join(l['modifiers']) or '-'} | "
                   f"units: {l['units']} | dx pointer: {','.join(map(str, l['diagnosis_pointers'])) or '-'} | ${l['charge']:.2f}")
    out += ["", f"NOTE: {note}"]
    return "\n".join(out) + "\n"


WORDS = {1: "one", 2: "two", 3: "three"}


def t_freetext(c: dict, note: str) -> str:
    p, ins, b, r, o = c["patient"], c["insurance"], c["billing_provider"], c["rendering_provider"], c["ordering_provider"]
    sex = {"F": "female", "M": "male"}[p["sex"]]
    s = [BANNER, ""]
    born = f", born {_date_long(p['date_of_birth'])}," if p["date_of_birth"] else ""
    s.append(f"{p['full_name']} ({sex}){born} was seen on {_date_long(c['service_date'])} by {r['name']}"
             + (f" (individual NPI {r['npi']})" if r["npi"] else "")
             + f" at {b['name']}, which bills under group NPI {b['npi']}. The visit took place in the clinic office,"
             f" place of service {c['place_of_service']}.")
    cov = f"Coverage is through {ins['payer_name']}"
    cov += f"; the card lists subscriber number {ins['member_id']}." if ins["member_id"] else "; the card was not scanned."
    s.append(cov)
    if c["prior_authorization_number"]:
        s.append(f"The plan approved the service under authorization {c['prior_authorization_number']}.")
    s.append(note)
    dx = " and ".join(f"{d['description'].lower()} ({d['code']})" for d in c["diagnoses"])
    s.append(f"Conditions addressed today, in order: {dx}.")
    for l in c["service_lines"]:
        ptr = l["diagnosis_pointers"]
        which = ("all listed conditions" if len(ptr) == len(c["diagnoses"]) and len(ptr) > 1
                 else " and ".join(f"condition #{x}" for x in ptr))
        mod = f", modifier {' and '.join(l['modifiers'])}" if l["modifiers"] else ""
        s.append(f"We billed {l['procedure_code']} ({l['description'].lower()}{mod}), {WORDS.get(l['units'], l['units'])} "
                 f"unit{'s' if l['units'] != 1 else ''}, supporting {which}, at ${l['charge']:.2f}.")
    if o["name"]:
        s.append(f"The study was ordered by {o['name']}, NPI {o['npi']}.")
    return "\n".join(s[:2]) + "\n".join([""] + [" ".join(s[2:])]) + "\n"


TEMPLATES = {"superbill": t_superbill, "narrative": t_narrative, "soap": t_soap, "export": t_export,
             "freetext": t_freetext}
NOTES = {
    "strep": "Sore throat and fever for 3 days. Tonsillar exudate. Rapid strep A positive in office.",
    "cough_xray": "Dry cough for three weeks, no fever, scattered wheezes. Chest X-ray ordered.",
    "knee_injection": "Chronic right knee pain from osteoarthritis. Separate evaluation of new hip complaint. Corticosteroid injection right knee.",
    "diabetes": "Follow-up of type 2 diabetes and hypertension. A1c drawn today. Medications reviewed.",
    "hypertension": "Blood pressure follow-up, home readings reviewed, medication continued.",
    "headache_mri": "Recurrent headaches with new neurologic symptom; MRI brain with and without contrast performed.",
    "skin_biopsy": "Changing pigmented lesion on back; full skin exam; tangential biopsy performed.",
    "flu_vaccine": "Nurse visit for seasonal influenza vaccination. No other services.",
    "uti": "Dysuria and urinary frequency for two days. Dipstick urinalysis performed in office.",
    "ankle_xray": "Inversion injury of right ankle yesterday. Ankle X-ray, 3 views.",
    "back_pain": "New patient with low back pain after lifting, no red flags. Home exercise plan.",
    "telehealth_anxiety": "Video visit from patient's home for anxiety follow-up. Medication adjusted.",
}

# Guidance passage(s) a good retriever should surface for each denial
DENIAL_KB = {1: ["KB-ORDERING-PROVIDER", "KB-RENDERING-NPI"], 2: ["KB-PRIOR-AUTH"], 3: ["KB-BUNDLING"],
             4: ["KB-ELIGIBILITY"], 5: ["KB-BILLING-NPI", "KB-CORRECTED-VS-APPEAL"], 6: ["KB-DIAGNOSIS-MISMATCH"],
             7: ["KB-CORRECTED-VS-APPEAL"], 8: [], 9: ["KB-ORDERING-PROVIDER"], 10: ["KB-MEDICAL-NECESSITY"]}

# Denials: (number, claim number, text style, codes [(type, group, code)], denied amount)
DENIALS = [
    (1, 2, "table", [("CARC", "CO", "16"), ("RARC", None, "N265"), ("RARC", None, "N290")], 120.00),
    (2, 11, "table", [("CARC", "CO", "197")], 1450.00),
    (3, 12, "letter", [("CARC", "CO", "97"), ("RARC", None, "N20")], 145.00),
    (4, 9, "table", [("CARC", "CO", "31")], 180.00),
    (5, 13, "line", [("CARC", "CO", "16"), ("RARC", None, "N257"), ("RARC", None, "MA130")], 145.00),
    (6, 19, "letter", [("CARC", "CO", "11")], 310.00),
    (7, 14, "line", [("CARC", "CO", "16"), ("RARC", None, "M77")], 255.00),
    (8, 16, "table", [("CARC", "CO", "45"), ("CARC", "PR", "2")], 60.00),
    (9, 10, "line", [("CARC", "CO", "16"), ("RARC", None, "N264"), ("RARC", None, "N265")], 12.00),
    (10, 24, "letter", [("CARC", "CO", "50")], 355.00),
]


def denial_text(n, claim, style, codes, amount) -> str:
    payer = claim["insurance"]["payer_name"]
    head = [BANNER.replace("PATIENT OR PROVIDER", "REMITTANCE"), "", f"Payer: {payer}",
            f"Claim #: SYN-CLM-{claim['_n']:04d}   Patient: {claim['patient']['full_name']}",
            f"Date of Service: {claim['service_date']}", ""]
    carcs = [(g, c) for t, g, c in codes if t == "CARC"]
    rarcs = [c for t, g, c in codes if t == "RARC"]
    if style == "table":
        body = [f"LINE 1  {claim['service_lines'][0]['procedure_code']}  BILLED ${amount:.2f}  PAID $0.00"]
        body += [f"        ADJ {g}-{c}  ${amount:.2f}" for g, c in carcs]
        if rarcs:
            body.append("        REMARKS: " + ", ".join(rarcs))
    elif style == "line":
        body = ["Status: DENIED. Adjustment " + " / ".join(f"{g} {c}" for g, c in carcs)
                + (f"; remark codes {' and '.join(rarcs)}" if rarcs else "") + f". Amount ${amount:.2f}."]
    else:
        body = ["Dear Provider,", "",
                f"We reviewed the claim above. The service was adjusted with reason code(s) "
                + ", ".join(f"{g}-{c}" for g, c in carcs)
                + (f" and remark code(s) {', '.join(rarcs)}" if rarcs else "")
                + f". The denied amount is ${amount:.2f}. You may request reconsideration within the time frame in your agreement.",
                "", "Sincerely,", "Claims Department (synthetic)"]
    return "\n".join(head + body) + "\n"


def main() -> None:
    CLAIMS_DIR.mkdir(parents=True, exist_ok=True)
    DENIALS_DIR.mkdir(parents=True, exist_ok=True)
    truth = []
    claims_by_n = {}
    for n, scen, tmpl, pi, ci, ri, defects in PLAN:
        base = build_claim(n, scen, pi, ci, ri)
        claim, notes = apply_defects(base, defects)
        claim["_n"] = n
        claims_by_n[n] = claim
        text = TEMPLATES[tmpl](claim, NOTES[scen])
        fname = f"claim_{n:02d}.txt"
        (CLAIMS_DIR / fname).write_text(text, encoding="utf-8")
        expected = {k: v for k, v in claim.items() if not k.startswith("_")}
        truth.append({
            "claim_id": f"SYN-CLM-{n:04d}", "file": fname, "scenario": scen, "template": tmpl,
            "expected_claim": expected,
            "seeded_defects": [{"defect": d, "expected_rule": DEFECT_RULES[d], "detectable_by_rules": DEFECT_RULES[d] is not None}
                               for d in defects],
            "defect_notes": notes,
        })
    (CLAIMS_DIR / "ground_truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")

    dtruth = []
    for n, cn, style, codes, amount in DENIALS:
        claim = claims_by_n[cn]
        fname = f"denial_{n:02d}.txt"
        (DENIALS_DIR / fname).write_text(denial_text(n, claim, style, codes, amount), encoding="utf-8")
        dtruth.append({"denial_id": f"SYN-DEN-{n:04d}", "file": fname, "claim_id": f"SYN-CLM-{cn:04d}",
                       "payer_name": claim["insurance"]["payer_name"], "denied_amount": amount,
                       "expected_codes": [{"code_type": t, "group_code": g, "code": c} for t, g, c in codes],
                       "expected_guidance": DENIAL_KB[n]})
    (DENIALS_DIR / "ground_truth.json").write_text(json.dumps(dtruth, indent=2), encoding="utf-8")
    print(f"Wrote {len(truth)} claims and {len(dtruth)} denials.")


if __name__ == "__main__":
    main()
