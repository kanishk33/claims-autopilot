"""Feature 2: deterministic precheck rules and risk scoring."""

from __future__ import annotations

from datetime import date

import pytest

from claims_autopilot.evaluation import evaluate_extraction, evaluate_precheck, parser_only_extract
from claims_autopilot.questioner import checklist, follow_up_questions
from claims_autopilot.schemas import IssueCategory, RiskLevel, ServiceLine
from claims_autopilot.validator import load_rules, npi_check_digit_ok, validate_claim

from conftest import example_a_claim, example_b_claim

AS_OF = date(2026, 10, 23)


def rules_fired(claim):
    return {i.rule_id for i in validate_claim(claim, as_of=AS_OF).issues}


def test_example_a_is_low_risk():
    r = validate_claim(example_a_claim(), as_of=AS_OF)
    assert r.risk_level == RiskLevel.LOW
    assert r.errors == [] and r.risks == []
    assert {i.rule_id for i in r.issues} == {"CLIA_WAIVED_QW"}  # advisory only
    assert r.ready_to_submit


def test_example_b_flags_missing_information():
    r = validate_claim(example_b_claim(), as_of=AS_OF)
    by_rule = {i.rule_id: i for i in r.issues}
    assert by_rule["REQ_RENDERING_NPI"].category == IssueCategory.VALIDATION_ERROR
    assert by_rule["REQ_PLACE_OF_SERVICE"].category == IssueCategory.VALIDATION_ERROR
    ordering = by_rule["ORDERING_PROVIDER_REQUIRED"]
    assert ordering.category == IssueCategory.DENIAL_RISK  # a risk, not a certain denial
    assert ordering.demo_assumption and "71046" in ordering.message and ordering.why_it_matters
    assert r.risk_level == RiskLevel.HIGH and not r.ready_to_submit
    assert any("ordered" in q["question"] for q in follow_up_questions(r))
    assert len(checklist(r)) == len(r.issues)


def test_npi_check_digit():
    assert npi_check_digit_ok("1234567893")
    assert not npi_check_digit_ok("1234567890")   # the placeholder from the original brief
    assert not npi_check_digit_ok("12345")
    c = example_a_claim()
    c.billing_provider.npi = "1234567890"
    assert "FMT_NPI_CHECK_DIGIT" in rules_fired(c)


def test_line_level_rules():
    c = example_a_claim()
    c.service_lines[1].diagnosis_pointers = [3]
    c.service_lines[0].units = None
    assert {"FMT_DX_POINTER_INVALID", "REQ_UNITS"} <= rules_fired(c)
    c = example_a_claim()
    c.service_lines[0].units = 3
    c.service_lines.append(c.service_lines[1].model_copy())
    assert {"MAX_UNITS", "DUPLICATE_LINE"} <= rules_fired(c)


def test_modifier_25_and_telehealth_rules():
    c = example_a_claim()
    c.service_lines[1] = ServiceLine(line_number=2, procedure_code="20610", units=1, diagnosis_pointers=[1])
    assert "MODIFIER_25_EM_WITH_PROCEDURE" in rules_fired(c)
    c.service_lines[0].modifiers = ["25"]
    assert "MODIFIER_25_EM_WITH_PROCEDURE" not in rules_fired(c)
    t = example_a_claim()
    t.place_of_service = "10"
    assert "TELEHEALTH_MODIFIER" in rules_fired(t)


def test_dates_and_member_id():
    c = example_a_claim()
    c.service_date = "2027-01-01"
    assert "FMT_DOS_FUTURE" in rules_fired(c)
    c.service_date = "2025-01-02"
    assert "TIMELY_FILING" in rules_fired(c)
    c = example_a_claim()
    c.insurance.member_id = "ExamplePayer PPO EP-78451233"
    assert "FMT_MEMBER_ID_COMBINED" in rules_fired(c)


def test_every_rule_explains_itself():
    rules = load_rules()
    for r in rules["required_fields"] + rules["code_rules"]:
        assert r["why_it_matters"] and r["question"], r["id"]


def test_seeded_defects_are_caught_without_false_alarms():
    res = evaluate_precheck()
    assert res["claims"] >= 20
    assert res["defects_caught"] == res["defects_detectable_by_rule_design"]
    assert res["claims_with_unexpected_flags"] == []
    assert res["claim_error_rate_after_fixing_flags"] < res["claim_error_rate_without_precheck"]


def test_verifier_baseline_never_invents_values():
    res = evaluate_extraction(parser_only_extract, "baseline")
    assert res["claims_evaluated"] >= 20 and res["hallucinated_values"] == 0
