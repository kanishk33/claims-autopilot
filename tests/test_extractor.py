"""Feature 1: claim extraction, source verification, OpenAI error handling, packaging."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import openai
import pytest
from openai.lib._pydantic import to_strict_json_schema

from claims_autopilot import config
from claims_autopilot.extractor import extract_claim, verify_against_source
from claims_autopilot.llm import LLMError, check_api_access, translate_error
from claims_autopilot.schemas import ClaimData, FieldStatus

from conftest import FakeOpenAI, example_a_claim, example_b_claim

ROOT = Path(__file__).resolve().parents[1]


# --- Example A -----------------------------------------------------------
def test_example_a_preserves_every_value(text_a):
    result = extract_claim(text_a, client=FakeOpenAI(parsed=example_a_claim()), model="test-model")
    p = result.packet
    assert p.patient.full_name == "SYNTHETIC PATIENT A"
    assert p.patient.date_of_birth == "1987-04-12"
    assert p.insurance.payer_name == "ExamplePayer PPO"
    assert p.insurance.member_id == "EP-78451233"
    assert p.billing_provider.npi == "1234567893" and isinstance(p.billing_provider.npi, str)
    assert p.rendering_provider.npi == "1234567893"
    assert p.service_date == "2026-02-12"
    assert p.place_of_service == "11"
    assert [d.code for d in p.diagnoses] == ["J02.9", "R50.9"]
    assert [(l.procedure_code, l.units, l.diagnosis_pointers) for l in p.service_lines] == [
        ("99213", 1, [1, 2]), ("87880", 1, [1])]
    # A correct LLM answer is left alone.
    assert not [a for a in p.verifier_actions if a.action in ("conflict", "removed_unsupported")]
    assert p.field_status["insurance.member_id"] == FieldStatus.PROVIDED
    assert result.llm.model == "test-model"


def test_example_a_repairs_known_llm_failures(text_a):
    """Reproduces the failures from the previous attempt and checks they are fixed."""
    bad = example_a_claim().model_copy(deep=True)
    bad.insurance.payer_name = "ExamplePayer PPO EP-78451233"   # member ID glued to payer
    bad.insurance.member_id = None
    bad.billing_provider.npi = None                            # NPIs dropped
    bad.rendering_provider.npi = None
    bad.service_lines = bad.service_lines[:1]                  # procedure line dropped
    bad.service_lines[0].diagnosis_pointers = []
    bad.insurance.group_number = "GRP-99999"                   # invented value
    p = verify_against_source(bad, text_a)
    assert p.insurance.payer_name == "ExamplePayer PPO"
    assert p.insurance.member_id == "EP-78451233"
    assert p.billing_provider.npi == "1234567893"
    assert p.rendering_provider.npi == "1234567893"
    assert [l.procedure_code for l in p.service_lines] == ["99213", "87880"]
    assert p.service_lines[0].diagnosis_pointers == [1, 2]
    assert p.insurance.group_number is None
    actions = {(a.field, a.action) for a in p.verifier_actions}
    assert ("insurance.group_number", "removed_unsupported") in actions
    assert ("service_lines", "recovered_from_source") in actions


def test_insurance_returned_as_string_does_not_crash():
    c = ClaimData.model_validate({"insurance": "ExamplePayer PPO", "billing_provider": {"npi": 1234567893}})
    assert c.insurance.payer_name == "ExamplePayer PPO"
    assert c.insurance.member_id is None
    assert c.billing_provider.npi == "1234567893"


# --- Example B -----------------------------------------------------------
def test_example_b_marks_missing_and_never_invents(text_b):
    llm = example_b_claim().model_copy(deep=True)
    llm.ordering_provider.npi = "1999999999"   # the model "guessed" an ordering NPI
    llm.place_of_service = None
    p = extract_claim(text_b, client=FakeOpenAI(parsed=llm)).packet
    assert p.ordering_provider.npi is None, "invented NPI must be removed"
    assert p.ordering_provider.name is None
    assert p.rendering_provider.npi is None
    assert p.place_of_service is None
    assert p.service_lines[0].procedure_code == "71046"
    assert p.patient.date_of_birth == "1975-04-03"   # normalized from 04/03/1975
    for f in ("ordering_provider.npi", "rendering_provider.npi", "place_of_service"):
        assert p.field_status[f] == FieldStatus.MISSING, f


# --- Schema ----------------------------------------------------------------
def test_schema_is_valid_for_openai_structured_outputs():
    schema = to_strict_json_schema(ClaimData)
    text = json.dumps(schema)
    assert '"default"' not in text
    ins = schema["$defs"]["Insurance"]
    assert ins["type"] == "object" and set(ins["required"]) == {"payer_name", "plan_type", "member_id", "group_number"}
    assert schema["$defs"]["Provider"]["properties"]["npi"]["anyOf"][0]["type"] == "string"


# --- OpenAI error handling -----------------------------------------------
# The OpenAI SDK builds errors from an httpx response (newer SDKs use httpx2).
httpx = getattr(openai._exceptions, "httpx2", None) or getattr(openai._exceptions, "httpx")


def _status_error(cls, status: int, code: str | None = None):
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    body = {"error": {"message": "x", "code": code}}
    return cls("x", response=httpx.Response(status, request=req, json=body), body=body)


@pytest.mark.parametrize("exc,kind", [
    (_status_error(openai.RateLimitError, 429, "insufficient_quota"), "insufficient_quota"),
    (_status_error(openai.RateLimitError, 429, "rate_limit_exceeded"), "rate_limit"),
    (_status_error(openai.AuthenticationError, 401, "invalid_api_key"), "authentication"),
    (_status_error(openai.NotFoundError, 404, "model_not_found"), "model_access"),
    (_status_error(openai.PermissionDeniedError, 403), "model_access"),
    (_status_error(openai.BadRequestError, 400), "bad_request"),
])
def test_translate_error(exc, kind):
    err = translate_error(exc, "some-model")
    assert err.kind == kind and err.message


def test_failed_call_raises_and_never_returns_fake_output(text_a):
    client = FakeOpenAI(error=_status_error(openai.RateLimitError, 429, "insufficient_quota"))
    with pytest.raises(LLMError) as info:
        extract_claim(text_a, client=client)
    assert info.value.kind == "insufficient_quota"
    assert "Billing" in info.value.hint


@pytest.mark.parametrize("kwargs,kind", [({"parsed": None}, "malformed"), ({"parsed": ClaimData(), "refusal": "no"}, "refusal"),
                                         ({"parsed": ClaimData(), "finish_reason": "length"}, "truncated")])
def test_bad_responses_are_reported(text_a, kwargs, kind):
    with pytest.raises(LLMError) as info:
        extract_claim(text_a, client=FakeOpenAI(**kwargs))
    assert info.value.kind == kind


def test_config_problems(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    assert any("not set" in p for p in config.config_problems())
    monkeypatch.setenv("OPENAI_API_KEY", "sk-your-key-here")
    assert any("placeholder" in p for p in config.config_problems())
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-abcdefgh1234")
    monkeypatch.setenv("OPENAI_MODEL", "my-model")
    assert config.config_problems() == []
    s = config.get_settings()
    assert s.openai_model == "my-model" and s.masked_key() == "sk-...1234"
    # Non-OpenAI key is fine when an OpenAI-compatible endpoint (e.g. Gemini) is configured
    monkeypatch.setenv("OPENAI_API_KEY", "AIzaTestKey1234567890")
    assert any("sk-" in p for p in config.config_problems())
    monkeypatch.setenv("OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    assert config.config_problems() == []
    assert config.get_settings().base_url.startswith("https://generativelanguage")


def test_check_api_access_steps(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-abcdefgh1234")
    report = check_api_access(full=True, client=FakeOpenAI(parsed=lambda schema: schema(ok=True)))
    assert report["ok"] and [s["step"] for s in report["steps"]] == ["configuration", "model_access", "structured_output_call"]
    monkeypatch.setenv("OPENAI_API_KEY", "")
    assert check_api_access(full=True)["ok"] is False


# --- Packaging / previous-attempt regressions ------------------------------
def test_package_imports_without_sys_path_hacks():
    for py in list((ROOT / "src").rglob("*.py")) + [ROOT / "app.py"]:
        assert "sys.path" not in py.read_text(), py


def test_app_future_import_rule_and_syntax():
    tree = ast.parse((ROOT / "app.py").read_text())
    future = [i for i, n in enumerate(tree.body) if isinstance(n, ast.ImportFrom) and n.module == "__future__"]
    first_stmt = 1 if isinstance(tree.body[0], ast.Expr) else 0  # module docstring allowed first
    assert all(i == first_stmt for i in future)


def test_no_private_pandas_api_used():
    for py in (ROOT / "src").rglob("*.py"):
        assert "NoNewAttributesMixin" not in py.read_text()


def test_no_api_key_in_repo():
    for py in list(ROOT.rglob("*.py")) + list(ROOT.glob("*.md")):
        if ".venv" in py.parts:
            continue
        assert not __import__("re").search(r"sk-[A-Za-z0-9]{20,}", py.read_text()), py


# --- Optional live test ------------------------------------------------------
@pytest.mark.live
@pytest.mark.skipif(not config.get_settings().has_api_key or os.getenv("RUN_LIVE_TESTS") != "1",
                    reason="set OPENAI_API_KEY and RUN_LIVE_TESTS=1 to call the real API")
def test_live_example_a(text_a):
    p = extract_claim(text_a).packet
    assert p.insurance.member_id == "EP-78451233"
    assert {l.procedure_code for l in p.service_lines} == {"99213", "87880"}
