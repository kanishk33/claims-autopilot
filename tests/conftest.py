"""Shared test helpers.

`FakeOpenAI` imitates the small part of the OpenAI client we use
(`chat.completions.parse` and `models.retrieve`), so tests never hit the
network and never need an API key.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from claims_autopilot.config import DATA_DIR
from claims_autopilot.schemas import ClaimData


class FakeOpenAI:
    def __init__(self, parsed=None, error: Exception | None = None, refusal: str | None = None,
                 finish_reason: str = "stop"):
        self.calls: list[dict] = []
        self._parsed, self._error, self._refusal, self._finish = parsed, error, refusal, finish_reason
        self.chat = SimpleNamespace(completions=SimpleNamespace(parse=self._parse))
        self.models = SimpleNamespace(retrieve=lambda model: SimpleNamespace(id=model))

    def _parse(self, *, model, messages, response_format):
        self.calls.append({"model": model, "messages": messages, "response_format": response_format})
        if self._error:
            raise self._error
        parsed = self._parsed(response_format) if callable(self._parsed) else self._parsed
        message = SimpleNamespace(parsed=parsed, refusal=self._refusal)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=self._finish)],
                               model=model, usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50))


@pytest.fixture
def text_a() -> str:
    return (DATA_DIR / "sample_superbill_good.txt").read_text()


@pytest.fixture
def text_b() -> str:
    return (DATA_DIR / "sample_superbill_incomplete.txt").read_text()


def example_a_claim() -> ClaimData:
    """What a correct extraction of Example A looks like."""
    return ClaimData.model_validate({
        "patient": {"full_name": "SYNTHETIC PATIENT A", "date_of_birth": "1987-04-12", "sex": "F"},
        "insurance": {"payer_name": "ExamplePayer PPO", "plan_type": "PPO", "member_id": "EP-78451233"},
        "billing_provider": {"name": "Sample Family Medicine Group", "npi": "1234567893"},
        "rendering_provider": {"name": "Dr. Alex Example, MD", "npi": "1234567893"},
        "service_date": "2026-02-12",
        "place_of_service": "11",
        "diagnoses": [{"pointer": 1, "code": "J02.9", "description": "Acute pharyngitis, unspecified"},
                      {"pointer": 2, "code": "R50.9", "description": "Fever, unspecified"}],
        "service_lines": [
            {"line_number": 1, "procedure_code": "99213", "units": 1, "modifiers": [], "diagnosis_pointers": [1, 2], "charge": 145.0},
            {"line_number": 2, "procedure_code": "87880", "units": 1, "modifiers": [], "diagnosis_pointers": [1], "charge": 35.0},
        ],
    })


def example_b_claim() -> ClaimData:
    return ClaimData.model_validate({
        "patient": {"full_name": "SYNTHETIC PATIENT B", "date_of_birth": "1975-04-03", "sex": "M"},
        "insurance": {"payer_name": "DemoHealth HMO", "plan_type": "HMO", "member_id": "DH-55120987"},
        "billing_provider": {"name": "Sample Imaging & Primary Care", "npi": "1245319599"},
        "rendering_provider": {"name": "Dr. Morgan Placeholder", "npi": None},
        "service_date": "2026-03-03",
        "diagnoses": [{"pointer": 1, "code": "R05.9", "description": "Cough, unspecified"}],
        "service_lines": [{"line_number": 1, "procedure_code": "71046", "units": 1, "diagnosis_pointers": [1], "charge": 120.0}],
    })
