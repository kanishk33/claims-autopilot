"""Headless smoke test of the Streamlit app (no browser, no API calls)."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture
def no_key(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("CLAIMS_DB_PATH", str(tmp_path / "app.db"))


def test_app_starts_with_four_tabs(no_key):
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["① Build Claim", "② Precheck", "③ Denial Resolution", "④ Analytics"]


def test_extract_without_key_shows_error_not_fake_output(no_key):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.text_area[0].set_value("Patient Name: SYNTHETIC PATIENT A\nCPT 99213").run()
    extract = next(b for b in at.button if "Extract claim" in b.label)
    extract.click().run()
    assert not at.exception
    assert any("OpenAI request failed" in e.value for e in at.error)
    assert at.session_state["packet"] is None
