"""Feature 3: denial code extraction, reference lookup, RAG, letters, analytics DB."""

from __future__ import annotations

import pandas as pd

from claims_autopilot.config import DATA_DIR, REFERENCE_PATH
from claims_autopilot.database import connect, load_queries, run_query, save_claim, save_denial
from claims_autopilot.denial import analyze_denial, extract_codes, prepare_denial
from claims_autopilot.evaluation import evaluate_denials
from claims_autopilot.generator import draft_letter, unsupported_identifiers
from claims_autopilot.retrieval import retrieve
from claims_autopilot.schemas import AppealLetter, DenialAnalysis
from claims_autopilot.validator import validate_claim

from conftest import FakeOpenAI, example_a_claim, example_b_claim


def sample_denial() -> str:
    return (DATA_DIR / "sample_denial.txt").read_text()


def test_sample_denial_codes_and_official_meanings():
    codes = extract_codes(sample_denial())
    assert [(c.group_code, c.code_type, c.code) for c in codes] == [("CO", "CARC", "16"), (None, "RARC", "N265"), (None, "RARC", "N290")]
    assert all(c.found_in_reference and c.source.startswith("https://") for c in codes)
    assert codes[1].description == "Missing/incomplete/invalid ordering provider primary identifier."


def test_code_formats():
    text = "Adj CO 197 and PR-2; remark codes M77, MA130. Reason code 50. Also N999."
    got = {(c.code_type, c.code) for c in extract_codes(text)}
    assert {("CARC", "197"), ("CARC", "2"), ("RARC", "M77"), ("RARC", "MA130"), ("CARC", "50"), ("RARC", "N999")} == got
    unknown = prepare_denial(text).warnings
    assert any("N999" in w for w in unknown)


def test_crosscheck_with_claim_and_retrieval():
    d = prepare_denial(sample_denial(), example_b_claim())
    assert any("ordering_provider.npi" in n and "still has no value" in n for n in d.crosscheck)
    ids = [p.id for p in d.passages]
    assert ids[:3] == ["CARC-16", "RARC-N265", "RARC-N290"]
    assert "KB-ORDERING-PROVIDER" in ids
    assert retrieve("prior authorization missing", k=2)[0].id in {"KB-PRIOR-AUTH", "CARC-197"}


def test_reference_file_is_sourced():
    df = pd.read_csv(REFERENCE_PATH, dtype=str)
    assert not df.duplicated(["code_type", "code"]).any()
    assert df["source_url"].str.startswith("https://").all()
    assert df["retrieved_on"].notna().all() and df["official_description"].str.len().gt(3).all()


def test_analyze_denial_filters_uncited_sources():
    analysis = DenialAnalysis(
        plain_language_summary="Missing ordering and rendering NPIs.", likely_root_cause="Data omitted.",
        recommended_path="corrected_claim", correction_steps=["Add NPIs"], information_needed=["Ordering NPI"],
        cited_source_ids=["RARC-N265", "MADE-UP-SOURCE"])
    client = FakeOpenAI(parsed=analysis)
    d = analyze_denial(sample_denial(), example_b_claim(), client=client)
    assert d.analysis.cited_source_ids == ["RARC-N265"]
    assert any("MADE-UP-SOURCE" in w for w in d.warnings)
    prompt = client.calls[0]["messages"][1]["content"]
    assert "Missing/incomplete/invalid ordering provider primary identifier." in prompt  # grounded in reference


def test_letter_flags_invented_identifiers():
    d = analyze_denial(sample_denial(), example_b_claim(), client=FakeOpenAI(parsed=DenialAnalysis(
        plain_language_summary="s", likely_root_cause="r", recommended_path="corrected_claim",
        correction_steps=["x"], information_needed=[], cited_source_ids=[])))
    letter = AppealLetter(subject="Corrected claim SYN-CLM-0002",
                          body="DRAFT - FOR HUMAN REVIEW - SYNTHETIC DATA\nOrdering NPI 1999999999. [ORDERING PROVIDER NAME]",
                          placeholders=[], attachments_to_gather=["Signed order"])
    _, _, warnings = draft_letter(d, example_b_claim(), client=FakeOpenAI(parsed=letter))
    assert warnings == ["1999999999"]
    assert "[ORDERING PROVIDER NAME]" in letter.placeholders
    assert unsupported_identifiers("CPT 71046 on 2026-03-03", "71046 2026-03-03") == []


def test_denial_evaluation():
    res = evaluate_denials()
    assert res["code_precision"] == 1.0 and res["code_recall"] == 1.0


def test_sqlite_analytics(tmp_path):
    db = tmp_path / "t.db"
    with connect(db) as conn:
        for cid, claim in (("A", example_a_claim()), ("B", example_b_claim())):
            save_claim(conn, claim, validate_claim(claim), claim_id=cid)
        save_denial(conn, extract_codes(sample_denial()), denial_id="D1", claim_id="B", denied_amount=120.0,
                    recommended_path="corrected_claim")
        results = {name: run_query(conn, name) for name in load_queries()}
    assert results["kpis"].iloc[0]["claims"] == 2 and results["kpis"].iloc[0]["clean_claims"] == 1
    assert set(results["risk_distribution"]["risk_level"]) == {"LOW", "HIGH"}
    assert "REQ_RENDERING_NPI" in set(results["top_issues"]["rule_id"])
    assert "CO-16" in set(results["denial_codes_summary"]["code"])
