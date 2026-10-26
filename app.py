"""Claims Autopilot - Streamlit user interface.

Run with:  streamlit run app.py

PORTFOLIO PROTOTYPE. Synthetic data only. Nothing is ever submitted to a payer.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from claims_autopilot.config import (
    DATA_DIR,
    OUTPUTS_DIR,
    SYNTHETIC_CLAIMS_DIR,
    SYNTHETIC_DENIALS_DIR,
    config_problems,
    get_settings,
    setup_logging,
)
from claims_autopilot.database import connect, load_queries, run_query, save_claim, save_denial
from claims_autopilot.denial import analyze_denial, prepare_denial
from claims_autopilot.extractor import extract_claim
from claims_autopilot.generator import draft_letter
from claims_autopilot.llm import LLMError, check_api_access
from claims_autopilot.questioner import checklist, draft_provider_query, follow_up_questions
from claims_autopilot.schemas import ClaimPacket, FieldStatus, IssueCategory, RiskLevel
from claims_autopilot.validator import load_rules, validate_claim

setup_logging()
st.set_page_config(page_title="Claims Autopilot", page_icon="🩺", layout="wide")

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
.block-container {padding-top: 3rem; max-width: 1280px;}
.ca-hero {display:flex; align-items:center; justify-content:space-between; gap:1rem;
          background: linear-gradient(120deg, #0F766E 0%, #115E59 55%, #134E4A 100%);
          color:#fff; border-radius:16px; padding:18px 24px; margin-bottom:14px;}
.ca-hero h1 {color:#fff; font-size:1.65rem; margin:0; padding:0;}
.ca-hero p {color:#CCFBF1; margin:2px 0 0 0; font-size:0.95rem;}
.ca-pill {display:inline-block; padding:3px 10px; border-radius:999px; font-size:0.75rem; font-weight:600;
          background:rgba(255,255,255,0.16); color:#fff; margin-left:6px; border:1px solid rgba(255,255,255,0.3);}
.ca-card {border:1px solid #E2E8F0; border-radius:12px; padding:14px 16px; background:#fff; height:100%;}
.ca-card h4 {margin:0 0 8px 0; font-size:0.8rem; letter-spacing:.06em; text-transform:uppercase; color:#64748B;}
.ca-row {display:flex; justify-content:space-between; gap:8px; padding:3px 0; font-size:0.92rem;
         border-bottom:1px dashed #EEF2F6;}
.ca-row span:first-child {color:#64748B;}
.ca-row span:last-child {font-weight:600; text-align:right;}
.ca-missing {color:#B91C1C !important;}
.ca-verify {color:#B45309 !important;}
.ca-risk {border-radius:14px; padding:16px 18px; color:#fff;}
.ca-risk .lvl {font-size:2rem; font-weight:800; line-height:1;}
.ca-risk .sub {opacity:.9; font-size:.9rem; margin-top:4px;}
.risk-HIGH {background:#B91C1C;} .risk-MEDIUM {background:#B45309;} .risk-LOW {background:#15803D;}
.ca-note {font-size:0.82rem; color:#64748B;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="ca-hero">
  <div><h1>🩺 Claims Autopilot</h1>
  <p>LLM-assisted claim building, pre-submission checks, and denial resolution for revenue-cycle teams.</p></div>
  <div><span class="ca-pill">SYNTHETIC DATA ONLY</span><span class="ca-pill">PORTFOLIO PROTOTYPE</span>
  <span class="ca-pill">HUMAN REVIEW REQUIRED</span></div>
</div>
""",
    unsafe_allow_html=True,
)

ss = st.session_state
for key, default in {"claim_text": "", "packet": None, "extract_meta": None, "denial_text": "",
                     "denial": None, "letter": None, "provider_query": None}.items():
    ss.setdefault(key, default)


def show_llm_error(err: LLMError) -> None:
    st.error(f"**OpenAI request failed ({err.kind}).** {err.message}")
    if err.hint:
        st.info(f"How to fix: {err.hint}")
    st.caption("No result was produced. The app never fills in fake LLM output when a request fails.")


def sample_files(folder: Path, pattern: str) -> dict[str, Path]:
    return {p.stem: p for p in sorted(folder.glob(pattern))}


# ---------------------------------------------------------------------------
# Sidebar: configuration and API status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("OpenAI connection")
    settings = get_settings()
    problems = config_problems(settings)
    st.write(f"**Model:** `{settings.openai_model}`")
    if settings.base_url:
        st.write(f"**Endpoint:** `{settings.base_url}`")
    st.write(f"**API key:** `{settings.masked_key()}`")
    if problems:
        for p in problems:
            st.warning(p)
    else:
        st.caption("Key is configured. That does not prove quota or model access, so run the test below.")
    if st.button("Test OpenAI connection", use_container_width=True):
        with st.spinner("Checking configuration, model access, and quota..."):
            report = check_api_access(full=True)
        for step in report["steps"]:
            (st.success if step["ok"] else st.error)(f"{step['step']}: {step['detail']}")
    st.divider()
    st.subheader("About")
    st.caption(
        "Portfolio prototype using synthetic data. It does not submit claims, make coverage or "
        "medical-necessity decisions, or claim HIPAA compliance. Rules marked *demo assumption* are "
        "examples, not payer policy. Code meanings come from a small sourced subset "
        "(see `data/carc_rarc_reference.csv`)."
    )

tab_build, tab_check, tab_denial, tab_analytics = st.tabs(
    ["① Build Claim", "② Precheck", "③ Denial Resolution", "④ Analytics"]
)

# ---------------------------------------------------------------------------
# Tab 1: Build Claim
# ---------------------------------------------------------------------------
STATUS_LABEL = {FieldStatus.PROVIDED: "✅ provided", FieldStatus.MISSING: "❌ missing",
                FieldStatus.NEEDS_VERIFICATION: "⚠️ verify"}


def row(label: str, value, path: str, packet: ClaimPacket, optional: bool = False) -> str:
    status = packet.field_status.get(path)
    empty = value in (None, "")
    cls = ("" if optional else "ca-missing") if empty else (
        "ca-verify" if status == FieldStatus.NEEDS_VERIFICATION else "")
    shown = ("not listed" if optional else "missing") if empty else value
    if status == FieldStatus.NEEDS_VERIFICATION:
        shown = f"{shown} ⚠️"
    return f'<div class="ca-row"><span>{label}</span><span class="{cls}">{shown}</span></div>'


def card(title: str, rows: list[str]) -> str:
    return f'<div class="ca-card"><h4>{title}</h4>{"".join(rows)}</div>'


with tab_build:
    left, right = st.columns([5, 7], gap="large")
    with left:
        st.markdown("#### Visit summary or superbill")
        samples = {"Example A: complete superbill": DATA_DIR / "sample_superbill_good.txt",
                   "Example B: incomplete visit summary": DATA_DIR / "sample_superbill_incomplete.txt"}
        samples.update({f"Eval set: {k}": v for k, v in sample_files(SYNTHETIC_CLAIMS_DIR, "claim_*.txt").items()})

        def _load_claim_sample() -> None:
            if ss.claim_choice in samples:
                ss.claim_text = samples[ss.claim_choice].read_text()

        st.selectbox("Load a synthetic sample", ["(paste your own)"] + list(samples), key="claim_choice",
                     on_change=_load_claim_sample)
        st.text_area("Synthetic text only. Do not paste real patient information.", key="claim_text", height=420)
        go = st.button("✨ Extract claim with OpenAI", type="primary", use_container_width=True,
                       disabled=not ss.claim_text.strip())
        if go:
            with st.status("Extracting structured claim...", expanded=True) as status:
                try:
                    st.write("Calling OpenAI (Structured Outputs)...")
                    result = extract_claim(ss.claim_text)
                    st.write("Verifying every value against the source text...")
                    ss.packet = result.packet
                    ss.extract_meta = result.llm
                    ss.denial, ss.letter, ss.provider_query = None, None, None
                    status.update(label="Claim extracted and verified", state="complete")
                except LLMError as err:
                    status.update(label="Extraction failed", state="error")
                    show_llm_error(err)
                except ValueError as err:
                    status.update(label="Extraction failed", state="error")
                    st.error(str(err))

    with right:
        packet: ClaimPacket | None = ss.packet
        if packet is None:
            st.info("Load a sample (or paste text) and click **Extract claim with OpenAI**. The structured "
                    "claim, field status, and service lines will appear here.")
        else:
            meta = ss.extract_meta
            if meta is not None:
                st.success(f"LLM request succeeded · model `{meta.model}` · {meta.seconds}s · "
                           f"tokens in/out {meta.input_tokens}/{meta.output_tokens}")
            else:
                st.info("Claim loaded from JSON (no LLM call was made for this claim).")
            st.markdown("#### Claim summary")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(card("Patient & coverage", [
                    row("Patient", packet.patient.full_name, "patient.full_name", packet),
                    row("Date of birth", packet.patient.date_of_birth, "patient.date_of_birth", packet),
                    row("Sex", packet.patient.sex, "patient.sex", packet),
                    row("Payer", packet.insurance.payer_name, "insurance.payer_name", packet),
                    row("Member ID", packet.insurance.member_id, "insurance.member_id", packet),
                ]), unsafe_allow_html=True)
            with c2:
                st.markdown(card("Providers & encounter", [
                    row("Billing NPI", packet.billing_provider.npi, "billing_provider.npi", packet),
                    row("Rendering NPI", packet.rendering_provider.npi, "rendering_provider.npi", packet),
                    row("Ordering provider", packet.ordering_provider.name, "ordering_provider.name", packet, optional=True),
                    row("Date of service", packet.service_date, "service_date", packet),
                    row("Place of service", packet.place_of_service, "place_of_service", packet),
                ]), unsafe_allow_html=True)

            st.markdown("#### Service lines")
            dx_map = {d.pointer: d.code for d in packet.diagnoses}
            lines_df = pd.DataFrame([{
                "Line": l.line_number, "CPT/HCPCS": l.procedure_code, "Modifiers": ", ".join(l.modifiers) or "—",
                "Units": l.units, "Dx pointers": ", ".join(str(p) for p in l.diagnosis_pointers) or "—",
                "Dx codes": ", ".join(dx_map.get(p, f"?{p}") for p in l.diagnosis_pointers) or "—",
                "Charge": f"${l.charge:,.2f}" if l.charge is not None else "—",
            } for l in packet.service_lines])
            st.dataframe(lines_df, hide_index=True, use_container_width=True)
            st.markdown("#### Diagnoses")
            st.dataframe(pd.DataFrame([{"Pointer": d.pointer, "ICD-10-CM": d.code, "Description (as written)": d.description or "—"}
                                       for d in packet.diagnoses]), hide_index=True, use_container_width=True)

            n_missing, n_verify = len(packet.missing_fields), len(packet.fields_to_verify)
            m1, m2, m3 = st.columns(3)
            m1.metric("Fields provided", sum(1 for s in packet.field_status.values() if s == FieldStatus.PROVIDED))
            m2.metric("Missing", n_missing)
            m3.metric("Needs verification", n_verify)
            with st.expander("Field status (provided / missing / needs verification)", expanded=n_missing + n_verify > 0):
                st.dataframe(pd.DataFrame([{"Field": f, "Status": STATUS_LABEL[s]} for f, s in packet.field_status.items()]),
                             hide_index=True, use_container_width=True)
            with st.expander(f"Source verification log ({len(packet.verifier_actions)} actions)"):
                if packet.verifier_actions:
                    st.dataframe(pd.DataFrame([a.model_dump() for a in packet.verifier_actions]), hide_index=True,
                                 use_container_width=True)
                else:
                    st.write("Every LLM value was found in the source text and nothing labeled was missed.")
            with st.expander("Structured claim JSON"):
                st.json(packet.model_dump(mode="json"))
            st.download_button("Download claim JSON", packet.model_dump_json(indent=2), "claim_packet.json",
                               "application/json")

# ---------------------------------------------------------------------------
# Tab 2: Precheck
# ---------------------------------------------------------------------------
with tab_check:
    with st.expander("Import or edit a claim packet JSON (for example to apply answers to follow-up questions)"):
        current = ss.packet.model_dump_json(indent=2) if ss.packet else ""
        edited = st.text_area("Claim packet JSON", value=current, height=260, key="json_editor")
        if st.button("Apply JSON"):
            try:
                ss.packet = ClaimPacket.model_validate_json(edited)
                ss.extract_meta = None
                st.success("Claim updated. Precheck re-ran below.")
            except ValidationError as exc:
                st.error(f"The JSON does not match the claim schema:\n\n{exc}")

    packet = ss.packet
    if packet is None:
        st.info("Build a claim in tab ① first (or import a claim packet JSON above).")
    else:
        report = validate_claim(packet)
        risk_col, metric_col = st.columns([4, 8], gap="large")
        with risk_col:
            st.markdown(
                f'<div class="ca-risk risk-{report.risk_level.value}"><div class="lvl">{report.risk_level.value} RISK</div>'
                f'<div class="sub">Score {report.risk_score} · {"ready after review" if report.ready_to_submit else "fix errors before submission"}</div></div>',
                unsafe_allow_html=True)
            for line in report.risk_explanation:
                st.caption(line)
        with metric_col:
            a, b, c = st.columns(3)
            a.metric("Validation errors", len(report.errors), help="Missing or invalid per the configured rules.")
            b.metric("Denial risks", len(report.risks), help="Payer-dependent: may cause a denial for some payers.")
            c.metric("Advisories", sum(1 for i in report.issues if i.category == IssueCategory.INFO))
            st.caption("Validation errors are definite problems under the rules file. Denial risks depend on the payer, "
                       "plan, and policy, so they are flagged for review, not treated as certain denials.")

        st.markdown("#### Pre-submission checklist")
        if report.issues:
            st.data_editor(pd.DataFrame(checklist(report)).assign(Status=False),
                           column_config={"Status": st.column_config.CheckboxColumn("Done")},
                           hide_index=True, use_container_width=True, key="checklist_editor")
        else:
            st.success("No issues found under the configured rules. This is not a guarantee of payment.")

        qs = follow_up_questions(report)
        if qs:
            st.markdown("#### Follow-up questions for the provider's office")
            for q in qs:
                st.markdown(f"- **[{q['priority']}]** {q['question']}  \n  <span class='ca-note'>{q['rule_id']} · {q['field']}</span>",
                            unsafe_allow_html=True)
            if st.button("✉️ Draft one message to the office with OpenAI"):
                try:
                    with st.spinner("Drafting..."):
                        ss.provider_query = draft_provider_query(report.issues, claim_ref="(draft claim)").parsed
                except LLMError as err:
                    show_llm_error(err)
            if ss.provider_query:
                st.text_input("Subject", ss.provider_query.subject)
                st.text_area("Message (review before sending)", ss.provider_query.message, height=200)

        demo_rules = [r["id"] for r in load_rules()["code_rules"]]
        st.caption(f"Demo-assumption rules configured in data/validation_rules.yml: {', '.join(demo_rules)}")
        if st.button("💾 Save claim + precheck to analytics database"):
            with connect() as conn:
                cid = save_claim(conn, packet, report)
            st.success(f"Saved as {cid}. See tab ④.")

# ---------------------------------------------------------------------------
# Tab 3: Denial Resolution
# ---------------------------------------------------------------------------
with tab_denial:
    left, right = st.columns([5, 7], gap="large")
    with left:
        st.markdown("#### Remittance / denial text")
        dsamples = {"Sample denial (Example B)": DATA_DIR / "sample_denial.txt"}
        dsamples.update({f"Eval set: {k}": v for k, v in sample_files(SYNTHETIC_DENIALS_DIR, "denial_*.txt").items()})

        def _load_denial_sample() -> None:
            if ss.denial_choice in dsamples:
                ss.denial_text = dsamples[ss.denial_choice].read_text()

        st.selectbox("Load a synthetic denial", ["(paste your own)"] + list(dsamples), key="denial_choice",
                     on_change=_load_denial_sample)
        st.text_area("Synthetic remittance text", key="denial_text", height=300)
        use_claim = st.checkbox("Cross-check against the claim from tab ①", value=ss.packet is not None,
                                disabled=ss.packet is None)
        if st.button("🔎 Analyze denial", type="primary", use_container_width=True, disabled=not ss.denial_text.strip()):
            claim = ss.packet if use_claim else None
            ss.letter = None
            try:
                with st.status("Analyzing denial...", expanded=True) as status:
                    st.write("Extracting CARC/RARC codes and looking them up in the local reference...")
                    ss.denial = prepare_denial(ss.denial_text, claim)
                    st.write("Retrieving guidance and asking OpenAI for an explanation...")
                    ss.denial = analyze_denial(ss.denial_text, claim)
                    status.update(label="Denial analyzed", state="complete")
            except LLMError as err:
                status.update(label="LLM step failed (code lookup still shown)", state="error")
                show_llm_error(err)

    with right:
        d = ss.denial
        if d is None:
            st.info("Load a denial and click **Analyze denial**.")
        else:
            for w in d.warnings:
                st.warning(w)
            st.markdown("#### Denial codes (from local reference, not the LLM)")
            st.dataframe(pd.DataFrame(d.reference_steps), hide_index=True, use_container_width=True,
                         column_config={"source": st.column_config.LinkColumn("source")})
            if d.crosscheck:
                st.markdown("#### Cross-check with the current claim")
                for n in d.crosscheck:
                    st.markdown(f"- {n}")
            if d.analysis:
                a = d.analysis
                st.success(f"LLM request succeeded · model `{d.llm.model}` · {d.llm.seconds}s")
                st.markdown("#### What happened")
                st.write(a.plain_language_summary)
                st.markdown(f"**Likely root cause:** {a.likely_root_cause}")
                st.markdown(f"**Recommended path:** `{a.recommended_path}`")
                st.markdown("#### Recommended correction steps")
                for i, s in enumerate(a.correction_steps, 1):
                    st.markdown(f"{i}. {s}")
                if a.information_needed:
                    st.markdown("#### Information a human must supply")
                    for s in a.information_needed:
                        st.markdown(f"- {s}")
                with st.expander(f"Retrieved sources ({len(d.passages)}) · cited: {', '.join(a.cited_source_ids) or 'none'}"):
                    for p in d.passages:
                        st.markdown(f"**[{p.id}] {p.title}** · _{p.source}_  \n{p.text}")
                if st.button("📝 Draft letter for human review", type="primary"):
                    try:
                        with st.spinner("Drafting..."):
                            ss.letter = draft_letter(d, ss.packet if use_claim else None)
                    except LLMError as err:
                        show_llm_error(err)
            else:
                with st.expander(f"Retrieved sources ({len(d.passages)})"):
                    for p in d.passages:
                        st.markdown(f"**[{p.id}] {p.title}** · _{p.source}_  \n{p.text}")
            if ss.letter:
                letter, meta, warnings = ss.letter
                st.markdown("#### Draft letter (not submitted anywhere)")
                if warnings:
                    st.warning("These identifiers in the draft were not in the input. Check them: " + ", ".join(warnings))
                body = st.text_area(letter.subject, letter.body, height=360)
                if letter.attachments_to_gather:
                    st.markdown("**Documents to gather (not attached):** " + "; ".join(letter.attachments_to_gather))
                st.download_button("Download draft (.txt)", f"{letter.subject}\n\n{body}", "draft_letter.txt")
            if st.button("💾 Save denial to analytics database"):
                with connect() as conn:
                    did = save_denial(conn, d.codes, recommended_path=d.analysis.recommended_path if d.analysis else None)
                st.success(f"Saved as {did}.")

# ---------------------------------------------------------------------------
# Tab 4: Analytics
# ---------------------------------------------------------------------------
with tab_analytics:
    st.markdown("#### Claims & denial analytics (SQLite + SQL)")
    st.caption("Built from the labeled synthetic evaluation set plus anything you saved from tabs ② and ③.")
    col_a, col_b = st.columns(2)
    if col_a.button("Load synthetic dataset into the database"):
        import runpy
        runpy.run_path(str(Path(__file__).parent / "scripts" / "seed_database.py"), run_name="__main__")
        st.success("Loaded 28 synthetic claims and 10 synthetic denials.")
    if col_b.button("Run evaluation (no API calls)"):
        import runpy
        with st.spinner("Evaluating..."):
            runpy.run_path(str(Path(__file__).parent / "scripts" / "evaluate.py"), run_name="__main__")
        st.success("Evaluation finished. Results below and in docs/EVALUATION.md.")

    with connect() as conn:
        kpi = run_query(conn, "kpis").iloc[0]
        if not kpi["claims"]:
            st.info("The database is empty. Click **Load synthetic dataset into the database**.")
        else:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Claims", int(kpi["claims"]))
            k2.metric("Clean-claim rate", f"{kpi['clean_claim_rate_pct']}%")
            k3.metric("Total charges", f"${kpi['total_charges']:,.0f}")
            k4.metric("Charges on hold (errors)", f"${kpi['charges_on_hold']:,.0f}")
            g1, g2 = st.columns(2)
            with g1:
                st.markdown("**Risk distribution**")
                st.bar_chart(run_query(conn, "risk_distribution").set_index("risk_level")["claims"], color="#0F766E", sort=False)
                st.markdown("**Issues per claim by payer**")
                st.dataframe(run_query(conn, "issues_by_payer"), hide_index=True, use_container_width=True)
            with g2:
                st.markdown("**Most frequent precheck issues**")
                st.dataframe(run_query(conn, "top_issues"), hide_index=True, use_container_width=True)
                st.markdown("**Denial codes**")
                st.dataframe(run_query(conn, "denial_codes_summary"), hide_index=True, use_container_width=True)
            st.markdown("**Denials by recommended path**")
            st.dataframe(run_query(conn, "denial_paths"), hide_index=True, use_container_width=True)
            with st.expander("SQL behind these tables (src/claims_autopilot/sql/analytics.sql)"):
                for name, sql in load_queries().items():
                    st.markdown(f"`{name}`")
                    st.code(sql, language="sql")

    results_file = OUTPUTS_DIR / "evaluation_results.json"
    if results_file.exists():
        res = json.loads(results_file.read_text())
        st.markdown("#### Latest evaluation (synthetic set)")
        e1, e2, e3, e4 = st.columns(4)
        pb, pc, dn = res["parser_baseline"], res["precheck"], res["denials"]
        llm_res = res.get("llm_extraction")
        e1.metric("LLM extraction accuracy", f"{100 * llm_res['field_accuracy']:.1f}%" if llm_res else "not run",
                  help="Run `python scripts/evaluate.py --llm` with a working key.")
        e2.metric("Parser-only baseline", f"{100 * pb['field_accuracy']:.1f}%")
        e3.metric("Error-rate reduction (sim.)", f"{100 * pc['relative_error_rate_reduction']:.1f}%")
        e4.metric("Denial code recall", f"{100 * dn['code_recall']:.1f}%")
        st.caption(f"Generated {res['generated_at']}. Full report: docs/EVALUATION.md")
