# 5-minute demo script

Before the demo: `python scripts/check_openai.py` passes, `python scripts/seed_database.py` has run, and the app is open (`streamlit run app.py`).

---

**0:00 · The problem (30 s)**
> "Billing teams turn visit notes into claims by hand, and missing or inconsistent data is a major cause of denials. When a denial comes back, someone has to decode CARC and RARC codes and work out the fix. Claims Autopilot is a prototype that helps with all three steps: building the claim, checking it before submission, and resolving denials. It uses synthetic data only, and a human reviews everything."

**0:30 · Build Claim, Example A (60 s)**
- Select *Example A*, click **Extract claim with OpenAI**.
> "The model returns JSON that must match a Pydantic schema; that's OpenAI Structured Outputs. Insurance name and member ID are separate fields, NPIs are strings, and each procedure line has units, modifiers, and diagnosis pointers."
- Open **Source verification log**.
> "LLMs sometimes drop or invent values, so every identifier is checked against the source text. Unsupported values are removed, and clearly labeled values the model missed are restored. The claim only contains what was actually written."

**1:30 · Precheck (60 s)**
- Switch to Example B, extract, then open **Precheck**.
> "These rules are in a YAML file, so a billing team can change them without code. Red items are validation errors: the rendering NPI and place of service are missing. The chest X-ray without an ordering provider is a *denial risk*, because whether it is required depends on the payer. Each issue explains why it matters and generates a question for the provider's office. The risk score is a transparent formula, not a black box."

**2:30 · Denial Resolution (75 s)**
- Load *Sample denial*, click **Analyze denial**.
> "The codes CO-16, N265, and N290 are looked up in a local table with sources, not from the model's memory. The app cross-checks the denial against the claim and sees those fields are still empty. Retrieved guidance goes to the LLM, which recommends a corrected claim rather than an appeal, because CO-16 is a data problem. It cites the passages it used."
- Click **Draft letter**.
> "The draft uses placeholders for anything we don't have. If it ever included an NPI or code that wasn't in the input, the app would flag it. Nothing is ever submitted automatically."

**3:45 · Analytics and evaluation (60 s)**
- Open **Analytics**.
> "Everything lands in SQLite, and these tables are plain SQL: clean-claim rate, top issues, issues by payer, denial codes. I evaluated the pipeline on 28 labeled synthetic claims with seeded defects. The rules catch every defect they're designed to see, and the claim error rate drops from 64% to 21% in that simulation. The three kinds of defect the rules can't catch, like a valid but clinically wrong diagnosis, show where human review still matters."

**4:45 · Close (15 s)**
> "Next steps: publish the live LLM accuracy numbers, parse synthetic 835 files, and add coverage-policy retrieval from the CMS Medicare Coverage Database."

---

**Likely questions**
- *Was it trained on these codes?* No. It retrieves local reference data at run time (RAG). No fine-tuning.
- *Is it HIPAA compliant?* No. It's a prototype for synthetic data only.
- *What if the API fails?* The app shows the exact error (auth, quota, rate limit, model access) and produces no result. It never fakes one.
