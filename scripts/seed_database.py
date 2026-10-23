"""Load the labeled synthetic claims and denials into the SQLite analytics DB.

    python scripts/seed_database.py

Claims are validated with the same rules the app uses; denials are stored
with the CARC/RARC codes extracted from their text. The "recommended path"
stored for seeded denials comes from the reference table (no LLM call).
"""

from __future__ import annotations

from claims_autopilot.config import SYNTHETIC_DENIALS_DIR, get_settings
from claims_autopilot.database import connect, run_query, save_claim, save_denial
from claims_autopilot.denial import deterministic_steps, extract_codes
from claims_autopilot.evaluation import EVAL_AS_OF, load_claim_truth, load_denial_truth
from claims_autopilot.schemas import ClaimData
from claims_autopilot.validator import validate_claim


def main() -> None:
    with connect() as conn:
        for row in load_claim_truth():
            claim = ClaimData.model_validate(row["expected_claim"])
            save_claim(conn, claim, validate_claim(claim, as_of=EVAL_AS_OF), claim_id=row["claim_id"],
                       source="synthetic_eval")
        for row in load_denial_truth():
            codes = extract_codes((SYNTHETIC_DENIALS_DIR / row["file"]).read_text())
            steps = deterministic_steps(codes)
            path = next((s["typical_path"] for s in steps if s["type"] == "CARC"), "needs_human_review")
            save_denial(conn, codes, denial_id=row["denial_id"], claim_id=row["claim_id"],
                        payer_name=row["payer_name"], denied_amount=row["denied_amount"], recommended_path=path)
        print(run_query(conn, "kpis").to_string(index=False))
    print(f"Database: {get_settings().db_path}")


if __name__ == "__main__":
    main()
