"""SQLite analytics store.

Claims, validation results, and denials are saved to a local SQLite file so
they can be analysed with plain SQL (see sql/analytics.sql). SQLite ships
with Python, so there is nothing extra to install.
"""

from __future__ import annotations

import re
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd

from .config import get_settings
from .schemas import ClaimData, DenialCode, ValidationReport

SQL_DIR = Path(__file__).parent / "sql"


@contextmanager
def connect(db_path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    path = Path(db_path) if db_path else get_settings().db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        conn.executescript((SQL_DIR / "schema.sql").read_text())
        yield conn
        conn.commit()
    finally:
        conn.close()


def load_queries() -> dict[str, str]:
    """Named queries from analytics.sql (blocks start with '-- name: x')."""
    text = (SQL_DIR / "analytics.sql").read_text()
    out = {}
    for block in re.split(r"^-- name: ", text, flags=re.MULTILINE)[1:]:
        name, _, sql = block.partition("\n")
        out[name.strip()] = sql.strip()
    return out


def save_claim(conn: sqlite3.Connection, claim: ClaimData, report: ValidationReport,
               claim_id: str | None = None, source: str = "app") -> str:
    claim_id = claim_id or f"APP-{uuid.uuid4().hex[:8].upper()}"
    conn.execute("DELETE FROM claims WHERE claim_id = ?", (claim_id,))
    total = sum(l.charge or 0 for l in claim.service_lines)
    conn.execute(
        """INSERT INTO claims (claim_id, source, payer_name, plan_type, service_date, place_of_service,
                               billing_npi, rendering_npi, ordering_npi, total_charge, risk_level,
                               risk_score, n_errors, n_risks)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (claim_id, source, claim.insurance.payer_name, claim.insurance.plan_type, claim.service_date,
         claim.place_of_service, claim.billing_provider.npi, claim.rendering_provider.npi,
         claim.ordering_provider.npi, total, report.risk_level.value, report.risk_score,
         len(report.errors), len(report.risks)),
    )
    conn.executemany("INSERT INTO diagnoses VALUES (?,?,?)",
                     [(claim_id, d.pointer, d.code) for d in claim.diagnoses])
    conn.executemany(
        "INSERT INTO service_lines VALUES (?,?,?,?,?,?,?)",
        [(claim_id, l.line_number, l.procedure_code, ",".join(l.modifiers), l.units,
          ",".join(map(str, l.diagnosis_pointers)), l.charge) for l in claim.service_lines],
    )
    conn.executemany(
        """INSERT INTO validation_issues (claim_id, rule_id, category, severity, field, line_number, demo_assumption)
           VALUES (?,?,?,?,?,?,?)""",
        [(claim_id, i.rule_id, i.category.value, i.severity.value, i.field, i.line_number, int(i.demo_assumption))
         for i in report.issues],
    )
    return claim_id


def save_denial(conn: sqlite3.Connection, codes: list[DenialCode], *, denial_id: str | None = None,
                claim_id: str | None = None, payer_name: str | None = None, denied_amount: float | None = None,
                recommended_path: str | None = None) -> str:
    denial_id = denial_id or f"DEN-{uuid.uuid4().hex[:8].upper()}"
    conn.execute("DELETE FROM denials WHERE denial_id = ?", (denial_id,))
    conn.execute("INSERT INTO denials (denial_id, claim_id, payer_name, denied_amount, recommended_path) VALUES (?,?,?,?,?)",
                 (denial_id, claim_id, payer_name, denied_amount, recommended_path))
    conn.executemany("INSERT INTO denial_codes VALUES (?,?,?,?)",
                     [(denial_id, c.code_type, c.group_code, c.code) for c in codes])
    return denial_id


def run_query(conn: sqlite3.Connection, name: str) -> pd.DataFrame:
    return pd.read_sql_query(load_queries()[name], conn)
