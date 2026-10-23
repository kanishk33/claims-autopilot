-- Claims Autopilot analytics schema (SQLite). Synthetic data only.
CREATE TABLE IF NOT EXISTS claims (
    claim_id            TEXT PRIMARY KEY,
    source              TEXT NOT NULL,          -- 'app' or 'synthetic_eval'
    payer_name          TEXT,
    plan_type           TEXT,
    service_date        TEXT,
    place_of_service    TEXT,
    billing_npi         TEXT,
    rendering_npi       TEXT,
    ordering_npi        TEXT,
    total_charge        REAL,
    risk_level          TEXT,
    risk_score          INTEGER,
    n_errors            INTEGER,
    n_risks             INTEGER,
    created_at          TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS diagnoses (
    claim_id   TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
    pointer    INTEGER NOT NULL,
    icd10_code TEXT NOT NULL,
    PRIMARY KEY (claim_id, pointer)
);

CREATE TABLE IF NOT EXISTS service_lines (
    claim_id            TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
    line_number         INTEGER NOT NULL,
    procedure_code      TEXT NOT NULL,
    modifiers           TEXT,            -- comma separated
    units               INTEGER,
    diagnosis_pointers  TEXT,            -- comma separated
    charge              REAL,
    PRIMARY KEY (claim_id, line_number)
);

CREATE TABLE IF NOT EXISTS validation_issues (
    issue_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id        TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
    rule_id         TEXT NOT NULL,
    category        TEXT NOT NULL,
    severity        TEXT NOT NULL,
    field           TEXT,
    line_number     INTEGER,
    demo_assumption INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS denials (
    denial_id       TEXT PRIMARY KEY,
    claim_id        TEXT,
    payer_name      TEXT,
    denied_amount   REAL,
    recommended_path TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS denial_codes (
    denial_id   TEXT NOT NULL REFERENCES denials(denial_id) ON DELETE CASCADE,
    code_type   TEXT NOT NULL,      -- CARC / RARC
    group_code  TEXT,
    code        TEXT NOT NULL
);
