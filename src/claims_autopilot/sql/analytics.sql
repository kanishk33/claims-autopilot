-- name: kpis
SELECT
    COUNT(*)                                                    AS claims,
    SUM(CASE WHEN n_errors = 0 THEN 1 ELSE 0 END)               AS clean_claims,
    ROUND(100.0 * SUM(CASE WHEN n_errors = 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) AS clean_claim_rate_pct,
    ROUND(SUM(total_charge), 2)                                 AS total_charges,
    ROUND(SUM(CASE WHEN n_errors > 0 THEN total_charge ELSE 0 END), 2) AS charges_on_hold
FROM claims;

-- name: risk_distribution
SELECT risk_level, COUNT(*) AS claims, ROUND(SUM(total_charge), 2) AS charges
FROM claims
GROUP BY risk_level
ORDER BY CASE risk_level WHEN 'HIGH' THEN 1 WHEN 'MEDIUM' THEN 2 ELSE 3 END;

-- name: top_issues
SELECT rule_id, category, severity, COUNT(*) AS occurrences, COUNT(DISTINCT claim_id) AS claims_affected
FROM validation_issues
GROUP BY rule_id, category, severity
ORDER BY occurrences DESC, rule_id;

-- name: issues_by_payer
SELECT c.payer_name,
       COUNT(DISTINCT c.claim_id)                                   AS claims,
       COUNT(v.issue_id)                                            AS issues,
       ROUND(1.0 * COUNT(v.issue_id) / COUNT(DISTINCT c.claim_id), 2) AS issues_per_claim
FROM claims c
LEFT JOIN validation_issues v ON v.claim_id = c.claim_id
GROUP BY c.payer_name
ORDER BY issues_per_claim DESC;

-- name: procedure_volume
SELECT s.procedure_code, COUNT(*) AS lines, SUM(s.units) AS units, ROUND(SUM(s.charge), 2) AS charges
FROM service_lines s
GROUP BY s.procedure_code
ORDER BY lines DESC, s.procedure_code;

-- name: denial_codes_summary
SELECT d.code_type, COALESCE(d.group_code || '-', '') || d.code AS code, COUNT(*) AS occurrences,
       ROUND(SUM(n.denied_amount), 2) AS denied_amount
FROM denial_codes d
JOIN denials n ON n.denial_id = d.denial_id
GROUP BY d.code_type, d.group_code, d.code
ORDER BY occurrences DESC, code;

-- name: denial_paths
SELECT recommended_path, COUNT(*) AS denials, ROUND(SUM(denied_amount), 2) AS denied_amount
FROM denials
GROUP BY recommended_path
ORDER BY denials DESC;
