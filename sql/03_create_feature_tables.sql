-- =============================================================
-- 03_create_feature_tables.sql
-- Provider-level feature vectors. Populated by the Python
-- feature engineering scripts. One row per provider per period.
-- =============================================================

CREATE SCHEMA IF NOT EXISTS features;

-- -------------------------------------------------------------
-- Provider Feature Vectors
-- One row per (npi, period_year). Refreshed by DAG 2 each week.
-- -------------------------------------------------------------
DROP TABLE IF EXISTS features.provider_features CASCADE;
CREATE TABLE features.provider_features (
    id                           SERIAL PRIMARY KEY,
    at_physn_npi                 VARCHAR(20)  NOT NULL,
    period_year                  SMALLINT     NOT NULL,
    computed_at                  TIMESTAMP    DEFAULT NOW(),
    batch_id                     INTEGER,

    -- Volume and velocity
    total_carrier_claims         INTEGER,
    carrier_claims_per_bene      NUMERIC(10,4),
    prior_period_claim_count     INTEGER,
    claim_volume_growth_pct      NUMERIC(8,4),  -- % change vs prior period

    -- HCPCS code distribution
    distinct_hcpcs_codes         INTEGER,
    top_hcpcs_code               VARCHAR(10),
    top_hcpcs_code_share         NUMERIC(8,4),  -- fraction of claims on top code
    hcpcs_concentration_score    NUMERIC(8,4),  -- Herfindahl index over code dist.

    -- Billing amount features
    avg_submitted_charge         NUMERIC(12,2),
    avg_allowed_amt              NUMERIC(12,2),
    avg_submitted_to_allowed_ratio NUMERIC(8,4),
    p95_submitted_to_allowed_ratio NUMERIC(8,4), -- 95th pct ratio

    -- Beneficiary features
    distinct_beneficiaries       INTEGER,
    avg_claims_per_beneficiary   NUMERIC(8,4),
    beneficiaries_per_state      INTEGER,        -- geographic dispersion proxy
    high_chronic_burden_benes_pct NUMERIC(8,4), -- % benes with 3+ chronic conditions

    -- Place of service
    distinct_pos_codes           INTEGER,
    pct_claims_office            NUMERIC(8,4),   -- POS=11
    pct_claims_home              NUMERIC(8,4),   -- POS=12
    pct_claims_nursing_facility  NUMERIC(8,4),   -- POS=31-32

    -- Temporal patterns
    pct_weekend_claims           NUMERIC(8,4),
    pct_holiday_adjacent_claims  NUMERIC(8,4),
    max_claims_in_single_day     INTEGER,

    -- Duplicate detection
    exact_duplicate_count        INTEGER,
    near_duplicate_count         INTEGER,        -- same bene+code within 3 days
    duplicate_rate               NUMERIC(8,4),   -- duplicates / total claims

    -- Post-death billing
    claims_after_bene_death      INTEGER,
    claims_in_bene_death_year    INTEGER,

    -- Outpatient (if provider appears in outpatient as well)
    total_outpatient_claims      INTEGER,
    avg_outpatient_payment       NUMERIC(12,2),

    UNIQUE (at_physn_npi, period_year)
);

CREATE INDEX idx_pf_npi         ON features.provider_features (at_physn_npi);
CREATE INDEX idx_pf_year        ON features.provider_features (period_year);
CREATE INDEX idx_pf_batch       ON features.provider_features (batch_id);
CREATE INDEX idx_pf_computed    ON features.provider_features (computed_at);

-- -------------------------------------------------------------
-- Claim-Level Duplicate Pairs
-- Identified during feature engineering; used in the dashboard
-- claim explanation view.
-- -------------------------------------------------------------
DROP TABLE IF EXISTS features.duplicate_claim_pairs CASCADE;
CREATE TABLE features.duplicate_claim_pairs (
    id              SERIAL PRIMARY KEY,
    clm_id_a        VARCHAR(50),
    clm_id_b        VARCHAR(50),
    desynpuf_id     VARCHAR(50),
    at_physn_npi    VARCHAR(20),
    hcpcs_cd        VARCHAR(10),
    date_a          DATE,
    date_b          DATE,
    date_diff_days  SMALLINT,
    duplicate_type  VARCHAR(20),  -- 'exact' or 'near'
    detected_at     TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_dup_npi  ON features.duplicate_claim_pairs (at_physn_npi);
CREATE INDEX idx_dup_bene ON features.duplicate_claim_pairs (desynpuf_id);

COMMENT ON SCHEMA features IS
    'Provider-level feature vectors used as input to ML models. Refreshed weekly by DAG 2.';
