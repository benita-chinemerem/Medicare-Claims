-- =============================================================
-- 02_create_analytics_tables.sql
-- Analytics schema: typed, cleaned, indexed tables derived from
-- staging. Carrier claims are partitioned by claim year.
-- This is the layer that feature engineering queries run against.
-- =============================================================

CREATE SCHEMA IF NOT EXISTS analytics;

-- -------------------------------------------------------------
-- Beneficiary Summary (cleaned, typed)
-- -------------------------------------------------------------
DROP TABLE IF EXISTS analytics.beneficiary_summary CASCADE;
CREATE TABLE analytics.beneficiary_summary (
    desynpuf_id              VARCHAR(50)  NOT NULL,
    year                     SMALLINT     NOT NULL,
    sample_id                SMALLINT     NOT NULL,
    birth_date               DATE,
    death_date               DATE,
    sex_cd                   CHAR(1),     -- 1=Male, 2=Female
    race_cd                  CHAR(1),
    esrd_ind                 CHAR(1),
    state_code               VARCHAR(5),
    county_cd                VARCHAR(10),
    part_a_months            SMALLINT,    -- coverage months in year
    part_b_months            SMALLINT,
    hmo_months               SMALLINT,
    -- Chronic condition flags (1=Yes, 2=No per codebook)
    flag_alzheimer            SMALLINT,
    flag_chf                  SMALLINT,
    flag_chronic_kidney       SMALLINT,
    flag_cancer               SMALLINT,
    flag_copd                 SMALLINT,
    flag_depression           SMALLINT,
    flag_diabetes             SMALLINT,
    flag_ischemic_heart       SMALLINT,
    flag_osteoporosis         SMALLINT,
    flag_ra_oa                SMALLINT,
    flag_stroke               SMALLINT,
    -- Computed: number of chronic conditions present
    chronic_condition_count   SMALLINT    GENERATED ALWAYS AS (
        CASE WHEN flag_alzheimer   = 1 THEN 1 ELSE 0 END +
        CASE WHEN flag_chf         = 1 THEN 1 ELSE 0 END +
        CASE WHEN flag_chronic_kidney = 1 THEN 1 ELSE 0 END +
        CASE WHEN flag_cancer      = 1 THEN 1 ELSE 0 END +
        CASE WHEN flag_copd        = 1 THEN 1 ELSE 0 END +
        CASE WHEN flag_depression  = 1 THEN 1 ELSE 0 END +
        CASE WHEN flag_diabetes    = 1 THEN 1 ELSE 0 END +
        CASE WHEN flag_ischemic_heart = 1 THEN 1 ELSE 0 END +
        CASE WHEN flag_osteoporosis = 1 THEN 1 ELSE 0 END +
        CASE WHEN flag_ra_oa       = 1 THEN 1 ELSE 0 END +
        CASE WHEN flag_stroke      = 1 THEN 1 ELSE 0 END
    ) STORED,
    reimbursement_inpatient  NUMERIC(12,2),
    reimbursement_outpatient NUMERIC(12,2),
    reimbursement_carrier    NUMERIC(12,2),
    PRIMARY KEY (desynpuf_id, year)
);

CREATE INDEX idx_bene_desynpuf_id ON analytics.beneficiary_summary (desynpuf_id);
CREATE INDEX idx_bene_year ON analytics.beneficiary_summary (year);
CREATE INDEX idx_bene_death_date ON analytics.beneficiary_summary (death_date) WHERE death_date IS NOT NULL;

-- -------------------------------------------------------------
-- Carrier Claims (partitioned by year)
-- -------------------------------------------------------------
DROP TABLE IF EXISTS analytics.carrier_claims CASCADE;
CREATE TABLE analytics.carrier_claims (
    clm_id                   VARCHAR(50)   NOT NULL,
    desynpuf_id              VARCHAR(50)   NOT NULL,
    at_physn_npi             VARCHAR(20),  -- primary provider identifier
    clm_from_dt              DATE,
    clm_thru_dt              DATE,
    claim_year               SMALLINT      GENERATED ALWAYS AS (
                                 EXTRACT(YEAR FROM clm_from_dt)::SMALLINT
                             ) STORED,
    clm_pmt_amt              NUMERIC(12,2),
    submitted_charge_amt     NUMERIC(12,2),  -- NCH_CARR_CLM_SBMTD_CHRG_AMT
    allowed_amt              NUMERIC(12,2),  -- NCH_CARR_CLM_ALLWD_AMT
    -- Submitted-to-allowed ratio (key fraud signal)
    submitted_to_allowed_ratio NUMERIC(8,4) GENERATED ALWAYS AS (
        CASE WHEN allowed_amt > 0
             THEN submitted_charge_amt / allowed_amt
             ELSE NULL END
    ) STORED,
    place_of_service_cd      VARCHAR(5),
    nch_clm_type_cd          VARCHAR(5),
    prncpal_dgns_cd          VARCHAR(10),
    -- Primary HCPCS code (line 1) for aggregate features
    primary_hcpcs_cd         VARCHAR(10),
    -- All HCPCS codes as array for distribution analysis
    hcpcs_codes              TEXT[],
    sample_id                SMALLINT,
    is_weekend_claim         BOOLEAN GENERATED ALWAYS AS (
                                 EXTRACT(DOW FROM clm_from_dt) IN (0, 6)
                             ) STORED,
    loaded_at                TIMESTAMP DEFAULT NOW()
) PARTITION BY LIST (claim_year);

CREATE TABLE analytics.carrier_claims_2008 PARTITION OF analytics.carrier_claims FOR VALUES IN (2008);
CREATE TABLE analytics.carrier_claims_2009 PARTITION OF analytics.carrier_claims FOR VALUES IN (2009);
CREATE TABLE analytics.carrier_claims_2010 PARTITION OF analytics.carrier_claims FOR VALUES IN (2010);

CREATE INDEX idx_carrier_npi       ON analytics.carrier_claims (at_physn_npi);
CREATE INDEX idx_carrier_bene      ON analytics.carrier_claims (desynpuf_id);
CREATE INDEX idx_carrier_date      ON analytics.carrier_claims (clm_from_dt);
CREATE INDEX idx_carrier_year      ON analytics.carrier_claims (claim_year);
CREATE INDEX idx_carrier_hcpcs     ON analytics.carrier_claims (primary_hcpcs_cd);
CREATE INDEX idx_carrier_pos       ON analytics.carrier_claims (place_of_service_cd);

-- -------------------------------------------------------------
-- Outpatient Claims
-- -------------------------------------------------------------
DROP TABLE IF EXISTS analytics.outpatient_claims CASCADE;
CREATE TABLE analytics.outpatient_claims (
    clm_id                   VARCHAR(50)  NOT NULL,
    desynpuf_id              VARCHAR(50)  NOT NULL,
    prvdr_num                VARCHAR(20),
    at_physn_npi             VARCHAR(20),
    clm_from_dt              DATE,
    clm_thru_dt              DATE,
    claim_year               SMALLINT     GENERATED ALWAYS AS (
                                 EXTRACT(YEAR FROM clm_from_dt)::SMALLINT
                             ) STORED,
    clm_pmt_amt              NUMERIC(12,2),
    clm_fac_type_cd          VARCHAR(5),
    prncpal_dgns_cd          VARCHAR(10),
    hcpcs_codes              TEXT[],
    revenue_codes            TEXT[],
    is_weekend_claim         BOOLEAN GENERATED ALWAYS AS (
                                 EXTRACT(DOW FROM clm_from_dt) IN (0, 6)
                             ) STORED,
    sample_id                SMALLINT,
    loaded_at                TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (clm_id)
);

CREATE INDEX idx_op_npi       ON analytics.outpatient_claims (at_physn_npi);
CREATE INDEX idx_op_bene      ON analytics.outpatient_claims (desynpuf_id);
CREATE INDEX idx_op_date      ON analytics.outpatient_claims (clm_from_dt);
CREATE INDEX idx_op_year      ON analytics.outpatient_claims (claim_year);
CREATE INDEX idx_op_prvdr     ON analytics.outpatient_claims (prvdr_num);

-- Incremental scoring watermark — tracks how far each DAG run has read
DROP TABLE IF EXISTS analytics.scoring_watermark CASCADE;
CREATE TABLE analytics.scoring_watermark (
    claim_type    VARCHAR(20) PRIMARY KEY,
    last_clm_date DATE,
    batch_id      INTEGER,
    updated_at    TIMESTAMP DEFAULT NOW()
);

INSERT INTO analytics.scoring_watermark (claim_type, last_clm_date, batch_id)
VALUES ('carrier', '2008-01-01', 0), ('outpatient', '2008-01-01', 0)
ON CONFLICT DO NOTHING;

COMMENT ON SCHEMA analytics IS
    'Typed, cleaned, indexed tables. Feature engineering and ML queries run here.';
