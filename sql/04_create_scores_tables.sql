-- =============================================================
-- 04_create_scores_tables.sql
-- Output layer: risk scores, SHAP explanations, model registry.
-- These tables feed the Power BI dashboard directly.
-- =============================================================

CREATE SCHEMA IF NOT EXISTS scores;

-- -------------------------------------------------------------
-- Model Registry
-- Tracks every trained model version, its parameters, and
-- evaluation metrics. DAG 3 writes here; DAG 2 reads the
-- 'promoted' model for scoring.
-- -------------------------------------------------------------
DROP TABLE IF EXISTS scores.model_registry CASCADE;
CREATE TABLE scores.model_registry (
    model_id         SERIAL PRIMARY KEY,
    model_type       VARCHAR(50)  NOT NULL,  -- 'isolation_forest' | 'xgboost'
    model_version    VARCHAR(20)  NOT NULL,
    trained_at       TIMESTAMP    DEFAULT NOW(),
    training_samples INTEGER,
    n_estimators     INTEGER,
    contamination    NUMERIC(6,4),
    -- Evaluation metrics (from held-out or injected-scenario test)
    test_auc_roc     NUMERIC(6,4),
    test_precision   NUMERIC(6,4),
    test_recall      NUMERIC(6,4),
    test_f1          NUMERIC(6,4),
    avg_anomaly_score NUMERIC(8,4),
    -- Promotion status
    is_active        BOOLEAN      DEFAULT FALSE,
    promoted_at      TIMESTAMP,
    notes            TEXT,
    model_path       VARCHAR(500) -- path to serialised .pkl file
);

-- Ensure exactly one active model per type at a time
CREATE UNIQUE INDEX idx_model_active ON scores.model_registry (model_type)
    WHERE is_active = TRUE;

-- -------------------------------------------------------------
-- Provider Risk Scores
-- One row per (npi, scoring_run_id). Dashboard Page 2 reads this.
-- -------------------------------------------------------------
DROP TABLE IF EXISTS scores.provider_risk_scores CASCADE;
CREATE TABLE scores.provider_risk_scores (
    id               SERIAL PRIMARY KEY,
    at_physn_npi     VARCHAR(20)  NOT NULL,
    scoring_run_id   INTEGER      NOT NULL,  -- references scores.scoring_runs.id
    model_id         INTEGER      REFERENCES scores.model_registry (model_id),
    scored_at        TIMESTAMP    DEFAULT NOW(),
    period_year      SMALLINT,
    batch_id         INTEGER,

    -- Raw model output
    isolation_forest_score NUMERIC(8,6),  -- raw score in [-1, 0]
    risk_score       SMALLINT,            -- normalized 0-100 (higher = riskier)
    risk_decile      SMALLINT,            -- 1-10 (10 = highest risk)
    is_flagged       BOOLEAN,             -- risk_score >= threshold

    -- Top feature contributions (from SHAP; human-readable)
    top_reason_1     TEXT,
    top_reason_2     TEXT,
    top_reason_3     TEXT,

    -- Quick-access key metrics for dashboard display
    total_claims     INTEGER,
    distinct_benes   INTEGER,
    avg_submitted_to_allowed_ratio NUMERIC(8,4),
    duplicate_rate   NUMERIC(8,4),
    pct_weekend_claims NUMERIC(8,4)
);

CREATE INDEX idx_prs_npi        ON scores.provider_risk_scores (at_physn_npi);
CREATE INDEX idx_prs_run        ON scores.provider_risk_scores (scoring_run_id);
CREATE INDEX idx_prs_flagged    ON scores.provider_risk_scores (is_flagged) WHERE is_flagged = TRUE;
CREATE INDEX idx_prs_decile     ON scores.provider_risk_scores (risk_decile);
CREATE INDEX idx_prs_scored_at  ON scores.provider_risk_scores (scored_at);

-- -------------------------------------------------------------
-- SHAP Values (detailed, per provider per feature)
-- Dashboard Page 3 (Claim Explanation) reads this.
-- -------------------------------------------------------------
DROP TABLE IF EXISTS scores.shap_values CASCADE;
CREATE TABLE scores.shap_values (
    id               SERIAL PRIMARY KEY,
    at_physn_npi     VARCHAR(20)  NOT NULL,
    scoring_run_id   INTEGER      NOT NULL,
    feature_name     VARCHAR(100) NOT NULL,
    shap_value       NUMERIC(12,6),
    feature_value    NUMERIC(16,4),       -- actual value of the feature for this provider
    feature_rank     SMALLINT             -- rank by |shap_value| for this provider
);

CREATE INDEX idx_shap_npi  ON scores.shap_values (at_physn_npi);
CREATE INDEX idx_shap_run  ON scores.shap_values (scoring_run_id);
CREATE INDEX idx_shap_feat ON scores.shap_values (feature_name);

-- -------------------------------------------------------------
-- Scoring Runs
-- One row per DAG 2 execution. Dashboard Page 4 reads this for
-- pipeline health monitoring.
-- -------------------------------------------------------------
DROP TABLE IF EXISTS scores.scoring_runs CASCADE;
CREATE TABLE scores.scoring_runs (
    id                  SERIAL PRIMARY KEY,
    dag_run_id          VARCHAR(200),
    run_type            VARCHAR(20)  DEFAULT 'weekly',  -- 'weekly' | 'manual'
    started_at          TIMESTAMP    DEFAULT NOW(),
    completed_at        TIMESTAMP,
    status              VARCHAR(20)  DEFAULT 'running',  -- 'running' | 'success' | 'failed'
    model_id            INTEGER      REFERENCES scores.model_registry (model_id),
    batch_id            INTEGER,
    carrier_claims_processed  INTEGER,
    outpatient_claims_processed INTEGER,
    providers_scored    INTEGER,
    providers_flagged   INTEGER,
    error_message       TEXT
);

-- -------------------------------------------------------------
-- Dashboard summary view (Power BI reads this directly)
-- -------------------------------------------------------------
CREATE OR REPLACE VIEW scores.vw_dashboard_overview AS
SELECT
    sr.id                       AS run_id,
    sr.completed_at             AS run_date,
    sr.carrier_claims_processed + sr.outpatient_claims_processed AS total_claims_processed,
    sr.providers_scored,
    sr.providers_flagged,
    -- At-risk dollar proxy: sum of avg submitted charges for flagged providers
    ROUND(
        SUM(CASE WHEN prs.is_flagged THEN prs.avg_submitted_to_allowed_ratio * 1000 ELSE 0 END)
    , 2) AS at_risk_score_proxy,
    mr.model_version,
    mr.model_type
FROM scores.scoring_runs sr
LEFT JOIN scores.provider_risk_scores prs ON prs.scoring_run_id = sr.id
LEFT JOIN scores.model_registry mr ON mr.model_id = sr.model_id
WHERE sr.status = 'success'
GROUP BY sr.id, sr.completed_at, sr.carrier_claims_processed,
         sr.outpatient_claims_processed, sr.providers_scored,
         sr.providers_flagged, mr.model_version, mr.model_type
ORDER BY sr.completed_at DESC;

COMMENT ON SCHEMA scores IS
    'Risk scores, SHAP values, and model registry. Direct source for the Power BI dashboard.';
