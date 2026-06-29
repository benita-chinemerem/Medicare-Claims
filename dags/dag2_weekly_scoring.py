"""
dag2_weekly_scoring.py

Recurring weekly scoring DAG. Scheduled every Sunday at 02:00 UTC.

This DAG is the operational heart of the framework. Each run:
    1. Ingests the next batch of held-back claim records (simulates new weekly feed)
    2. Refreshes provider-level feature aggregations in the features schema
    3. Scores providers using the active Isolation Forest model
    4. Computes SHAP values for all flagged providers
    5. Writes risk scores and reason codes to the scores schema
    6. Updates the scoring_runs log (feeds dashboard Page 4 — Pipeline Health)

Multi-week Airflow run history from this DAG is the primary evidence of
continuous, automated monitoring.
"""

from __future__ import annotations

import os
import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.operators.empty import EmptyOperator

log = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "fraud-anomaly-ai",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
}

DB_CONN_STR = (
    f"postgresql+psycopg2://{os.environ.get('FRAUD_DB_USER', 'airflow')}:"
    f"{os.environ.get('FRAUD_DB_PASSWORD', 'airflow')}@"
    f"{os.environ.get('FRAUD_DB_HOST', 'postgres')}:"
    f"{os.environ.get('FRAUD_DB_PORT', '5432')}/"
    f"{os.environ.get('FRAUD_DB_NAME', 'fraud_claims')}"
)

WEEKLY_BATCH_SIZE  = int(os.environ.get("WEEKLY_BATCH_SIZE", 500_000))
RISK_THRESHOLD     = int(os.environ.get("RISK_SCORE_THRESHOLD", 75))
MODEL_PATH         = os.environ.get("MODEL_PATH", "/opt/airflow/models")
CONTAMINATION      = float(os.environ.get("CONTAMINATION_RATE", 0.05))


# ---------------------------------------------------------------------------
# Task functions
# ---------------------------------------------------------------------------

def create_scoring_run(**context) -> int:
    """
    Ensures all reporting schemas/tables exist, handles system seeding, 
    and opens or updates a scoring_run log session cleanly.
    """
    from sqlalchemy import create_engine, text

    engine = create_engine(DB_CONN_STR)
    dag_run_id = context["run_id"]

    # SELF-HEALING DATABASE INITIALIZATION
    log.info("Running database guards to verify operational tables...")
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS scores;"))
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS analytics;"))

        # Initialize core run telemetry tracking
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS scores.scoring_runs (
                id SERIAL PRIMARY KEY,
                dag_run_id VARCHAR(255) NOT NULL UNIQUE,
                run_type VARCHAR(50) NOT NULL,
                status VARCHAR(50) DEFAULT 'running',
                model_id INT,
                batch_id INT,
                carrier_claims_processed INT,
                providers_scored INT,
                providers_flagged INT,
                error_message TEXT,
                execution_date TIMESTAMP DEFAULT NOW(),
                completed_at TIMESTAMP
            );
        """))

        # Initialize mock ML registry tracking structures
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS scores.model_registry (
                model_id SERIAL PRIMARY KEY,
                model_type VARCHAR(50) NOT NULL,
                is_active BOOLEAN DEFAULT FALSE,
                registered_at TIMESTAMP DEFAULT NOW()
            );
        """))

        # Initialize score evaluation output targets
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS scores.provider_risk_scores (
                id SERIAL PRIMARY KEY,
                scoring_run_id INT REFERENCES scores.scoring_runs(id),
                at_physn_npi VARCHAR(50),
                risk_score FLOAT,
                batch_id INT
            );
        """))

        # Initialize and bootstrap processing stream counters
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS analytics.scoring_watermark (
                claim_type VARCHAR(50) PRIMARY KEY,
                last_clm_date DATE,
                batch_id INT,
                updated_at TIMESTAMP DEFAULT NOW()
            );
        """))

        # Seed initial incremental pipeline state tracker if empty
        conn.execute(text("""
            INSERT INTO analytics.scoring_watermark (claim_type, last_clm_date, batch_id)
            VALUES ('carrier', '1970-01-01', 0)
            ON CONFLICT (claim_type) DO NOTHING;
        """))

    log.info("Database schemas ready. Opening scoring log entry...")
    with engine.begin() as conn:
        result = conn.execute(
            text("""
                INSERT INTO scores.scoring_runs (dag_run_id, run_type)
                VALUES (:dag_run_id, 'weekly')
                ON CONFLICT (dag_run_id) DO UPDATE 
                SET status = 'running'
                RETURNING id
            """),
            {"dag_run_id": dag_run_id},
        )
        run_id = result.scalar()

    log.info("Created or recycled scoring run session token: %d", run_id)
    context["ti"].xcom_push(key="scoring_run_id", value=run_id)
    return run_id


def ingest_batch(**context):
    """
    Reads the next WEEKLY_BATCH_SIZE carrier claims not yet processed
    (using the scoring_watermark table to track position) and marks them
    as belonging to this scoring run.
    """
    from sqlalchemy import create_engine, text
    import pandas as pd

    engine = create_engine(DB_CONN_STR)
    run_id = context["ti"].xcom_pull(key="scoring_run_id")

    with engine.begin() as conn:
        result = conn.execute(
            text("SELECT last_clm_date, batch_id FROM analytics.scoring_watermark WHERE claim_type = 'carrier'")
        )
        row = result.fetchone()
        last_date = row[0]
        batch_id  = (row[1] or 0) + 1

        # Fixed: Explicitly CAST clm_from_dt as a DATE to bypass string mismatch errors
        df = pd.read_sql(
            text("""
                SELECT at_physn_npi, desynpuf_id, clm_from_dt, clm_id,
                       submitted_charge_amt, allowed_amt,
                       submitted_to_allowed_ratio, primary_hcpcs_cd,
                       place_of_service_cd,
                       (EXTRACT(ISODOW FROM CAST(clm_from_dt AS DATE)) IN (6, 7)) AS is_weekend_claim
                FROM analytics.carrier_claims
                WHERE CAST(clm_from_dt AS DATE) > :last_date
                ORDER BY CAST(clm_from_dt AS DATE) ASC
                LIMIT :batch_size
            """),
            conn,
            params={"last_date": last_date, "batch_size": WEEKLY_BATCH_SIZE},
        )

        if df.empty:
            log.info("No new carrier claims to process in this batch.")
            context["ti"].xcom_push(key="batch_id", value=batch_id)
            context["ti"].xcom_push(key="carrier_count", value=0)
            return

        new_watermark = df["clm_from_dt"].max()
        conn.execute(
            text("""
                UPDATE analytics.scoring_watermark
                SET last_clm_date = :new_date, batch_id = :bid, updated_at = NOW()
                WHERE claim_type = 'carrier'
            """),
            {"new_date": new_watermark, "bid": batch_id},
        )

    context["ti"].xcom_push(key="batch_id", value=batch_id)
    context["ti"].xcom_push(key="carrier_count", value=len(df))
    log.info("Ingested %d carrier claims up to %s (batch %d)", len(df), new_watermark, batch_id)


def check_batch_has_data(**context) -> bool:
    """Short-circuit: skip scoring tasks if no new data was ingested."""
    carrier_count = context["ti"].xcom_pull(key="carrier_count")
    if not carrier_count:
        log.info("Empty batch — short-circuiting downstream scoring tasks.")
        return False
    return True


def refresh_provider_features(**context):
    """
    Calls the feature engineering module to recompute provider-level
    feature vectors for all providers active in the current batch.
    """
    import sys
    sys.path.insert(0, "/opt/airflow/scripts")
    from etl.feature_engineering import compute_provider_features

    batch_id = context["ti"].xcom_pull(key="batch_id")
    compute_provider_features(db_conn_str=DB_CONN_STR, batch_id=batch_id)
    log.info("Provider features refreshed for batch %d.", batch_id)


def score_providers(**context):
    """
    Loads the active Isolation Forest model and scores all providers
    in the current batch.
    """
    import sys
    sys.path.insert(0, "/opt/airflow/ml")
    from isolation_forest import score_providers_batch

    run_id   = context["ti"].xcom_pull(key="scoring_run_id")
    batch_id = context["ti"].xcom_pull(key="batch_id")

    flagged_count = score_providers_batch(
        db_conn_str=DB_CONN_STR,
        scoring_run_id=run_id,
        batch_id=batch_id,
        risk_threshold=RISK_THRESHOLD,
        model_path=MODEL_PATH,
    )

    context["ti"].xcom_push(key="flagged_count", value=flagged_count)
    log.info("Scored providers. %d flagged (risk_score >= %d).", flagged_count, RISK_THRESHOLD)


def compute_shap_values(**context):
    """
    Generates SHAP values for all flagged providers in this run.
    """
    import sys
    sys.path.insert(0, "/opt/airflow/ml")
    from shap_explainer import explain_flagged_providers

    run_id = context["ti"].xcom_pull(key="scoring_run_id")
    explain_flagged_providers(
        db_conn_str=DB_CONN_STR,
        scoring_run_id=run_id,
        model_path=MODEL_PATH,
    )
    log.info("SHAP values computed for scoring run %d.", run_id)


def close_scoring_run(**context):
    """
    Marks the scoring run as complete and records summary statistics.
    """
    from sqlalchemy import create_engine, text

    engine = create_engine(DB_CONN_STR)
    run_id        = context["ti"].xcom_pull(key="scoring_run_id")
    carrier_count = context["ti"].xcom_pull(key="carrier_count") or 0
    flagged_count = context["ti"].xcom_pull(key="flagged_count") or 0

    with engine.begin() as conn:
        result = conn.execute(
            text("SELECT model_id FROM scores.model_registry WHERE model_type = 'isolation_forest' AND is_active = TRUE")
        )
        model_row = result.fetchone()
        model_id = model_row[0] if model_row else None

        providers_scored = conn.execute(
            text("SELECT COUNT(DISTINCT at_physn_npi) FROM scores.provider_risk_scores WHERE scoring_run_id = :rid"),
            {"rid": run_id},
        ).scalar()

        conn.execute(
            text("""
                UPDATE scores.scoring_runs SET
                    completed_at = NOW(),
                    status = 'success',
                    model_id = :mid,
                    batch_id = :bid,
                    carrier_claims_processed = :cc,
                    providers_scored = :ps,
                    providers_flagged = :pf
                WHERE id = :rid
            """),
            {
                "mid": model_id,
                "bid": context["ti"].xcom_pull(key="batch_id"),
                "cc":  carrier_count,
                "ps":  providers_scored,
                "pf":  flagged_count,
                "rid": run_id,
            },
        )

    log.info(
        "Scoring run %d complete. Providers scored: %d | Flagged: %d",
        run_id, providers_scored or 0, flagged_count
    )


def mark_run_failed(**context):
    """Marks the scoring run as failed if any upstream task errors."""
    from sqlalchemy import create_engine, text

    engine  = create_engine(DB_CONN_STR)
    run_id  = context["ti"].xcom_pull(key="scoring_run_id")
    err_msg = str(context.get("exception", "Unknown error"))

    if run_id:
        with engine.begin() as conn:
            conn.execute(
                text("""
                    UPDATE scores.scoring_runs
                    SET completed_at = NOW(), status = 'failed', error_message = :err
                    WHERE id = :rid
                """),
                {"err": err_msg[:500], "rid": run_id},
            )


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id="dag2_weekly_scoring",
    default_args=DEFAULT_ARGS,
    description=(
        "Weekly recurring scoring DAG. Ingests the next claim batch, refreshes "
        "provider features, scores with Isolation Forest, computes SHAP explanations, "
        "and updates the dashboard."
    ),
    schedule_interval="0 2 * * 0",   # Every Sunday at 02:00 UTC
    start_date=datetime(2024, 1, 7),
    catchup=False,
    tags=["fraud-anomaly-ai", "scoring", "weekly"],
    doc_md=__doc__,
    on_failure_callback=mark_run_failed,
) as dag:

    start = EmptyOperator(task_id="start")

    t_create_run = PythonOperator(
        task_id="create_scoring_run",
        python_callable=create_scoring_run,
    )

    t_ingest = PythonOperator(
        task_id="ingest_batch",
        python_callable=ingest_batch,
    )

    t_check = ShortCircuitOperator(
        task_id="check_batch_has_data",
        python_callable=check_batch_has_data,
        doc_md="Short-circuits the DAG if no new claims were ingested this run.",
    )

    t_features = PythonOperator(
        task_id="refresh_provider_features",
        python_callable=refresh_provider_features,
    )

    t_score = PythonOperator(
        task_id="score_providers",
        python_callable=score_providers,
    )

    t_shap = PythonOperator(
        task_id="compute_shap_values",
        python_callable=compute_shap_values,
    )

    t_close = PythonOperator(
        task_id="close_scoring_run",
        python_callable=close_scoring_run,
        trigger_rule="all_done",
    )

    end = EmptyOperator(task_id="scoring_complete")

    (
        start
        >> t_create_run
        >> t_ingest
        >> t_check
        >> t_features
        >> t_score
        >> t_shap
        >> t_close
        >> end
    )