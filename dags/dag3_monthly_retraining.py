"""
dag3_monthly_retraining.py

Monthly model retraining DAG. Scheduled for the 1st of each month at 03:00 UTC.

On each run:
    1. Extracts the full current feature vector dataset from features.provider_features
    2. Retrains the Isolation Forest on all accumulated data
    3. Evaluates the new model against the prior model on a held-out feature subset
    4. Logs all metrics to scores.model_registry
    5. Promotes the new model (sets is_active = TRUE) if it outperforms the current one

This DAG demonstrates the retraining discipline that production payer and CMS
contractor integrity systems maintain to keep pace with evolving billing patterns.
"""

from __future__ import annotations

import os
import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago

log = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "fraud-anomaly-ai",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=15),
}

DB_CONN_STR = (
    f"postgresql+psycopg2://{os.environ.get('FRAUD_DB_USER', 'airflow')}:"
    f"{os.environ.get('FRAUD_DB_PASSWORD', 'airflow')}@"
    f"{os.environ.get('FRAUD_DB_HOST', 'postgres')}:"
    f"{os.environ.get('FRAUD_DB_PORT', '5432')}/"
    f"{os.environ.get('FRAUD_DB_NAME', 'fraud_claims')}"
)

MODEL_PATH    = os.environ.get("MODEL_PATH", "/opt/airflow/models")
CONTAMINATION = float(os.environ.get("CONTAMINATION_RATE", 0.05))

# Minimum improvement threshold for model promotion
PROMOTION_IMPROVEMENT_THRESHOLD = 0.01  # 1% improvement in mean anomaly separation


# ---------------------------------------------------------------------------
# Task functions
# ---------------------------------------------------------------------------

def extract_training_features(**context) -> dict:
    """
    Reads all rows from features.provider_features into a held-out split.
    Returns metadata (row count, feature list) via XCom; the actual data
    is written to disk to avoid XCom size limits.
    """
    import pandas as pd
    from sqlalchemy import create_engine, text

    engine = create_engine(DB_CONN_STR)

    df = pd.read_sql(
        text("""
            SELECT *
            FROM features.provider_features
            ORDER BY computed_at ASC
        """),
        engine,
    )

    if df.empty:
        raise ValueError(
            "No features found in features.provider_features. "
            "Run DAG 1 (backfill) and at least one DAG 2 (scoring) cycle first."
        )

    os.makedirs(MODEL_PATH, exist_ok=True)
    train_path = os.path.join(MODEL_PATH, "training_features.parquet")
    df.to_parquet(train_path, index=False)

    log.info("Extracted %d provider-year feature rows for retraining.", len(df))
    context["ti"].xcom_push(key="train_row_count", value=len(df))
    context["ti"].xcom_push(key="train_path", value=train_path)
    return {"rows": len(df), "path": train_path}


def train_new_model(**context):
    """
    Trains a new Isolation Forest on the full feature dataset.
    Serialises the model to disk and registers it in scores.model_registry
    with status is_active = FALSE (pending evaluation).
    """
    import sys
    sys.path.insert(0, "/opt/airflow/ml")
    from isolation_forest import train_isolation_forest
    from model_registry import register_model

    train_path = context["ti"].xcom_pull(key="train_path")
    train_rows = context["ti"].xcom_pull(key="train_row_count")

    model_version = datetime.now().strftime("if_%Y%m%d")
    model_file    = os.path.join(MODEL_PATH, f"isolation_forest_{model_version}.pkl")

    metrics = train_isolation_forest(
        features_path=train_path,
        model_output_path=model_file,
        contamination=CONTAMINATION,
    )

    model_id = register_model(
        db_conn_str=DB_CONN_STR,
        model_type="isolation_forest",
        model_version=model_version,
        training_samples=train_rows,
        n_estimators=metrics["n_estimators"],
        contamination=CONTAMINATION,
        avg_anomaly_score=metrics["avg_anomaly_score"],
        model_path=model_file,
        is_active=False,  # not yet promoted
    )

    log.info("Trained model %s (id=%d). Avg anomaly score: %.4f",
             model_version, model_id, metrics["avg_anomaly_score"])
    context["ti"].xcom_push(key="new_model_id", value=model_id)
    context["ti"].xcom_push(key="new_avg_score", value=metrics["avg_anomaly_score"])
    context["ti"].xcom_push(key="model_version", value=model_version)


def evaluate_and_decide(**context) -> str:
    """
    Compares the new model to the currently active model using the held-out
    feature subset. Returns the task_id for the next branch:
      - 'promote_new_model' if the new model is better
      - 'keep_current_model' otherwise
    """
    import sys
    sys.path.insert(0, "/opt/airflow/ml")
    from isolation_forest import evaluate_model_comparison
    from model_registry import get_active_model_id

    new_model_id   = context["ti"].xcom_pull(key="new_model_id")
    train_path     = context["ti"].xcom_pull(key="train_path")
    current_model_id = get_active_model_id(DB_CONN_STR, model_type="isolation_forest")

    if current_model_id is None:
        # No active model exists yet; always promote the new one
        log.info("No active model found. Promoting new model %d automatically.", new_model_id)
        context["ti"].xcom_push(key="promote_reason", value="first_model")
        return "promote_new_model"

    improvement = evaluate_model_comparison(
        db_conn_str=DB_CONN_STR,
        new_model_id=new_model_id,
        current_model_id=current_model_id,
        features_path=train_path,
        model_path=MODEL_PATH,
    )

    log.info("Model comparison: improvement = %.4f (threshold = %.4f)",
             improvement, PROMOTION_IMPROVEMENT_THRESHOLD)

    if improvement >= PROMOTION_IMPROVEMENT_THRESHOLD:
        context["ti"].xcom_push(key="promote_reason", value=f"improvement_{improvement:.4f}")
        return "promote_new_model"
    else:
        context["ti"].xcom_push(key="promote_reason", value=f"no_improvement_{improvement:.4f}")
        return "keep_current_model"


def promote_new_model(**context):
    """
    Sets the new model as is_active = TRUE and deactivates the prior model.
    """
    import sys
    sys.path.insert(0, "/opt/airflow/ml")
    from model_registry import promote_model

    new_model_id = context["ti"].xcom_pull(key="new_model_id")
    reason       = context["ti"].xcom_pull(key="promote_reason")

    promote_model(
        db_conn_str=DB_CONN_STR,
        model_id=new_model_id,
        model_type="isolation_forest",
        notes=f"Promoted on {datetime.now().date()}. Reason: {reason}",
    )
    log.info("Model %d promoted to active.", new_model_id)


def keep_current_model(**context):
    """No-op: logs the decision to keep the existing model."""
    reason = context["ti"].xcom_pull(key="promote_reason")
    log.info(
        "Keeping current active model. New model did not meet improvement threshold. "
        "Reason: %s", reason
    )


def log_retraining_summary(**context):
    """Writes a summary of the retraining run to the model registry notes."""
    from sqlalchemy import create_engine, text

    engine = create_engine(DB_CONN_STR)
    new_model_id  = context["ti"].xcom_pull(key="new_model_id")
    model_version = context["ti"].xcom_pull(key="model_version")
    train_rows    = context["ti"].xcom_pull(key="train_row_count")

    summary = (
        f"Retraining run {datetime.now().isoformat()}. "
        f"Version: {model_version}. "
        f"Training rows: {train_rows}."
    )

    with engine.connect() as conn:
        conn.execute(
            text("UPDATE scores.model_registry SET notes = :n WHERE model_id = :mid"),
            {"n": summary, "mid": new_model_id},
        )
        conn.commit()

    log.info("Retraining summary logged: %s", summary)


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id="dag3_monthly_retraining",
    default_args=DEFAULT_ARGS,
    description=(
        "Monthly model retraining. Retrains Isolation Forest on accumulated feature data, "
        "evaluates against current model, and promotes if performance improves."
    ),
    schedule_interval="0 3 1 * *",   # 1st of every month at 03:00 UTC
    start_date=datetime(2024, 2, 1),
    catchup=False,
    tags=["fraud-anomaly-ai", "retraining", "monthly"],
    doc_md=__doc__,
) as dag:

    start = EmptyOperator(task_id="start")

    t_extract = PythonOperator(
        task_id="extract_training_features",
        python_callable=extract_training_features,
    )

    t_train = PythonOperator(
        task_id="train_new_model",
        python_callable=train_new_model,
    )

    t_evaluate = BranchPythonOperator(
        task_id="evaluate_and_decide",
        python_callable=evaluate_and_decide,
        doc_md="Compares new vs current model. Branches to promote or keep.",
    )

    t_promote = PythonOperator(
        task_id="promote_new_model",
        python_callable=promote_new_model,
    )

    t_keep = PythonOperator(
        task_id="keep_current_model",
        python_callable=keep_current_model,
    )

    t_summary = PythonOperator(
        task_id="log_retraining_summary",
        python_callable=log_retraining_summary,
        trigger_rule="none_failed_min_one_success",
    )

    end = EmptyOperator(task_id="retraining_complete", trigger_rule="none_failed_min_one_success")

    (
        start
        >> t_extract
        >> t_train
        >> t_evaluate
        >> [t_promote, t_keep]
        >> t_summary
        >> end
    )
