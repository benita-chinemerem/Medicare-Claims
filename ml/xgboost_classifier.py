"""
ml/xgboost_classifier.py

Supervised XGBoost fraud-detection model trained on the labeled dataset
produced by inject_anomalies.run_injection().

This module demonstrates the supervised architecture (Section 6.4 of the
project spec) that would be used in production when historical investigation
outcomes are available as training labels. Here, labels come from three
controlled injection scenarios: upcoding, phantom billing, and duplicate
submission.

Outputs:
  - Trained XGBoost model bundle saved to MODEL_PATH/xgboost_supervised_YYYYMMDD.pkl
  - Per-scenario Precision / Recall / F1 written to scores.supervised_model_metrics
  - Metrics pushed via XCom for display in DAG 3 logs

Called by dag3_monthly_retraining.py → run_supervised_demo task.
"""

from __future__ import annotations

import logging
import os
import pickle
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, text
import xgboost as xgb

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── Feature columns ───────────────────────────────────────────────────────────
# Uses the subset of features that are directly affected by the three
# injection types. All 19 IF features are included so XGBoost sees the
# same view of each provider that Isolation Forest does.
SUPERVISED_FEATURE_COLUMNS = [
    "total_carrier_claims",
    "distinct_beneficiaries",
    "distinct_hcpcs_codes",
    "avg_submitted_to_allowed_ratio",
    "p95_submitted_to_allowed_ratio",
    "pct_weekend_claims",
    "max_claims_in_single_day",
    "avg_claims_per_beneficiary",
    "top_hcpcs_code_share",
    "hcpcs_concentration_score",
    "exact_duplicate_count",
    "near_duplicate_count",
    "duplicate_rate",
]

# ── Model hyperparameters ─────────────────────────────────────────────────────
# scale_pos_weight = (n_negative / n_positive) ≈ (0.85 / 0.15) ≈ 5.7
# Adjusted because three anomaly groups together = 15% of all providers.
XGB_PARAMS = {
    "n_estimators":      300,
    "max_depth":         4,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "scale_pos_weight":  5,
    "eval_metric":       "logloss",
    "use_label_encoder": False,
    "random_state":      42,
    "n_jobs":            1,      # forced to 1 — multiprocessing disabled in Airflow
}

TEST_SIZE   = 0.20
RANDOM_SEED = 42
# ─────────────────────────────────────────────────────────────────────────────


def _ensure_metrics_table(engine):
    """Create scores.supervised_model_metrics if it does not exist."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS scores.supervised_model_metrics (
                id              SERIAL PRIMARY KEY,
                model_version   VARCHAR(50)  NOT NULL,
                scenario        VARCHAR(50)  NOT NULL,
                precision_score FLOAT,
                recall_score    FLOAT,
                f1_score        FLOAT,
                auc_roc         FLOAT,
                support         INT,
                trained_at      TIMESTAMP DEFAULT NOW()
            );
        """))


def train_xgboost(
    labeled_df:  pd.DataFrame,
    model_path:  str,
    db_conn_str: str,
    version_tag: Optional[str] = None,
) -> Dict[str, Dict]:
    """
    Train XGBoost on the labeled injected dataset.
    Evaluates per-scenario Precision, Recall, F1, and AUC-ROC.
    Saves model bundle to disk and writes metrics to PostgreSQL.

    Args:
        labeled_df:   Output of inject_anomalies.run_injection().
                      Must contain feature columns + 'anomaly_label' +
                      'anomaly_scenario'.
        model_path:   Directory to save the model .pkl file.
        db_conn_str:  SQLAlchemy connection string for fraud_claims DB.
        version_tag:  Optional version string (defaults to YYYYMMDD today).

    Returns:
        Dict mapping scenario name → {precision, recall, f1, auc_roc, support}.
    """
    version = version_tag or datetime.now().strftime("%Y%m%d")

    # ── 1. Build feature matrix ───────────────────────────────────────────────
    available_cols = [c for c in SUPERVISED_FEATURE_COLUMNS if c in labeled_df.columns]
    missing = set(SUPERVISED_FEATURE_COLUMNS) - set(available_cols)
    if missing:
        log.warning("Features missing from labeled_df (will skip): %s", missing)

    X         = labeled_df[available_cols].fillna(0).values.astype(float)
    y         = labeled_df["anomaly_label"].values.astype(int)
    scenarios = labeled_df["anomaly_scenario"].values

    log.info(
        "XGBoost training — %d samples | %d features | %.1f%% positive",
        len(X), len(available_cols), y.mean() * 100,
    )

    # ── 2. Train / test split (stratified) ───────────────────────────────────
    (X_train, X_test,
     y_train, y_test,
     sc_train, sc_test) = train_test_split(
        X, y, scenarios,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    # ── 3. Feature scaling ────────────────────────────────────────────────────
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── 4. Train ──────────────────────────────────────────────────────────────
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    log.info("XGBoost training complete.")

    # ── 5. Overall evaluation ─────────────────────────────────────────────────
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    overall = classification_report(y_test, y_pred, output_dict=True)
    try:
        overall_auc = roc_auc_score(y_test, y_pred_prob)
    except Exception:
        overall_auc = None

    log.info(
        "Overall  — Precision: %.4f | Recall: %.4f | F1: %.4f | AUC-ROC: %s",
        overall.get("1", {}).get("precision", 0),
        overall.get("1", {}).get("recall", 0),
        overall.get("1", {}).get("f1-score", 0),
        f"{overall_auc:.4f}" if overall_auc else "n/a",
    )

    # ── 6. Per-scenario evaluation ────────────────────────────────────────────
    # For each scenario we evaluate on:
    #   - all clean test rows  (sc_test == "clean")  as negatives
    #   - that scenario's test rows as positives
    # This gives a meaningful binary classifier view per fraud type.
    scenario_metrics: Dict[str, Dict] = {}

    for scenario in ["upcoding", "phantom", "duplicate"]:
        mask = (sc_test == scenario) | (sc_test == "clean")
        if mask.sum() == 0:
            log.warning("No test samples for scenario '%s'. Skipping.", scenario)
            continue

        y_sc   = (sc_test[mask] == scenario).astype(int)
        yp_sc  = model.predict(X_test[mask])
        yp_prob_sc = model.predict_proba(X_test[mask])[:, 1]

        p, r, f, sup = precision_recall_fscore_support(
            y_sc, yp_sc, average="binary", zero_division=0
        )
        try:
            auc = float(roc_auc_score(y_sc, yp_prob_sc))
        except Exception:
            auc = None

        scenario_metrics[scenario] = {
            "precision": round(float(p), 4),
            "recall":    round(float(r), 4),
            "f1":        round(float(f), 4),
            "auc_roc":   round(auc, 4) if auc is not None else None,
            "support":   int(sup),
        }
        log.info(
            "Scenario %-22s | Precision: %.4f | Recall: %.4f | F1: %.4f "
            "| AUC-ROC: %s | Support: %d",
            scenario,
            p, r, f,
            f"{auc:.4f}" if auc else "n/a",
            sup,
        )

    # ── 7. Feature importance ─────────────────────────────────────────────────
    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        "feature":    available_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False)
    log.info("Top-5 XGBoost features by importance:\n%s", fi_df.head(5).to_string(index=False))

    # ── 8. Save model bundle ──────────────────────────────────────────────────
    os.makedirs(model_path, exist_ok=True)
    model_file = os.path.join(model_path, f"xgboost_supervised_{version}.pkl")
    bundle = {
        "model":            model,
        "scaler":           scaler,
        "feature_columns":  available_cols,
        "model_version":    f"xgb_{version}",
        "scenario_metrics": scenario_metrics,
        "feature_importance": fi_df.to_dict("records"),
    }
    with open(model_file, "wb") as fh:
        pickle.dump(bundle, fh)
    log.info("XGBoost bundle saved: %s", model_file)

    # ── 9. Write metrics to PostgreSQL ────────────────────────────────────────
    engine = create_engine(db_conn_str)
    _ensure_metrics_table(engine)

    rows = [
        {
            "model_version":   f"xgb_{version}",
            "scenario":        scenario,
            "precision_score": m["precision"],
            "recall_score":    m["recall"],
            "f1_score":        m["f1"],
            "auc_roc":         m["auc_roc"],
            "support":         m["support"],
        }
        for scenario, m in scenario_metrics.items()
    ]
    if rows:
        pd.DataFrame(rows).to_sql(
            "supervised_model_metrics",
            engine,
            schema="scores",
            if_exists="append",
            index=False,
            method="multi",
        )
        log.info(
            "Metrics written to scores.supervised_model_metrics (%d rows).", len(rows)
        )

    return scenario_metrics


if __name__ == "__main__":
    """Quick smoke-test: run injection + training locally."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from etl.inject_anomalies import run_injection

    conn_str = (
        f"postgresql+psycopg2://{os.environ.get('FRAUD_DB_USER','airflow')}:"
        f"{os.environ.get('FRAUD_DB_PASSWORD','airflow')}@"
        f"{os.environ.get('FRAUD_DB_HOST','localhost')}:5432/"
        f"{os.environ.get('FRAUD_DB_NAME','fraud_claims')}"
    )
    model_dir = os.environ.get("MODEL_PATH", "/opt/airflow/models")

    labeled = run_injection(conn_str)
    metrics = train_xgboost(labeled, model_dir, conn_str)
    for s, m in metrics.items():
        print(f"{s:20s}  P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}")
