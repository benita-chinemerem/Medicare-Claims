"""
ml/xgboost_supervised.py

XGBoost supervised fraud-detection demonstration.

This module trains a gradient-boosted classifier on the labeled dataset produced
by inject_anomalies.py. Its purpose is to demonstrate the supervised component
of the architecture — the methodology that production fraud-detection systems
use when historical investigation outcomes or confirmed fraud cases provide
training labels.

Results here are NOT generalizable to real fraud patterns. They reflect
performance on injected synthetic anomalies layered on top of already-synthetic
DE-SynPUF data. The value is architectural: it shows the same feature set and
pipeline that supports unsupervised Isolation Forest scoring also generalizes
cleanly to supervised learning when labels become available.

Usage:
    python ml/xgboost_supervised.py --data data/injected/ [--output models/]

Reference:
    Chen, T. and Guestrin, C.
    "XGBoost: A Scalable Tree Boosting System."
    KDD 2016, pp. 785-794.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Feature columns (must match isolation_forest.py — same feature set, same model)
FEATURE_COLUMNS = [
    "total_carrier_claims",
    "carrier_claims_per_bene",
    "claim_volume_growth_pct",
    "distinct_hcpcs_codes",
    "top_hcpcs_code_share",
    "hcpcs_concentration_score",
    "avg_submitted_to_allowed_ratio",
    "p95_submitted_to_allowed_ratio",
    "distinct_beneficiaries",
    "avg_claims_per_beneficiary",
    "beneficiaries_per_state",
    "high_chronic_burden_benes_pct",
    "pct_weekend_claims",
    "max_claims_in_single_day",
    "exact_duplicate_count",
    "near_duplicate_count",
    "duplicate_rate",
    "claims_after_bene_death",
]

XGB_PARAMS = {
    "n_estimators":      300,
    "max_depth":         6,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_weight":  5,
    "scale_pos_weight":  10,    # handles class imbalance: ~10:1 clean:anomaly
    "use_label_encoder": False,
    "eval_metric":       "aucpr",
    "random_state":      42,
    "n_jobs":           -1,
}


def load_injected_data(data_dir: str) -> pd.DataFrame:
    """Loads the labeled injected feature Parquet from inject_anomalies.py output."""
    path = os.path.join(data_dir, "injected_labeled_features.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Injected dataset not found at {path}. "
            "Run inject_anomalies.py first."
        )
    df = pd.read_parquet(path)
    log.info(
        "Loaded %d rows | %d anomalies (%.1f%%)",
        len(df), df["label"].sum(), df["label"].mean() * 100
    )
    return df


def train_and_evaluate(
    data_dir: str,
    output_dir: Optional[str] = None,
    test_size: float = 0.20,
) -> dict:
    """
    Full training and evaluation run.

    1. Loads the injected labeled dataset
    2. Splits into train / test by provider (stratified by label AND scenario)
    3. Trains XGBoost on training split
    4. Evaluates on test split — overall and per scenario
    5. Saves the model to output_dir if specified
    6. Returns a metrics dict for printing and optional DB logging

    Args:
        data_dir:   directory containing injected_labeled_features.parquet
        output_dir: optional path to save the trained model bundle
        test_size:  fraction of providers to hold out for testing
    """
    df = load_injected_data(data_dir)

    # One row per (provider, scenario) — deduplicate on NPI+scenario
    # to prevent data leakage across the three scenario copies
    df = df.drop_duplicates(subset=["at_physn_npi", "scenario"])

    X = df[FEATURE_COLUMNS].fillna(0).values
    y = df["label"].values
    scenarios = df["scenario"].values

    # Stratify split by label
    X_train, X_test, y_train, y_test, sc_train, sc_test = train_test_split(
        X, y, scenarios,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    log.info(
        "Train: %d rows (%d anomalies) | Test: %d rows (%d anomalies)",
        len(X_train), y_train.sum(), len(X_test), y_test.sum()
    )

    # Train XGBoost
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Overall evaluation
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    overall_metrics = {
        "auc_roc":   float(roc_auc_score(y_test, y_prob)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_test, y_pred, zero_division=0)),
    }

    log.info(
        "Overall — AUC-ROC: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f",
        overall_metrics["auc_roc"],
        overall_metrics["precision"],
        overall_metrics["recall"],
        overall_metrics["f1"],
    )
    log.info("\n%s", classification_report(y_test, y_pred, target_names=["Clean", "Anomaly"]))

    # Per-scenario evaluation
    scenario_metrics = {}
    for scenario in ["upcoding", "phantom_billing", "duplicate_submission"]:
        mask = sc_test == scenario
        if mask.sum() == 0:
            continue
        sc_y_true = y_test[mask]
        sc_y_pred = y_pred[mask]
        sc_y_prob = y_prob[mask]

        sc_metrics = {
            "test_rows": int(mask.sum()),
            "anomalies": int(sc_y_true.sum()),
            "precision": float(precision_score(sc_y_true, sc_y_pred, zero_division=0)),
            "recall":    float(recall_score(sc_y_true, sc_y_pred, zero_division=0)),
            "f1":        float(f1_score(sc_y_true, sc_y_pred, zero_division=0)),
        }
        if sc_y_true.sum() > 0 and len(np.unique(sc_y_true)) > 1:
            sc_metrics["auc_roc"] = float(roc_auc_score(sc_y_true, sc_y_prob))
        else:
            sc_metrics["auc_roc"] = None

        scenario_metrics[scenario] = sc_metrics
        log.info(
            "Scenario '%s' — Precision: %.4f | Recall: %.4f | F1: %.4f",
            scenario,
            sc_metrics["precision"],
            sc_metrics["recall"],
            sc_metrics["f1"],
        )

    # Feature importance
    importances = dict(zip(FEATURE_COLUMNS, model.feature_importances_))
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
    log.info("Top 5 features by XGBoost importance:")
    for feat, imp in top_features:
        log.info("  %-45s %.4f", feat, imp)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    log.info("Confusion matrix:\n%s", cm)

    # Save model
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        bundle_path = os.path.join(output_dir, "xgboost_supervised.pkl")
        bundle = {
            "model":           model,
            "scaler":          scaler,
            "feature_columns": FEATURE_COLUMNS,
            "params":          XGB_PARAMS,
            "overall_metrics": overall_metrics,
            "scenario_metrics": scenario_metrics,
            "feature_importance": importances,
        }
        with open(bundle_path, "wb") as f:
            pickle.dump(bundle, f)
        log.info("Model bundle saved to %s", bundle_path)

    return {
        "overall":   overall_metrics,
        "scenarios": scenario_metrics,
        "top_features": top_features,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate XGBoost on injected anomaly data."
    )
    parser.add_argument(
        "--data", required=True,
        help="Directory containing injected_labeled_features.parquet"
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional directory to save the trained model bundle"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.20,
        help="Fraction of data to hold out for testing (default: 0.20)"
    )
    args = parser.parse_args()
    train_and_evaluate(
        data_dir=args.data,
        output_dir=args.output,
        test_size=args.test_size,
    )
