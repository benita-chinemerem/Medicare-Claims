"""
ml/isolation_forest.py

Isolation Forest unsupervised anomaly detection for provider-level fraud risk scoring.

Key design decisions:
- Operates on provider-year feature vectors from features.provider_features
- Outputs a 0-100 risk score (higher = more anomalous)
- Contamination rate is configurable via .env (default 5%)
- All feature columns used are defined in FEATURE_COLUMNS below;
  change this list to add or remove features without touching model code

References:
    Liu, F.T., Ting, K.M., Zhou, Z-H. "Isolation Forest."
    Proceedings of the 8th IEEE International Conference on Data Mining, 2008.
"""

from __future__ import annotations

import os
import logging
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, text

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature columns fed to the model.
# These must exist in features.provider_features.
# ---------------------------------------------------------------------------
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


def _load_features(db_conn_str: str, batch_id: Optional[int] = None) -> pd.DataFrame:
    """
    Loads provider feature vectors from Postgres.
    If batch_id is provided, loads only features computed in that batch.
    Otherwise loads all available features.
    """
    engine = create_engine(db_conn_str)

    query = "SELECT * FROM features.provider_features"
    params: dict = {}
    if batch_id is not None:
        query += " WHERE batch_id = :bid"
        params["bid"] = batch_id

    df = pd.read_sql(text(query), engine, params=params)

    if df.empty:
        return df

    # Fill NaN values — isolation forest does not accept NaN
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(0)
    return df


def _normalize_isolation_score(raw_scores: np.ndarray) -> np.ndarray:
    """
    Converts Isolation Forest raw scores (negative = anomalous, in [-1, 0])
    to a 0-100 integer risk scale where 100 is most anomalous.
    """
    # Raw scores are in [-0.5, 0.5] approximately; decision_function output
    # is in (-inf, +inf) but practically in [-1, 1].
    # Rescale so -1 -> 100, +1 -> 0, clamped.
    normalized = np.clip((-raw_scores + 1) / 2 * 100, 0, 100)
    return normalized.astype(int)


def train_isolation_forest(
    features_path: str,
    model_output_path: str,
    contamination: float = 0.05,
    n_estimators: int = 200,
    random_state: int = 42,
) -> dict:
    """
    Trains an Isolation Forest on the feature dataset at features_path.
    Saves both the scaler and the model to model_output_path (as a dict pickle).

    Returns a metrics dict with training statistics for the model registry.
    """
    df = pd.read_parquet(features_path)

    if df.empty:
        raise ValueError("Training dataset is empty.")

    # Keep only numeric feature columns; drop metadata cols
    X = df[FEATURE_COLUMNS].fillna(0).values
    log.info("Training Isolation Forest on %d provider-year vectors, %d features.",
             X.shape[0], X.shape[1])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # Decision function scores: negative = anomalous
    raw_scores = model.decision_function(X_scaled)
    risk_scores = _normalize_isolation_score(raw_scores)

    metrics = {
        "n_estimators":     n_estimators,
        "contamination":    contamination,
        "avg_anomaly_score": float(np.mean(risk_scores)),
        "pct_flagged":       float(np.mean(risk_scores >= 75)),
        "training_rows":     X.shape[0],
    }

    # Save model bundle (scaler + model + feature columns)
    bundle = {
        "model":           model,
        "scaler":          scaler,
        "feature_columns": FEATURE_COLUMNS,
        "metrics":         metrics,
    }
    with open(model_output_path, "wb") as f:
        pickle.dump(bundle, f)

    log.info(
        "Model saved to %s. Avg risk score: %.2f | Pct flagged: %.2f%%",
        model_output_path, metrics["avg_anomaly_score"], metrics["pct_flagged"] * 100
    )
    return metrics


def score_providers_batch(
    db_conn_str: str,
    scoring_run_id: int,
    batch_id: int,
    risk_threshold: int,
    model_path: str,
) -> int:
    """
    Scores all providers in the current batch using the active Isolation Forest.
    Writes results to scores.provider_risk_scores.

    Returns the count of flagged providers.
    """
    engine = create_engine(db_conn_str)

    # Find active model file
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT model_path FROM scores.model_registry
                WHERE model_type = 'isolation_forest' AND is_active = TRUE
                LIMIT 1
            """)
        )
        row = result.fetchone()

    if row is None:
        # No promoted model yet — train one on the fly using current features
        log.warning("No active Isolation Forest found. Training initial model.")
        features_path = os.path.join(model_path, "training_features.parquet")
        model_output  = os.path.join(model_path, "isolation_forest_initial.pkl")

        # Write features to disk for training
        df_all = _load_features(db_conn_str)
        df_all.to_parquet(features_path, index=False)

        metrics = train_isolation_forest(
            features_path=features_path,
            model_output_path=model_output,
            contamination=float(os.environ.get("CONTAMINATION_RATE", 0.05)),
        )
        active_model_path = model_output
    else:
        active_model_path = row[0]

    # Load model bundle
    with open(active_model_path, "rb") as f:
        bundle = pickle.load(f)

    model:    IsolationForest = bundle["model"]
    scaler:   StandardScaler  = bundle["scaler"]
    feat_cols: list           = bundle["feature_columns"]

    # Load batch features
    df = _load_features(db_conn_str, batch_id=batch_id)
    if df.empty:
        log.info("No features for batch %d; nothing to score.", batch_id)
        return 0

    X = df[feat_cols].fillna(0).values
    X_scaled = scaler.transform(X)

    raw_scores  = model.decision_function(X_scaled)
    risk_scores = _normalize_isolation_score(raw_scores)

    # Compute risk deciles
    deciles = pd.qcut(risk_scores, q=10, labels=False, duplicates="drop") + 1

    # Write results
    rows_to_insert = []
    for i, (_, row_) in enumerate(df.iterrows()):
        is_flagged = int(risk_scores[i]) >= risk_threshold
        rows_to_insert.append({
            "at_physn_npi":                    row_["at_physn_npi"],
            "scoring_run_id":                  scoring_run_id,
            "period_year":                     row_.get("period_year"),
            "batch_id":                        batch_id,
            "isolation_forest_score":          float(raw_scores[i]),
            "risk_score":                      int(risk_scores[i]),
            "risk_decile":                     int(deciles[i]) if not np.isnan(deciles[i]) else None,
            "is_flagged":                      is_flagged,
            "total_claims":                    row_.get("total_carrier_claims"),
            "distinct_benes":                  row_.get("distinct_beneficiaries"),
            "avg_submitted_to_allowed_ratio":  row_.get("avg_submitted_to_allowed_ratio"),
            "duplicate_rate":                  row_.get("duplicate_rate"),
            "pct_weekend_claims":              row_.get("pct_weekend_claims"),
        })

    insert_df = pd.DataFrame(rows_to_insert)
    insert_df.to_sql(
        "provider_risk_scores",
        engine,
        schema="scores",
        if_exists="append",
        index=False,
        method="multi",
    )

    flagged_count = int(insert_df["is_flagged"].sum())
    log.info("Scored %d providers for run %d. Flagged: %d",
             len(insert_df), scoring_run_id, flagged_count)
    return flagged_count


def evaluate_model_comparison(
    db_conn_str: str,
    new_model_id: int,
    current_model_id: int,
    features_path: str,
    model_path: str,
) -> float:
    """
    Compares new vs current model by evaluating anomaly score separation on
    the held-out feature set.

    Returns the improvement as a float; positive means the new model is better.
    The comparison uses mean anomaly score delta as a simple proxy — in a
    production system with labeled data this would be AUC-ROC or precision-recall.
    """
    from sqlalchemy import create_engine, text as sqla_text

    engine = create_engine(db_conn_str)

    def get_model_path(model_id: int) -> Optional[str]:
        with engine.connect() as conn:
            result = conn.execute(
                sqla_text("SELECT model_path FROM scores.model_registry WHERE model_id = :mid"),
                {"mid": model_id},
            )
            row = result.fetchone()
            return row[0] if row else None

    new_path     = get_model_path(new_model_id)
    current_path = get_model_path(current_model_id)

    if not new_path or not os.path.exists(new_path):
        raise FileNotFoundError(f"New model file not found: {new_path}")
    if not current_path or not os.path.exists(current_path):
        log.warning("Current model file not found at %s. Promoting new model.", current_path)
        return 1.0  # force promotion

    df = pd.read_parquet(features_path)
    if df.empty:
        return 0.0

    def score_with_bundle(path: str) -> np.ndarray:
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        X = df[bundle["feature_columns"]].fillna(0).values
        X_scaled = bundle["scaler"].transform(X)
        return bundle["model"].decision_function(X_scaled)

    new_scores     = score_with_bundle(new_path)
    current_scores = score_with_bundle(current_path)

    # Higher mean score spread from center indicates better discrimination
    new_spread     = float(np.std(new_scores))
    current_spread = float(np.std(current_scores))
    improvement    = new_spread - current_spread

    log.info(
        "Model comparison: new_std=%.4f, current_std=%.4f, improvement=%.4f",
        new_spread, current_spread, improvement
    )
    return improvement
