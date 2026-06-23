"""
ml/model_registry.py

Utility functions for interacting with scores.model_registry in Postgres.

Called by:
    - ml/isolation_forest.py  (register + promote)
    - dags/dag3_monthly_retraining.py (get active model id)

Keeping this in a single module means DAG code and ML code share the same
database interaction logic rather than duplicating it.
"""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import create_engine, text

log = logging.getLogger(__name__)


def register_model(
    db_conn_str: str,
    model_type: str,
    model_version: str,
    training_samples: int,
    n_estimators: int,
    contamination: float,
    avg_anomaly_score: float,
    model_path: str,
    is_active: bool = False,
    test_auc_roc: Optional[float] = None,
    test_precision: Optional[float] = None,
    test_recall: Optional[float] = None,
    test_f1: Optional[float] = None,
    notes: Optional[str] = None,
) -> int:
    """
    Inserts a new model record into scores.model_registry.
    Returns the assigned model_id.
    """
    engine = create_engine(db_conn_str)

    with engine.connect() as conn:
        result = conn.execute(
            text("""
                INSERT INTO scores.model_registry (
                    model_type, model_version, training_samples, n_estimators,
                    contamination, avg_anomaly_score, is_active,
                    test_auc_roc, test_precision, test_recall, test_f1,
                    model_path, notes
                ) VALUES (
                    :model_type, :model_version, :training_samples, :n_estimators,
                    :contamination, :avg_anomaly_score, :is_active,
                    :test_auc_roc, :test_precision, :test_recall, :test_f1,
                    :model_path, :notes
                )
                RETURNING model_id
            """),
            {
                "model_type":        model_type,
                "model_version":     model_version,
                "training_samples":  training_samples,
                "n_estimators":      n_estimators,
                "contamination":     contamination,
                "avg_anomaly_score": avg_anomaly_score,
                "is_active":         is_active,
                "test_auc_roc":      test_auc_roc,
                "test_precision":    test_precision,
                "test_recall":       test_recall,
                "test_f1":           test_f1,
                "model_path":        model_path,
                "notes":             notes,
            },
        )
        model_id = result.scalar()
        conn.commit()

    log.info("Registered model %s v%s with id=%d (is_active=%s).",
             model_type, model_version, model_id, is_active)
    return model_id


def promote_model(
    db_conn_str: str,
    model_id: int,
    model_type: str,
    notes: Optional[str] = None,
) -> None:
    """
    Sets model_id as the active model for model_type.
    Deactivates all other models of the same type first.
    The unique index on (model_type) WHERE is_active = TRUE enforces
    single-active-model-per-type at the database level.
    """
    engine = create_engine(db_conn_str)

    with engine.connect() as conn:
        # Deactivate all current active models of this type
        conn.execute(
            text("""
                UPDATE scores.model_registry
                SET is_active = FALSE
                WHERE model_type = :mtype
                  AND is_active = TRUE
            """),
            {"mtype": model_type},
        )

        # Promote the new model
        conn.execute(
            text("""
                UPDATE scores.model_registry
                SET is_active = TRUE,
                    promoted_at = NOW(),
                    notes = COALESCE(:notes, notes)
                WHERE model_id = :mid
            """),
            {"notes": notes, "mid": model_id},
        )
        conn.commit()

    log.info("Model id=%d promoted to active for type '%s'.", model_id, model_type)


def get_active_model_id(
    db_conn_str: str,
    model_type: str,
) -> Optional[int]:
    """
    Returns the model_id of the currently active model for model_type,
    or None if no active model exists yet.
    """
    engine = create_engine(db_conn_str)

    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT model_id FROM scores.model_registry
                WHERE model_type = :mtype AND is_active = TRUE
                LIMIT 1
            """),
            {"mtype": model_type},
        )
        row = result.fetchone()

    return row[0] if row else None


def get_active_model_path(
    db_conn_str: str,
    model_type: str,
) -> Optional[str]:
    """
    Returns the filesystem path of the active model's serialised bundle,
    or None if no active model exists.
    """
    engine = create_engine(db_conn_str)

    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT model_path FROM scores.model_registry
                WHERE model_type = :mtype AND is_active = TRUE
                LIMIT 1
            """),
            {"mtype": model_type},
        )
        row = result.fetchone()

    return row[0] if row else None


def list_model_versions(
    db_conn_str: str,
    model_type: Optional[str] = None,
) -> list[dict]:
    """
    Returns all registered models as a list of dicts, optionally filtered
    by model_type. Sorted by trained_at descending.
    Useful for CLI inspection or notebook exploration.
    """
    engine = create_engine(db_conn_str)

    query = """
        SELECT model_id, model_type, model_version, trained_at,
               training_samples, avg_anomaly_score,
               test_auc_roc, test_f1, is_active, promoted_at, notes
        FROM scores.model_registry
    """
    params: dict = {}
    if model_type:
        query += " WHERE model_type = :mtype"
        params["mtype"] = model_type
    query += " ORDER BY trained_at DESC"

    with engine.connect() as conn:
        result = conn.execute(text(query), params)
        rows = [dict(row._mapping) for row in result.fetchall()]

    return rows
