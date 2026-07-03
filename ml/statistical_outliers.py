"""
ml/statistical_outliers.py

Per-feature statistical outlier scoring alongside the Isolation Forest
composite score (Section 6.3 of the project spec).

For each provider-year row in the current batch, computes:
  - z-score    : signed distance from the population mean in standard deviation
                 units. Positive = above average, negative = below.
  - zscore_outlier : True when |z-score| > ZSCORE_THRESHOLD (default 3.0)
  - iqr_outlier    : True when value falls outside the Tukey fence
                     (Q1 - 1.5*IQR, Q3 + 1.5*IQR)

Only rows where at least one flag is True are stored, keeping table size
manageable as batch counts accumulate.

Results are written to scores.provider_outlier_flags, indexed by
(scoring_run_id, at_physn_npi).

These signals serve two purposes for investigators:
  1. Single-feature anomaly flags that are simpler to explain than a
     composite Isolation Forest score.
  2. A cross-check on the IF score — if a provider ranks high on IF
     but has no single-feature flags, the composite score warrants
     manual review before actioning.

Called by DAG 2 (dag2_weekly_scoring.py → compute_statistical_outliers task).
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── Configuration ─────────────────────────────────────────────────────────────
ZSCORE_THRESHOLD = 3.0    # |z| > 3 → flag
IQR_MULTIPLIER   = 1.5    # Tukey standard fence

# Features to score. Binary / indicator columns are excluded since
# z-scores are not meaningful for 0/1 distributions.
OUTLIER_FEATURE_COLUMNS: List[str] = [
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
# has_prior_period is a binary flag — excluded from z-score / IQR scoring.
# ─────────────────────────────────────────────────────────────────────────────


def _ensure_outlier_table(engine) -> None:
    """Create scores.provider_outlier_flags if it does not yet exist."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS scores.provider_outlier_flags (
                id             SERIAL PRIMARY KEY,
                scoring_run_id INT          NOT NULL,
                at_physn_npi   VARCHAR(50)  NOT NULL,
                period_year    INT,
                feature_name   VARCHAR(100) NOT NULL,
                feature_value  FLOAT,
                zscore         FLOAT,
                zscore_outlier BOOLEAN      NOT NULL DEFAULT FALSE,
                iqr_outlier    BOOLEAN      NOT NULL DEFAULT FALSE,
                created_at     TIMESTAMP    DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_outlier_flags_run
                ON scores.provider_outlier_flags (scoring_run_id);
            CREATE INDEX IF NOT EXISTS idx_outlier_flags_npi
                ON scores.provider_outlier_flags (at_physn_npi);
        """))


def compute_outlier_flags(
    df: pd.DataFrame,
    feat_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute z-score and IQR outlier flags for every (provider, feature) pair.

    Args:
        df:         DataFrame with one row per provider-year.
                    Must contain 'at_physn_npi', 'period_year', and feature cols.
        feat_cols:  Feature columns to evaluate. Defaults to
                    OUTLIER_FEATURE_COLUMNS.

    Returns:
        Long-format DataFrame of flagged (provider, feature) pairs with columns:
            at_physn_npi, period_year, feature_name,
            feature_value, zscore, zscore_outlier, iqr_outlier
        Only rows where at least one flag is True are included.
    """
    if feat_cols is None:
        feat_cols = OUTLIER_FEATURE_COLUMNS

    available = [c for c in feat_cols if c in df.columns]
    if not available:
        log.warning("No matching feature columns found in DataFrame.")
        return pd.DataFrame()

    records = []

    for feat in available:
        series = df[feat].fillna(0).astype(float)

        # ── z-score ──────────────────────────────────────────────────────────
        mean = series.mean()
        std  = series.std(ddof=1)
        if std > 0:
            zscores = (series - mean) / std
        else:
            # All values are identical — no variation, no outliers on this feature
            zscores = pd.Series(0.0, index=series.index)

        # ── IQR fence ────────────────────────────────────────────────────────
        q1  = series.quantile(0.25)
        q3  = series.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            lower = q1 - IQR_MULTIPLIER * iqr
            upper = q3 + IQR_MULTIPLIER * iqr
            iqr_flag = (series < lower) | (series > upper)
        else:
            iqr_flag = pd.Series(False, index=series.index)

        zscore_flag = zscores.abs() > ZSCORE_THRESHOLD

        # ── Collect only flagged rows to keep storage compact ─────────────
        mask = zscore_flag | iqr_flag
        if mask.sum() == 0:
            continue

        chunk = pd.DataFrame({
            "at_physn_npi":  df.loc[mask, "at_physn_npi"].values,
            "period_year":   df.loc[mask, "period_year"].values,
            "feature_name":  feat,
            "feature_value": series[mask].values,
            "zscore":        zscores[mask].values.round(4),
            "zscore_outlier": zscore_flag[mask].values,
            "iqr_outlier":    iqr_flag[mask].values,
        })
        records.append(chunk)

    if not records:
        log.info("No statistical outliers detected in this batch.")
        return pd.DataFrame()

    result = pd.concat(records, ignore_index=True)
    log.info(
        "Statistical outlier scoring: %d (provider, feature) flag rows from %d providers.",
        len(result),
        result["at_physn_npi"].nunique(),
    )
    return result


def write_outlier_flags(
    engine,
    flagged_df: pd.DataFrame,
    scoring_run_id: int,
) -> None:
    """
    Write the flagged (provider, feature) rows to scores.provider_outlier_flags.
    Deletes any existing rows for this scoring_run_id first (idempotent).
    """
    _ensure_outlier_table(engine)

    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM scores.provider_outlier_flags WHERE scoring_run_id = :rid"),
            {"rid": scoring_run_id},
        )

    if flagged_df.empty:
        log.info("No outlier flag rows to write for run %d.", scoring_run_id)
        return

    out = flagged_df.copy()
    out["scoring_run_id"] = scoring_run_id

    out.to_sql(
        "provider_outlier_flags",
        engine,
        schema="scores",
        if_exists="append",
        index=False,
        method="multi",
        chunksize=5_000,
    )
    log.info(
        "Written %d outlier flag rows to scores.provider_outlier_flags (run %d).",
        len(out), scoring_run_id,
    )


def score_batch_outliers(
    db_conn_str: str,
    scoring_run_id: int,
    batch_id: int,
) -> int:
    """
    Main entry point called from DAG 2.

    Loads provider features for the current batch, computes z-score and
    IQR outlier flags for all continuous features, and writes results to
    scores.provider_outlier_flags.

    Args:
        db_conn_str:    SQLAlchemy connection string.
        scoring_run_id: Current scoring run ID (from scores.scoring_runs).
        batch_id:       Current batch ID (used to load features).

    Returns:
        Count of (provider, feature) flag rows written.
    """
    engine = create_engine(db_conn_str)

    with engine.connect() as conn:
        df = pd.read_sql(
            text("""
                SELECT *
                FROM features.provider_features
                WHERE batch_id = :bid
            """),
            conn,
            params={"bid": batch_id},
        )

    if df.empty:
        log.warning("No features found for batch_id=%d. Skipping outlier scoring.", batch_id)
        return 0

    log.info(
        "Computing statistical outliers for %d provider-year rows (batch %d).",
        len(df), batch_id,
    )
    flagged_df = compute_outlier_flags(df, OUTLIER_FEATURE_COLUMNS)
    write_outlier_flags(engine, flagged_df, scoring_run_id)
    return len(flagged_df)
