"""
ml/shap_explainer.py

SHAP (SHapley Additive exPlanations) computation for flagged providers.

For every provider that clears the risk-score threshold in a given scoring run,
this module:
    1. Loads the active Isolation Forest model bundle
    2. Computes SHAP values using the TreeExplainer
    3. Ranks features by absolute SHAP contribution per provider
    4. Generates three human-readable reason codes (top_reason_1/2/3)
    5. Writes per-feature SHAP rows to scores.shap_values
    6. Updates the top_reason columns in scores.provider_risk_scores

The reason codes are what investigators see in the dashboard. They are written
in plain language derived directly from the feature name and direction of the
SHAP value, so no separate translation layer is needed at the dashboard layer.

Reference:
    Lundberg, S.M. and Lee, S-I.
    "A Unified Approach to Interpreting Model Predictions."
    Advances in Neural Information Processing Systems 30, 2017.
"""

from __future__ import annotations

import logging
import pickle
from typing import Optional

import numpy as np
import pandas as pd
import shap
from sqlalchemy import create_engine, text

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Human-readable templates for each feature.
# {direction} is filled with "above" or "below" based on SHAP sign.
# {value} is the actual feature value for this provider.
# ---------------------------------------------------------------------------
REASON_TEMPLATES: dict[str, str] = {
    "total_carrier_claims":
        "Claim volume {value:.0f} ({direction} peer average)",
    "carrier_claims_per_bene":
        "Claims per beneficiary {value:.1f} ({direction} peer average)",
    "claim_volume_growth_pct":
        "Claim volume growth {value:+.1f}% vs prior period ({direction} normal range)",
    "distinct_hcpcs_codes":
        "Billed {value:.0f} distinct procedure codes ({direction} peer average)",
    "top_hcpcs_code_share":
        "Top procedure code accounts for {value:.0%} of all claims ({direction} peer)",
    "hcpcs_concentration_score":
        "Procedure code concentration score {value:.3f} ({direction} peer average)",
    "avg_submitted_to_allowed_ratio":
        "Average submitted-to-allowed ratio {value:.2f}x ({direction} peer average)",
    "p95_submitted_to_allowed_ratio":
        "95th-pct submitted-to-allowed ratio {value:.2f}x ({direction} peer average)",
    "distinct_beneficiaries":
        "Served {value:.0f} distinct beneficiaries ({direction} peer average)",
    "avg_claims_per_beneficiary":
        "Average {value:.1f} claims per beneficiary ({direction} peer average)",
    "beneficiaries_per_state":
        "Beneficiaries spread across {value:.0f} states ({direction} expected for specialty)",
    "high_chronic_burden_benes_pct":
        "{value:.0%} of beneficiaries have 3+ chronic conditions ({direction} peer average)",
    "pct_weekend_claims":
        "{value:.0%} of claims billed on weekends ({direction} peer average)",
    "max_claims_in_single_day":
        "Peak of {value:.0f} claims billed in a single day ({direction} peer maximum)",
    "exact_duplicate_count":
        "{value:.0f} exact duplicate claim pairs detected",
    "near_duplicate_count":
        "{value:.0f} near-duplicate claim pairs detected (same code, 3-day window)",
    "duplicate_rate":
        "Duplicate rate {value:.2%} ({direction} peer average)",
    "claims_after_bene_death":
        "{value:.0f} claims billed after beneficiary death date",
}


def _build_reason_text(feature_name: str, shap_value: float, feature_value: float) -> str:
    """
    Produces a single plain-English reason string for one feature's SHAP contribution.
    """
    template = REASON_TEMPLATES.get(
        feature_name,
        f"Feature '{feature_name}' value {{value:.4f}} ({{direction}} expected)"
    )
    direction = "above" if shap_value > 0 else "below"
    try:
        return template.format(value=feature_value, direction=direction)
    except (KeyError, ValueError):
        return f"{feature_name}: {feature_value:.4f} ({direction} expected)"


def explain_flagged_providers(
    db_conn_str: str,
    scoring_run_id: int,
    model_path: str,
) -> None:
    """
    Main entry point called by DAG 2.

    Loads all flagged providers for the given scoring_run_id, computes SHAP
    values, and writes results to scores.shap_values and updates reason codes
    in scores.provider_risk_scores.
    """
    engine = create_engine(db_conn_str)

    # Fetch flagged providers for this run
    flagged_df = pd.read_sql(
        text("""
            SELECT prs.id AS score_row_id,
                   prs.at_physn_npi,
                   prs.period_year,
                   prs.batch_id
            FROM scores.provider_risk_scores prs
            WHERE prs.scoring_run_id = :rid
              AND prs.is_flagged = TRUE
        """),
        engine,
        params={"rid": scoring_run_id},
    )

    if flagged_df.empty:
        log.info("No flagged providers in scoring run %d; skipping SHAP.", scoring_run_id)
        return

    log.info("Computing SHAP values for %d flagged providers (run %d).",
             len(flagged_df), scoring_run_id)

    # Load active model bundle
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
        log.error("No active model found; cannot compute SHAP values.")
        return

    with open(row[0], "rb") as f:
        bundle = pickle.load(f)

    model        = bundle["model"]
    scaler       = bundle["scaler"]
    feat_cols    = bundle["feature_columns"]

    # Load feature vectors for flagged providers
    npi_list = flagged_df["at_physn_npi"].unique().tolist()
    placeholders = ", ".join([f"'{n}'" for n in npi_list])

    features_df = pd.read_sql(
        text(f"""
            SELECT *
            FROM features.provider_features
            WHERE at_physn_npi IN ({placeholders})
        """),
        engine,
    )

    if features_df.empty:
        log.warning("No feature rows found for flagged NPIs.")
        return

    # Match features to flagged rows by NPI
    merged = flagged_df.merge(features_df, on="at_physn_npi", how="left")
    merged[feat_cols] = merged[feat_cols].fillna(0)

    X_raw    = merged[feat_cols].values
    X_scaled = scaler.transform(X_raw)

    # Compute SHAP values using TreeExplainer (fast for tree-based models)
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    # shap_values shape: (n_providers, n_features)
    if isinstance(shap_values, list):
        # Some SHAP versions return a list for multi-output; take index 0
        shap_values = shap_values[0]

    shap_rows   = []
    update_rows = []

    for i, (_, provider_row) in enumerate(merged.iterrows()):
        npi          = provider_row["at_physn_npi"]
        score_row_id = provider_row["score_row_id"]
        sv           = shap_values[i]
        fv           = X_raw[i]

        # Rank features by |SHAP value|
        ranked_indices = np.argsort(np.abs(sv))[::-1]

        # Build per-feature rows for scores.shap_values
        for rank, idx in enumerate(ranked_indices):
            feat_name = feat_cols[idx]
            shap_rows.append({
                "at_physn_npi":     npi,
                "scoring_run_id":   scoring_run_id,
                "feature_name":     feat_name,
                "shap_value":       float(sv[idx]),
                "feature_value":    float(fv[idx]),
                "feature_rank":     rank + 1,
            })

        # Generate top 3 plain-language reason codes
        reasons = []
        for idx in ranked_indices[:3]:
            feat_name = feat_cols[idx]
            reason    = _build_reason_text(feat_name, float(sv[idx]), float(fv[idx]))
            reasons.append(reason)

        while len(reasons) < 3:
            reasons.append(None)

        update_rows.append({
            "score_row_id": int(score_row_id),
            "top_reason_1": reasons[0],
            "top_reason_2": reasons[1],
            "top_reason_3": reasons[2],
        })

    # Write SHAP value rows
    if shap_rows:
        shap_df = pd.DataFrame(shap_rows)
        shap_df.to_sql(
            "shap_values",
            engine,
            schema="scores",
            if_exists="append",
            index=False,
            method="multi",
        )
        log.info("Written %d SHAP value rows for run %d.", len(shap_df), scoring_run_id)

    # Update reason codes in provider_risk_scores
    with engine.connect() as conn:
        for update in update_rows:
            conn.execute(
                text("""
                    UPDATE scores.provider_risk_scores
                    SET top_reason_1 = :r1,
                        top_reason_2 = :r2,
                        top_reason_3 = :r3
                    WHERE id = :sid
                """),
                {
                    "r1":  update["top_reason_1"],
                    "r2":  update["top_reason_2"],
                    "r3":  update["top_reason_3"],
                    "sid": update["score_row_id"],
                },
            )
        conn.commit()

    log.info("Reason codes updated for %d flagged providers.", len(update_rows))
