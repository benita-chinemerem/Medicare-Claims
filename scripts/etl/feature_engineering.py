"""
scripts/etl/feature_engineering.py

Provider-level feature computation for fraud anomaly detection.

Reads from the analytics schema and writes one row per (at_physn_npi, period_year)
to features.provider_features. All 18 features correspond directly to the
fraud signal categories described in the white paper (Section 5).

Feature groups:
    - Volume and velocity
    - HCPCS procedure code concentration
    - Billing amount ratios
    - Beneficiary characteristics
    - Place-of-service patterns
    - Temporal billing patterns
    - Duplicate and near-duplicate claim detection
    - Post-death billing

Called by DAG 2 (dag2_weekly_scoring.py) on each weekly run.
Can also be run standalone for a full feature refresh.

Usage:
    python scripts/etl/feature_engineering.py [--batch-id N]
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CHUNK_SIZE = 10_000


def _get_db_conn_str() -> str:
    return (
        f"postgresql+psycopg2://{os.environ.get('FRAUD_DB_USER', 'airflow')}:"
        f"{os.environ.get('FRAUD_DB_PASSWORD', 'airflow')}@"
        f"{os.environ.get('FRAUD_DB_HOST', 'postgres')}:"
        f"{os.environ.get('FRAUD_DB_PORT', '5432')}/"
        f"{os.environ.get('FRAUD_DB_NAME', 'fraud_claims')}"
    )


def _load_carrier_claims(engine, batch_id: Optional[int] = None) -> pd.DataFrame:
    """Loads carrier claims from the analytics schema, optionally filtered by batch watermark."""
    query = """
        SELECT cc.clm_id, cc.desynpuf_id, cc.at_physn_npi,
               cc.clm_from_dt,
               EXTRACT(YEAR FROM CAST(cc.clm_from_dt AS DATE))::INT AS claim_year,
               cc.submitted_charge_amt, cc.allowed_amt,
               cc.submitted_to_allowed_ratio,
               cc.primary_hcpcs_cd, cc.place_of_service_cd,
               (EXTRACT(ISODOW FROM CAST(cc.clm_from_dt AS DATE)) IN (6, 7)) AS is_weekend_claim,
               bs.death_date,
               bs.part_b_months,
               0 AS chronic_condition_count,  -- TEMPORARY PATCH: Replace 0 with the real column when found
               bs.state_code
        FROM analytics.carrier_claims cc
        LEFT JOIN analytics.beneficiary_summary bs
               ON bs.desynpuf_id = cc.desynpuf_id
              AND bs.year = EXTRACT(YEAR FROM CAST(cc.clm_from_dt AS DATE))::INT
    """
    params: dict = {}
    if batch_id is not None:
        query += """
            WHERE CAST(cc.clm_from_dt AS DATE) > (
                SELECT last_clm_date FROM analytics.scoring_watermark
                WHERE claim_type = 'carrier'
            )
        """
    return pd.read_sql(text(query), engine, params=params)


def _detect_duplicates(claims: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies exact and near-duplicate claim pairs within each provider's claims.

    Exact duplicate: same desynpuf_id + primary_hcpcs_cd + clm_from_dt
    Near duplicate:  same desynpuf_id + primary_hcpcs_cd, dates within 3 days

    Returns a DataFrame with columns [at_physn_npi, exact_dups, near_dups].
    """
    claims = claims.copy()
    claims["clm_from_dt"] = pd.to_datetime(claims["clm_from_dt"])

    results = []
    for npi, group in claims.groupby("at_physn_npi"):
        group = group.sort_values("clm_from_dt")

        # Exact duplicates
        exact_key = ["desynpuf_id", "primary_hcpcs_cd", "clm_from_dt"]
        exact_dups = group.duplicated(subset=exact_key, keep=False).sum()

        # Near-duplicates: same bene + code, dates within 3 days
        near_dups = 0
        grouped_bene_code = group.groupby(["desynpuf_id", "primary_hcpcs_cd"])
        for _, bc_group in grouped_bene_code:
            if len(bc_group) < 2:
                continue
            dates = bc_group["clm_from_dt"].sort_values().values
            for i in range(len(dates) - 1):
                diff = (dates[i + 1] - dates[i]).astype("timedelta64[D]").astype(int)
                if 0 < diff <= 3:
                    near_dups += 1

        results.append({
            "at_physn_npi":      npi,
            "exact_duplicate_count": int(exact_dups),
            "near_duplicate_count":  int(near_dups),
        })

    return pd.DataFrame(results)


def compute_provider_features(
    db_conn_str: str,
    batch_id: Optional[int] = None,
) -> int:
    """
    Main entry point. Computes all provider-level features and upserts
    into features.provider_features. Returns the number of provider-year
    rows written.
    """
    engine = create_engine(db_conn_str)

    log.info("Loading carrier claims from analytics schema...")
    claims = _load_carrier_claims(engine, batch_id=batch_id)

    if claims.empty:
        log.warning("No carrier claims found for feature computation.")
        return 0

    claims["clm_from_dt"] = pd.to_datetime(claims["clm_from_dt"])
    claims["claim_year"]  = claims["clm_from_dt"].dt.year

    log.info("Computing features for %d claim rows across %d providers...",
             len(claims), claims["at_physn_npi"].nunique())

    # -----------------------------------------------------------------------
    # 1. Volume and velocity
    # -----------------------------------------------------------------------
    vol = (
        claims.groupby(["at_physn_npi", "claim_year"])
        .agg(total_carrier_claims=("clm_id", "count"))
        .reset_index()
    )

    # Prior year claim count (for growth rate)
    vol_shifted = vol.copy()
    vol_shifted["claim_year_next"] = vol_shifted["claim_year"] + 1
    vol = vol.merge(
        vol_shifted[["at_physn_npi", "claim_year_next", "total_carrier_claims"]]
        .rename(columns={
            "claim_year_next":     "claim_year",
            "total_carrier_claims": "prior_period_claim_count",
        }),
        on=["at_physn_npi", "claim_year"],
        how="left",
    )
    vol["claim_volume_growth_pct"] = (
        (vol["total_carrier_claims"] - vol["prior_period_claim_count"])
        / vol["prior_period_claim_count"].replace(0, np.nan)
        * 100
    )

    # -----------------------------------------------------------------------
    # 2. Beneficiary counts and claims per beneficiary
    # -----------------------------------------------------------------------
    bene = (
        claims.groupby(["at_physn_npi", "claim_year"])
        .agg(
            distinct_beneficiaries=("desynpuf_id", "nunique"),
            beneficiaries_per_state=("state_code", "nunique"),
        )
        .reset_index()
    )

    claims_per_bene = (
        claims.groupby(["at_physn_npi", "claim_year", "desynpuf_id"])
        .size()
        .reset_index(name="claims_for_bene")
        .groupby(["at_physn_npi", "claim_year"])
        .agg(
            avg_claims_per_beneficiary=("claims_for_bene", "mean"),
            carrier_claims_per_bene=("claims_for_bene", "mean"),
        )
        .reset_index()
    )

    high_chronic = (
        claims[claims["chronic_condition_count"].notna()]
        .groupby(["at_physn_npi", "claim_year"])
        .apply(lambda g: (g["chronic_condition_count"] >= 3).mean())
        .reset_index(name="high_chronic_burden_benes_pct")
    )

    # -----------------------------------------------------------------------
    # 3. HCPCS code concentration
    # -----------------------------------------------------------------------
    hcpcs_dist = (
        claims.groupby(["at_physn_npi", "claim_year", "primary_hcpcs_cd"])
        .size()
        .reset_index(name="code_count")
    )

    distinct_codes = (
        hcpcs_dist.groupby(["at_physn_npi", "claim_year"])
        .agg(distinct_hcpcs_codes=("primary_hcpcs_cd", "nunique"))
        .reset_index()
    )

    total_by_provider = (
        hcpcs_dist.groupby(["at_physn_npi", "claim_year"])["code_count"]
        .sum()
        .reset_index(name="total_claims_for_share")
    )
    hcpcs_dist = hcpcs_dist.merge(total_by_provider, on=["at_physn_npi", "claim_year"])
    hcpcs_dist["share"] = hcpcs_dist["code_count"] / hcpcs_dist["total_claims_for_share"]

    # Herfindahl index: sum of squared shares (concentration measure)
    hhi = (
        hcpcs_dist.groupby(["at_physn_npi", "claim_year"])
        .apply(lambda g: (g["share"] ** 2).sum())
        .reset_index(name="hcpcs_concentration_score")
    )

    top_code = (
        hcpcs_dist.loc[
            hcpcs_dist.groupby(["at_physn_npi", "claim_year"])["code_count"].idxmax()
        ][["at_physn_npi", "claim_year", "primary_hcpcs_cd", "share"]]
        .rename(columns={
            "primary_hcpcs_cd": "top_hcpcs_code",
            "share":            "top_hcpcs_code_share",
        })
    )

    # -----------------------------------------------------------------------
    # 4. Billing amount ratios
    # -----------------------------------------------------------------------
    billing = (
        claims.groupby(["at_physn_npi", "claim_year"])
        .agg(
            avg_submitted_charge=("submitted_charge_amt", "mean"),
            avg_allowed_amt=("allowed_amt", "mean"),
            avg_submitted_to_allowed_ratio=("submitted_to_allowed_ratio", "mean"),
            p95_submitted_to_allowed_ratio=(
                "submitted_to_allowed_ratio",
                lambda x: x.quantile(0.95)
            ),
        )
        .reset_index()
    )

    # -----------------------------------------------------------------------
    # 5. Place-of-service breakdown
    # -----------------------------------------------------------------------
    def pos_pct(g, code):
        total = len(g)
        return (g["place_of_service_cd"] == str(code)).sum() / total if total else 0.0

    pos = (
        claims.groupby(["at_physn_npi", "claim_year"])
        .apply(lambda g: pd.Series({
            "distinct_pos_codes":          g["place_of_service_cd"].nunique(),
            "pct_claims_office":           pos_pct(g, "11"),
            "pct_claims_home":             pos_pct(g, "12"),
            "pct_claims_nursing_facility": pos_pct(g, "31"),
        }))
        .reset_index()
    )

    # -----------------------------------------------------------------------
    # 6. Temporal patterns
    # -----------------------------------------------------------------------
    temporal = (
        claims.groupby(["at_physn_npi", "claim_year"])
        .agg(
            pct_weekend_claims=("is_weekend_claim", "mean"),
            max_claims_in_single_day=("clm_from_dt", lambda x: x.value_counts().max()),
        )
        .reset_index()
    )

    # -----------------------------------------------------------------------
    # 7. Post-death billing
    # -----------------------------------------------------------------------
    has_death = claims[claims["death_date"].notna()].copy()
    has_death["death_date"] = pd.to_datetime(has_death["death_date"])
    has_death["after_death"] = has_death["clm_from_dt"] > has_death["death_date"]
    has_death["in_death_year"] = (
        has_death["clm_from_dt"].dt.year == has_death["death_date"].dt.year
    )

    death_flags = (
        has_death.groupby(["at_physn_npi", "claim_year"])
        .agg(
            claims_after_bene_death=("after_death",  "sum"),
            claims_in_bene_death_year=("in_death_year", "sum"),
        )
        .reset_index()
    )

    # -----------------------------------------------------------------------
    # 8. Duplicate detection
    # -----------------------------------------------------------------------
    log.info("Computing duplicate claim pairs (this may take a moment)...")
    dup_df = _detect_duplicates(claims)

    # -----------------------------------------------------------------------
    # Merge all feature groups
    # -----------------------------------------------------------------------
    base = vol.copy()

    for df, cols in [
        (bene,          ["at_physn_npi", "claim_year"]),
        (claims_per_bene, ["at_physn_npi", "claim_year"]),
        (high_chronic,  ["at_physn_npi", "claim_year"]),
        (distinct_codes, ["at_physn_npi", "claim_year"]),
        (hhi,           ["at_physn_npi", "claim_year"]),
        (top_code,      ["at_physn_npi", "claim_year"]),
        (billing,       ["at_physn_npi", "claim_year"]),
        (pos,           ["at_physn_npi", "claim_year"]),
        (temporal,      ["at_physn_npi", "claim_year"]),
        (death_flags,   ["at_physn_npi", "claim_year"]),
    ]:
        base = base.merge(df, on=cols, how="left")

    # Merge duplicate counts (no claim_year dimension — aggregate across years)
    base = base.merge(dup_df, on="at_physn_npi", how="left")

    # Fill missing duplicate counts with 0
    for col in ("exact_duplicate_count", "near_duplicate_count",
                "claims_after_bene_death", "claims_in_bene_death_year"):
        base[col] = base[col].fillna(0).astype(int)

    # Duplicate rate
    base["duplicate_rate"] = (
        (base["exact_duplicate_count"] + base["near_duplicate_count"])
        / base["total_carrier_claims"].replace(0, np.nan)
    ).fillna(0)

    # Batch ID
    base["batch_id"] = batch_id

    # Rename claim_year to period_year
    base = base.rename(columns={"claim_year": "period_year"})

    # -----------------------------------------------------------------------
    # Write to features.provider_features (upsert via delete + insert)
    # -----------------------------------------------------------------------
    npi_list    = base["at_physn_npi"].unique().tolist()
    year_list   = base["period_year"].unique().tolist()

    with engine.connect() as conn:
        if batch_id is not None:
            conn.execute(
                text("""
                    DELETE FROM features.provider_features
                    WHERE at_physn_npi = ANY(:npis)
                      AND period_year  = ANY(:years)
                """),
                {"npis": npi_list, "years": [int(y) for y in year_list]},
            )
        else:
            conn.execute(text("TRUNCATE features.provider_features"))
        conn.commit()

    rows_written = 0
    for i in range(0, len(base), CHUNK_SIZE):
        chunk = base.iloc[i : i + CHUNK_SIZE]
        chunk.to_sql(
            "provider_features", engine,
            schema="features", if_exists="append",
            index=False, method="multi",
        )
        rows_written += len(chunk)

    log.info(
        "Feature engineering complete. %d provider-year rows written to features schema.",
        rows_written,
    )
    return rows_written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute provider-level fraud-detection features.")
    parser.add_argument("--batch-id", type=int, default=None,
                        help="If set, restricts computation to the current scoring batch.")
    args = parser.parse_args()
    compute_provider_features(
        db_conn_str=_get_db_conn_str(),
        batch_id=args.batch_id,
    )