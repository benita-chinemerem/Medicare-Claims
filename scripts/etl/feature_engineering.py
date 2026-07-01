"""
scripts/etl/feature_engineering.py

Provider-level feature computation — SQL-based approach.

All heavy aggregations run inside PostgreSQL. Python only receives the
final aggregated provider-year rows (typically 10K-50K rows), never the
full 9M+ raw claim dataset. This avoids the OOM kill that occurs when
loading all carrier claims into a pandas DataFrame.

Called by DAG 2 (dag2_weekly_scoring.py) on each weekly run.

FIX (2026-06-30):
    Root cause of SIGKILL / zombie task:
        `where_clause_base` was always set to an empty string, so
        `base_claims` materialized the ENTIRE carrier_claims table on
        every run (~9M+ rows). PostgreSQL then had to scan and sort that
        full dataset 7 times in the same query for the various CTEs —
        particularly `near_dups` (LAG + OVER PARTITION BY + ORDER BY on
        9M rows) — which exhausted available memory and caused the OOM
        killer to send SIGKILL to the worker process.

    Fix:
        A new `active_npis` CTE runs first and identifies only the
        providers who have claims within the current batch window
        (batch_start_date → batch_end_date). `base_claims` is then
        filtered to those NPIs only, but still pulls their full claim
        history across all years — so prior-year growth features remain
        accurate. This turns a full-table scan into a targeted NPI-keyed
        lookup using the existing idx_carrier_npi index.

        A per-session `work_mem` increase is also applied so PostgreSQL
        has enough buffer for the LAG sort within each partition without
        spilling to disk.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CHUNK_SIZE = 5_000

# Amount of memory given to Postgres for sort operations within this session.
# Each parallel sort worker gets this much, so keep it reasonable.
WORK_MEM = os.environ.get("FEATURE_WORK_MEM", "512MB")


def _get_db_conn_str() -> str:
    return (
        f"postgresql+psycopg2://{os.environ.get('FRAUD_DB_USER', 'airflow')}:"
        f"{os.environ.get('FRAUD_DB_PASSWORD', 'airflow')}@"
        f"{os.environ.get('FRAUD_DB_HOST', 'postgres')}:"
        f"{os.environ.get('FRAUD_DB_PORT', '5432')}/"
        f"{os.environ.get('FRAUD_DB_NAME', 'fraud_claims')}"
    )


# ---------------------------------------------------------------------------
# All feature computation runs as SQL inside PostgreSQL.
# Python only receives the final aggregated rows.
#
# FIX: `active_npis` CTE is the new gate that limits `base_claims` to only
# the providers who had activity within the current batch window.
# batch_start_date / batch_end_date are passed as bound SQL parameters —
# NOT Python format strings — so there is no SQL injection risk.
# ---------------------------------------------------------------------------

FEATURES_SQL = """
WITH

-- 0. FIX: Identify only the providers active in this batch window.
--    This runs first against idx_carrier_npi + idx_carrier_date so it is fast.
--    base_claims is then filtered to these NPIs only, but keeps ALL of each
--    provider's historical years so that prior-year growth features are accurate.
active_npis AS MATERIALIZED (
    SELECT DISTINCT at_physn_npi
    FROM analytics.carrier_claims
    WHERE clm_from_dt > :batch_start_date
      AND clm_from_dt <= :batch_end_date
      AND at_physn_npi IS NOT NULL
),

-- 1. Create a base layer to parse dates efficiently so we don't repeat work.
--    FIX: WHERE clause now restricts to active_npis — eliminates the full-table
--    scan that was causing the OOM SIGKILL on every run.
base_claims AS MATERIALIZED (
    SELECT
        cc.*,
        EXTRACT(YEAR FROM cc.clm_from_dt::DATE)::INT AS claim_year,
        CASE WHEN EXTRACT(ISODOW FROM cc.clm_from_dt::DATE) IN (6, 7) THEN True ELSE False END AS is_weekend_claim
    FROM analytics.carrier_claims cc
    WHERE cc.at_physn_npi IN (SELECT at_physn_npi FROM active_npis)
),

-- 2. Base carrier claim stats per provider per year
carrier_base AS (
    SELECT
        cc.at_physn_npi,
        cc.claim_year,
        COUNT(*)                                                      AS total_carrier_claims,
        COUNT(DISTINCT cc.desynpuf_id)                                AS distinct_beneficiaries,
        COUNT(DISTINCT cc.primary_hcpcs_cd)                           AS distinct_hcpcs_codes,
        COUNT(DISTINCT cc.place_of_service_cd)                        AS distinct_pos_codes,
        COUNT(DISTINCT bs.state_code)                                 AS beneficiaries_per_state,
        AVG(cc.submitted_to_allowed_ratio)                            AS avg_submitted_to_allowed_ratio,
        PERCENTILE_CONT(0.95) WITHIN GROUP (
            ORDER BY cc.submitted_to_allowed_ratio
        )                                                             AS p95_submitted_to_allowed_ratio,
        AVG(cc.submitted_charge_amt)                                  AS avg_submitted_charge,
        AVG(cc.allowed_amt)                                           AS avg_allowed_amt,
        AVG(CASE WHEN cc.is_weekend_claim THEN 1.0 ELSE 0.0 END)      AS pct_weekend_claims,
        MAX(daily_counts.day_count)                                   AS max_claims_in_single_day,

        SUM(CASE WHEN bs.death_date IS NOT NULL
                  AND cc.clm_from_dt::DATE > bs.death_date
             THEN 1 ELSE 0 END)                                       AS claims_after_bene_death,

        SUM(CASE WHEN bs.death_date IS NOT NULL
                  AND cc.claim_year = EXTRACT(YEAR FROM bs.death_date)::INT
             THEN 1 ELSE 0 END)                                       AS claims_in_bene_death_year,

        AVG(CASE WHEN bc.chronic_condition_count >= 3
             THEN 1.0 ELSE 0.0 END)                                   AS high_chronic_burden_benes_pct,
        AVG(CASE WHEN cc.place_of_service_cd = '11'
             THEN 1.0 ELSE 0.0 END)                                   AS pct_claims_office,
        AVG(CASE WHEN cc.place_of_service_cd = '12'
             THEN 1.0 ELSE 0.0 END)                                   AS pct_claims_home,
        AVG(CASE WHEN cc.place_of_service_cd = '31'
             THEN 1.0 ELSE 0.0 END)                                   AS pct_claims_nursing_facility
    FROM base_claims cc
    LEFT JOIN analytics.beneficiary_summary bs
           ON bs.desynpuf_id = cc.desynpuf_id
          AND bs.year = cc.claim_year
    LEFT JOIN (
        SELECT desynpuf_id, year,
            (CASE WHEN flag_alzheimer    = 1 THEN 1 ELSE 0 END +
             CASE WHEN flag_chf          = 1 THEN 1 ELSE 0 END +
             CASE WHEN flag_chronic_kidney = 1 THEN 1 ELSE 0 END +
             CASE WHEN flag_cancer       = 1 THEN 1 ELSE 0 END +
             CASE WHEN flag_copd         = 1 THEN 1 ELSE 0 END +
             CASE WHEN flag_depression   = 1 THEN 1 ELSE 0 END +
             CASE WHEN flag_diabetes     = 1 THEN 1 ELSE 0 END +
             CASE WHEN flag_ischemic_heart = 1 THEN 1 ELSE 0 END +
             CASE WHEN flag_osteoporosis = 1 THEN 1 ELSE 0 END +
             CASE WHEN flag_ra_oa        = 1 THEN 1 ELSE 0 END +
             CASE WHEN flag_stroke       = 1 THEN 1 ELSE 0 END
            ) AS chronic_condition_count
        FROM analytics.beneficiary_summary
    ) bc ON bc.desynpuf_id = cc.desynpuf_id AND bc.year = cc.claim_year
    LEFT JOIN (
        SELECT at_physn_npi, claim_year, MAX(cnt) AS day_count
        FROM (
            SELECT at_physn_npi, claim_year,
                   cc.clm_from_dt::DATE AS claim_date,
                   COUNT(*) AS cnt
            FROM base_claims cc
            GROUP BY at_physn_npi, claim_year, cc.clm_from_dt::DATE
        ) daily
        GROUP BY at_physn_npi, claim_year
    ) daily_counts
    ON daily_counts.at_physn_npi = cc.at_physn_npi
   AND daily_counts.claim_year   = cc.claim_year
    GROUP BY cc.at_physn_npi, cc.claim_year
),

-- 3. Claims per beneficiary ratio
claims_per_bene AS (
    SELECT
        at_physn_npi,
        claim_year,
        AVG(bene_claims) AS avg_claims_per_beneficiary,
        AVG(bene_claims) AS carrier_claims_per_bene
    FROM (
        SELECT at_physn_npi, claim_year, desynpuf_id,
               COUNT(*) AS bene_claims
        FROM base_claims
        GROUP BY at_physn_npi, claim_year, desynpuf_id
    ) bene_counts
    GROUP BY at_physn_npi, claim_year
),

-- 4. Optimized HCPCS stats (window functions instead of correlated subqueries)
hcpcs_counts AS (
    SELECT at_physn_npi, claim_year, primary_hcpcs_cd, COUNT(*) as code_count
    FROM base_claims
    GROUP BY at_physn_npi, claim_year, primary_hcpcs_cd
),
hcpcs_ranked AS (
    SELECT at_physn_npi, claim_year, primary_hcpcs_cd, code_count,
           ROW_NUMBER() OVER(PARTITION BY at_physn_npi, claim_year ORDER BY code_count DESC) as rn,
           SUM(code_count) OVER(PARTITION BY at_physn_npi, claim_year) as total_claims
    FROM hcpcs_counts
),
hcpcs_stats AS (
    SELECT at_physn_npi, claim_year,
           MAX(CASE WHEN rn = 1 THEN primary_hcpcs_cd END) AS top_hcpcs_code,
           MAX(CASE WHEN rn = 1 THEN code_count::NUMERIC / total_claims END) AS top_hcpcs_code_share,
           SUM((code_count::NUMERIC / total_claims)^2) AS hcpcs_concentration_score
    FROM hcpcs_ranked
    GROUP BY at_physn_npi, claim_year
),

-- 5. Exact duplicate detection
exact_dups AS (
    SELECT at_physn_npi, claim_year, COUNT(*) AS exact_duplicate_count
    FROM (
        SELECT at_physn_npi, claim_year,
               desynpuf_id, primary_hcpcs_cd,
               clm_from_dt::DATE AS claim_date,
               COUNT(*) AS cnt
        FROM base_claims
        GROUP BY at_physn_npi, claim_year,
                 desynpuf_id, primary_hcpcs_cd,
                 clm_from_dt::DATE
        HAVING COUNT(*) > 1
    ) dups
    GROUP BY at_physn_npi, claim_year
),

-- 6. Near-duplicate detection (optimised date logic without self-joins).
--    NOTE: This CTE contains the heaviest sort in the query
--    (LAG OVER PARTITION BY at_physn_npi, desynpuf_id, primary_hcpcs_cd).
--    The active_npis filter above is what keeps this tractable — without it,
--    this sort ran over 9M+ rows and triggered the OOM SIGKILL.
near_dups AS (
    SELECT at_physn_npi, claim_year, SUM(is_near_dup) AS near_duplicate_count
    FROM (
        SELECT at_physn_npi, claim_year,
               CASE
                   WHEN (clm_from_dt::DATE - LAG(clm_from_dt::DATE)
                         OVER(PARTITION BY at_physn_npi, desynpuf_id, primary_hcpcs_cd
                              ORDER BY clm_from_dt::DATE)) BETWEEN 1 AND 3
                   THEN 1
                   ELSE 0
               END AS is_near_dup
        FROM base_claims
    ) lag_calc
    GROUP BY at_physn_npi, claim_year
),

-- 7. Prior year claim count for growth rate
prior_year AS (
    SELECT at_physn_npi,
           claim_year + 1   AS next_year,
           COUNT(*)         AS prior_period_claim_count
    FROM base_claims
    GROUP BY at_physn_npi, claim_year
),

-- 8. Outpatient stats per provider (standalone CTE — not filtered through
--    base_claims because outpatient_claims is a separate table)
op_stats AS (
    SELECT oc.at_physn_npi,
           EXTRACT(YEAR FROM oc.clm_from_dt::DATE)::SMALLINT AS claim_year,
           COUNT(*)         AS total_outpatient_claims,
           AVG(oc.clm_pmt_amt) AS avg_outpatient_payment
    FROM analytics.outpatient_claims oc
    WHERE oc.at_physn_npi IN (SELECT at_physn_npi FROM active_npis)
    GROUP BY oc.at_physn_npi, EXTRACT(YEAR FROM oc.clm_from_dt::DATE)::SMALLINT
)

-- Final join
SELECT
    cb.at_physn_npi,
    cb.claim_year                                      AS period_year,
    cb.total_carrier_claims,
    cb.distinct_beneficiaries,
    cb.distinct_hcpcs_codes,
    cb.distinct_pos_codes,
    cb.beneficiaries_per_state,
    cb.avg_submitted_to_allowed_ratio,
    cb.p95_submitted_to_allowed_ratio,
    cb.avg_submitted_charge,
    cb.avg_allowed_amt,
    cb.pct_weekend_claims,
    cb.max_claims_in_single_day,
    cb.claims_after_bene_death,
    cb.claims_in_bene_death_year,
    cb.high_chronic_burden_benes_pct,
    cb.pct_claims_office,
    cb.pct_claims_home,
    cb.pct_claims_nursing_facility,
    COALESCE(cpb.avg_claims_per_beneficiary, 0)        AS avg_claims_per_beneficiary,
    COALESCE(cpb.carrier_claims_per_bene, 0)           AS carrier_claims_per_bene,
    hs.top_hcpcs_code,
    COALESCE(hs.top_hcpcs_code_share, 0)               AS top_hcpcs_code_share,
    COALESCE(hs.hcpcs_concentration_score, 0)          AS hcpcs_concentration_score,
    COALESCE(ed.exact_duplicate_count, 0)              AS exact_duplicate_count,
    COALESCE(nd.near_duplicate_count, 0)               AS near_duplicate_count,
    CASE
        WHEN cb.total_carrier_claims > 0
        THEN (COALESCE(ed.exact_duplicate_count, 0) +
              COALESCE(nd.near_duplicate_count, 0))::NUMERIC
             / cb.total_carrier_claims
        ELSE 0
    END                                                AS duplicate_rate,
    COALESCE(py.prior_period_claim_count, 0)           AS prior_period_claim_count,
    CASE
        WHEN COALESCE(py.prior_period_claim_count, 0) > 0
        THEN (cb.total_carrier_claims - py.prior_period_claim_count)::NUMERIC
             / py.prior_period_claim_count * 100
        ELSE NULL
    END                                                AS claim_volume_growth_pct,
    COALESCE(op.total_outpatient_claims, 0)            AS total_outpatient_claims,
    COALESCE(op.avg_outpatient_payment, 0)             AS avg_outpatient_payment,
    :batch_id                                          AS batch_id

FROM carrier_base cb
LEFT JOIN claims_per_bene cpb
       ON cpb.at_physn_npi = cb.at_physn_npi
      AND cpb.claim_year   = cb.claim_year
LEFT JOIN hcpcs_stats hs
       ON hs.at_physn_npi  = cb.at_physn_npi
      AND hs.claim_year    = cb.claim_year
LEFT JOIN exact_dups ed
       ON ed.at_physn_npi  = cb.at_physn_npi
      AND ed.claim_year    = cb.claim_year
LEFT JOIN near_dups nd
       ON nd.at_physn_npi  = cb.at_physn_npi
      AND nd.claim_year    = cb.claim_year
LEFT JOIN prior_year py
       ON py.at_physn_npi  = cb.at_physn_npi
      AND py.next_year     = cb.claim_year
LEFT JOIN op_stats op
       ON op.at_physn_npi  = cb.at_physn_npi
      AND op.claim_year    = cb.claim_year
"""


def compute_provider_features(
    db_conn_str: str,
    batch_id: Optional[int] = None,
    # FIX: these two params scope the query to the current batch window.
    # batch_start_date: exclusive lower bound (the old watermark date).
    # batch_end_date:   inclusive upper bound (the new watermark date).
    # When both are None (e.g. CLI / full-refresh mode) the defaults cover
    # all dates, replicating the original full-table behaviour intentionally.
    batch_start_date: Optional[str] = None,
    batch_end_date: Optional[str] = None,
) -> int:
    """
    Runs all feature aggregations inside PostgreSQL.
    Only the final aggregated rows are pulled into Python.

    Args:
        db_conn_str:      SQLAlchemy connection string for the fraud_claims DB.
        batch_id:         Batch counter from the DAG run (written to output rows).
        batch_start_date: Exclusive lower-bound date string ('YYYY-MM-DD').
                          Only providers with claims AFTER this date are included.
                          Defaults to '1970-01-01' (full refresh).
        batch_end_date:   Inclusive upper-bound date string ('YYYY-MM-DD').
                          Only providers with claims ON OR BEFORE this date are
                          included. Defaults to '9999-12-31' (full refresh).
    """
    engine = create_engine(db_conn_str)

    # Default to full-refresh window when no date range is supplied.
    # This preserves backward-compatibility for CLI / manual runs.
    effective_start = batch_start_date or "1970-01-01"
    effective_end   = batch_end_date   or "9999-12-31"

    log.info(
        "Computing provider features via SQL aggregation in PostgreSQL "
        "(batch_start_date=%s, batch_end_date=%s, batch_id=%s)...",
        effective_start, effective_end, batch_id,
    )
    log.info("(All heavy computation runs inside the database — no large DataFrame loads)")

    with engine.connect() as conn:
        # Give PostgreSQL more sort memory for this session so the
        # LAG/OVER in near_dups doesn't spill to disk unnecessarily.
        conn.execute(text(f"SET work_mem = '{WORK_MEM}'"))

        df = pd.read_sql(
            text(FEATURES_SQL),
            conn,
            params={
                "batch_id":         batch_id or 0,
                "batch_start_date": effective_start,
                "batch_end_date":   effective_end,
            },
        )

    if df.empty:
        log.warning(
            "Feature query returned no rows for window %s → %s. "
            "If this is the first run, ensure DAG 1 (historical backfill) has completed.",
            effective_start, effective_end,
        )
        return 0

    log.info("SQL returned %d provider-year feature rows.", len(df))

    # Upsert: delete existing rows for these providers/years, then insert.
    npi_list  = df["at_physn_npi"].unique().tolist()
    year_list = [int(y) for y in df["period_year"].unique().tolist()]

    with engine.connect() as conn:
        if npi_list and year_list:
            conn.execute(
                text("""
                    DELETE FROM features.provider_features
                    WHERE at_physn_npi = ANY(:npis)
                      AND period_year  = ANY(:years)
                """),
                {"npis": npi_list, "years": year_list},
            )
        conn.commit()

    rows_written = 0
    for i in range(0, len(df), CHUNK_SIZE):
        chunk = df.iloc[i: i + CHUNK_SIZE]
        chunk.to_sql(
            "provider_features", engine,
            schema="features", if_exists="append",
            index=False, method="multi",
        )
        rows_written += len(chunk)
        log.info("Written %d / %d rows...", rows_written, len(df))

    log.info("Feature engineering complete. %d provider-year rows written.", rows_written)
    return rows_written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute provider features via SQL.")
    parser.add_argument("--batch-id",         type=int,   default=None)
    parser.add_argument(
        "--batch-start-date",
        type=str, default=None,
        help="Exclusive lower-bound date (YYYY-MM-DD). Omit for full refresh.",
    )
    parser.add_argument(
        "--batch-end-date",
        type=str, default=None,
        help="Inclusive upper-bound date (YYYY-MM-DD). Omit for full refresh.",
    )
    args = parser.parse_args()
    compute_provider_features(
        db_conn_str=_get_db_conn_str(),
        batch_id=args.batch_id,
        batch_start_date=args.batch_start_date,
        batch_end_date=args.batch_end_date,
    )