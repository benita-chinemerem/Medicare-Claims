"""
scripts/etl/feature_engineering.py

Provider-level feature computation — SQL-based approach.

All heavy aggregations run inside PostgreSQL. Python only receives the
final aggregated provider-year rows (typically 10K-50K rows), never the
full 9M+ raw claim dataset. This avoids the OOM kill that occurs when
loading all carrier claims into a pandas DataFrame.

Called by DAG 2 (dag2_weekly_scoring.py) on each weekly run.
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
# ---------------------------------------------------------------------------

FEATURES_SQL = """
WITH

-- 1. Create a base layer to parse dates efficiently so we don't repeat work
base_claims AS (
    SELECT 
        cc.*,
        LEFT(cc.clm_from_dt, 4)::INT AS claim_year,
        -- ISODOW returns 6 for Saturday, 7 for Sunday
        CASE WHEN EXTRACT(ISODOW FROM cc.clm_from_dt::DATE) IN (6, 7) THEN True ELSE False END AS is_weekend_claim
    FROM analytics.carrier_claims cc
    {where_clause_base}
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
                  AND cc.clm_from_dt > bs.death_date
             THEN 1 ELSE 0 END)                                       AS claims_after_bene_death,
        SUM(CASE WHEN bs.death_date IS NOT NULL
                  AND cc.claim_year = LEFT(bs.death_date, 4)::INT
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

-- 4. Optimized HCPCS stats (Using Window Functions instead of Correlated Subqueries)
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

-- 6. Near-duplicate detection (Optimized Date Logic)
near_dups AS (
    SELECT a.at_physn_npi, a.claim_year,
           COUNT(*) AS near_duplicate_count
    FROM base_claims a
    JOIN base_claims b
      ON  a.at_physn_npi      = b.at_physn_npi
      AND a.desynpuf_id       = b.desynpuf_id
      AND a.primary_hcpcs_cd  = b.primary_hcpcs_cd
      AND a.clm_id            < b.clm_id
      AND a.clm_from_dt::DATE - b.clm_from_dt::DATE BETWEEN -3 AND 3
    GROUP BY a.at_physn_npi, a.claim_year
),

-- 7. Prior year claim count for growth rate
prior_year AS (
    SELECT at_physn_npi,
           claim_year + 1   AS next_year,
           COUNT(*)         AS prior_period_claim_count
    FROM base_claims
    GROUP BY at_physn_npi, claim_year
),

-- 8. Outpatient stats per provider (Standalone CTE)
op_stats AS (
    SELECT at_physn_npi,
           LEFT(clm_from_dt, 4)::SMALLINT AS claim_year,
           COUNT(*)         AS total_outpatient_claims,
           AVG(clm_pmt_amt) AS avg_outpatient_payment
    FROM analytics.outpatient_claims
    WHERE at_physn_npi IS NOT NULL
    GROUP BY at_physn_npi, LEFT(clm_from_dt, 4)::SMALLINT
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
) -> int:
    """
    Runs all feature aggregations inside PostgreSQL.
    Only the final aggregated rows are pulled into Python.
    """
    engine = create_engine(db_conn_str)

    # For batch runs, we could add WHERE clauses to restrict to recent data.
    # Updated to perfectly match the new optimized base_claims CTE
    where_clause_base = ""

    sql = FEATURES_SQL.format(
        where_clause_base=where_clause_base,
    )

    log.info("Computing provider features via SQL aggregation in PostgreSQL...")
    log.info("(All heavy computation runs inside the database — no large DataFrame loads)")

    df = pd.read_sql(
        text(sql),
        engine,
        params={"batch_id": batch_id or 0},
    )

    if df.empty:
        log.warning("Feature query returned no rows. Ensure analytics tables are populated.")
        return 0

    log.info("SQL returned %d provider-year feature rows.", len(df))

    # Upsert: delete existing rows for these providers/years, then insert
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
    parser.add_argument("--batch-id", type=int, default=None)
    args = parser.parse_args()
    compute_provider_features(
        db_conn_str=_get_db_conn_str(),
        batch_id=args.batch_id,
    )