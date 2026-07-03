"""
scripts/etl/inject_anomalies.py

Creates a labeled dataset for XGBoost supervised training by injecting
three synthetic fraud anomaly types into a copy of the DE-SynPUF carrier
claims, then recomputing provider-level features on the modified data.

Anomaly scenarios (Section 6.4 of project spec):
  1. Upcoding          — inflated submitted charges for selected providers
  2. Phantom billing   — near-duplicate claims with 1-5 day date shifts
  3. Duplicate submission — exact duplicate claims for selected providers

Providers are assigned to mutually exclusive groups so labels are unambiguous:
  Group 0 — Clean           (~85% of providers)
  Group 1 — Upcoding        (~5%)
  Group 2 — Phantom billing  (~5%)
  Group 3 — Duplicate submission (~5%)

All injection happens on the `injected.carrier_claims` table (a copy of
analytics.carrier_claims) — the production analytics schema is never touched.

Called by DAG 3 (dag3_monthly_retraining.py → run_supervised_demo task).
"""

from __future__ import annotations

import logging
import os
import random
from typing import List

import pandas as pd
from sqlalchemy import create_engine, text

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── Injection parameters ──────────────────────────────────────────────────────
UPCODING_FACTOR  = 1.45    # submitted charges inflated by 45%
PHANTOM_RATE     = 0.30    # 30% of a provider's claims get a phantom copy
DUPLICATE_RATE   = 0.20    # 20% of claims get an exact duplicate
DATE_SHIFT_MIN   = 1       # phantom date shift minimum (days)
DATE_SHIFT_MAX   = 5       # phantom date shift maximum (days)
UPCODING_PCT     = 0.05    # fraction of all NPIs to upcode
PHANTOM_PCT      = 0.05    # fraction of all NPIs to phantom-bill
DUPLICATE_PCT    = 0.05    # fraction of all NPIs to duplicate-submit
RANDOM_SEED      = 42
# ─────────────────────────────────────────────────────────────────────────────


def _get_db_conn_str() -> str:
    return (
        f"postgresql+psycopg2://{os.environ.get('FRAUD_DB_USER', 'airflow')}:"
        f"{os.environ.get('FRAUD_DB_PASSWORD', 'airflow')}@"
        f"{os.environ.get('FRAUD_DB_HOST', 'postgres')}:"
        f"{os.environ.get('FRAUD_DB_PORT', '5432')}/"
        f"{os.environ.get('FRAUD_DB_NAME', 'fraud_claims')}"
    )


# ── Group assignment ──────────────────────────────────────────────────────────

def _get_all_npis(conn) -> List[str]:
    result = conn.execute(text(
        "SELECT DISTINCT at_physn_npi "
        "FROM analytics.carrier_claims "
        "WHERE at_physn_npi IS NOT NULL"
    ))
    return [row[0] for row in result.fetchall()]


def _assign_groups(npis: List[str]) -> pd.DataFrame:
    """
    Randomly shuffle all NPIs and assign mutually exclusive anomaly groups.
    Returns a DataFrame with columns: at_physn_npi, anomaly_group (0-3).
    """
    rng = random.Random(RANDOM_SEED)
    shuffled = npis.copy()
    rng.shuffle(shuffled)
    n       = len(shuffled)
    n_up    = int(n * UPCODING_PCT)
    n_ph    = int(n * PHANTOM_PCT)
    n_dup   = int(n * DUPLICATE_PCT)

    groups = {}
    for npi in shuffled[:n_up]:
        groups[npi] = 1                           # upcoding
    for npi in shuffled[n_up : n_up + n_ph]:
        groups[npi] = 2                           # phantom billing
    for npi in shuffled[n_up + n_ph : n_up + n_ph + n_dup]:
        groups[npi] = 3                           # duplicate submission
    for npi in shuffled[n_up + n_ph + n_dup :]:
        groups[npi] = 0                           # clean

    df = pd.DataFrame(
        [{"at_physn_npi": npi, "anomaly_group": grp} for npi, grp in groups.items()]
    )
    log.info(
        "Group assignment — clean: %d | upcoding: %d | phantom: %d | duplicate: %d",
        (df.anomaly_group == 0).sum(),
        (df.anomaly_group == 1).sum(),
        (df.anomaly_group == 2).sum(),
        (df.anomaly_group == 3).sum(),
    )
    return df


# ── Schema setup ──────────────────────────────────────────────────────────────

def _setup_injected_schema(conn):
    """
    Create a fresh copy of analytics.carrier_claims in the injected schema.
    Dropping and recreating makes this idempotent — safe to re-run.
    """
    conn.execute(text("CREATE SCHEMA IF NOT EXISTS injected;"))
    conn.execute(text("DROP TABLE IF EXISTS injected.carrier_claims;"))
    conn.execute(text("""
        CREATE TABLE injected.carrier_claims AS
        SELECT * FROM analytics.carrier_claims;
    """))
    conn.execute(text("""
        CREATE INDEX idx_inj_npi  ON injected.carrier_claims (at_physn_npi);
        CREATE INDEX idx_inj_date ON injected.carrier_claims (clm_from_dt);
    """))
    log.info("injected.carrier_claims created and indexed.")


# ── Injection functions ───────────────────────────────────────────────────────

def _inject_upcoding(conn, npis: List[str]):
    """
    Inflate submitted_charge_amt by UPCODING_FACTOR for all claims
    belonging to upcoding providers, and recompute submitted_to_allowed_ratio.
    """
    if not npis:
        return
    conn.execute(text("""
        UPDATE injected.carrier_claims
        SET
            submitted_charge_amt = ROUND(
                submitted_charge_amt::NUMERIC * :factor, 2
            ),
            submitted_to_allowed_ratio = CASE
                WHEN allowed_amt::NUMERIC > 0
                THEN ROUND(
                    (submitted_charge_amt::NUMERIC * :factor)
                    / allowed_amt::NUMERIC, 4
                )
                ELSE submitted_to_allowed_ratio
            END
        WHERE at_physn_npi = ANY(:npis)
    """), {"factor": UPCODING_FACTOR, "npis": npis})
    log.info("Upcoding injected for %d providers (factor %.2f).", len(npis), UPCODING_FACTOR)


def _inject_phantom_billing(conn, npis: List[str]):
    """
    Duplicate a random PHANTOM_RATE fraction of each phantom provider's
    claims with a random 1-5 day date shift. The duplicate clm_id gets a
    '_ph' suffix to keep it unique and distinguishable during debugging.
    """
    if not npis:
        return
    conn.execute(text("""
        INSERT INTO injected.carrier_claims (
            clm_id, desynpuf_id, at_physn_npi, primary_hcpcs_cd,
            place_of_service_cd, submitted_charge_amt, allowed_amt,
            submitted_to_allowed_ratio, clm_from_dt
        )
        SELECT
            clm_id || '_ph'                                      AS clm_id,
            desynpuf_id,
            at_physn_npi,
            primary_hcpcs_cd,
            place_of_service_cd,
            submitted_charge_amt,
            allowed_amt,
            submitted_to_allowed_ratio,
            (clm_from_dt::DATE
             + (FLOOR(RANDOM() * :shift_range + :shift_min)::INT
                || ' days')::INTERVAL
            )::TEXT                                              AS clm_from_dt
        FROM analytics.carrier_claims
        WHERE at_physn_npi = ANY(:npis)
          AND RANDOM() < :rate
    """), {
        "npis":        npis,
        "rate":        PHANTOM_RATE,
        "shift_min":   DATE_SHIFT_MIN,
        "shift_range": DATE_SHIFT_MAX - DATE_SHIFT_MIN,
    })
    log.info(
        "Phantom billing injected for %d providers (%.0f%% of claims duplicated, +%d–%d days).",
        len(npis), PHANTOM_RATE * 100, DATE_SHIFT_MIN, DATE_SHIFT_MAX,
    )


def _inject_duplicate_submission(conn, npis: List[str]):
    """
    Exact-duplicate a random DUPLICATE_RATE fraction of each provider's
    claims. The duplicate clm_id gets a '_dup' suffix.
    """
    if not npis:
        return
    conn.execute(text("""
        INSERT INTO injected.carrier_claims (
            clm_id, desynpuf_id, at_physn_npi, primary_hcpcs_cd,
            place_of_service_cd, submitted_charge_amt, allowed_amt,
            submitted_to_allowed_ratio, clm_from_dt
        )
        SELECT
            clm_id || '_dup'  AS clm_id,
            desynpuf_id,
            at_physn_npi,
            primary_hcpcs_cd,
            place_of_service_cd,
            submitted_charge_amt,
            allowed_amt,
            submitted_to_allowed_ratio,
            clm_from_dt
        FROM analytics.carrier_claims
        WHERE at_physn_npi = ANY(:npis)
          AND RANDOM() < :rate
    """), {"npis": npis, "rate": DUPLICATE_RATE})
    log.info(
        "Duplicate submission injected for %d providers (%.0f%% of claims duplicated).",
        len(npis), DUPLICATE_RATE * 100,
    )


# ── Feature recomputation on injected data ────────────────────────────────────

def _compute_injected_features(conn) -> pd.DataFrame:
    """
    Recompute the subset of provider features that are affected by the
    three injection types, running entirely inside PostgreSQL.

    Returns a DataFrame with one row per (at_physn_npi, claim_year).
    """
    log.info("Recomputing provider features on injected.carrier_claims ...")
    df = pd.read_sql(text("""
        WITH base AS (
            SELECT
                at_physn_npi,
                EXTRACT(YEAR FROM clm_from_dt::DATE)::INT            AS claim_year,
                desynpuf_id,
                primary_hcpcs_cd,
                clm_from_dt::DATE                                    AS claim_date,
                submitted_charge_amt::FLOAT                          AS sub_charge,
                allowed_amt::FLOAT                                   AS alw_amt,
                submitted_to_allowed_ratio::FLOAT                    AS sub_ratio,
                CASE WHEN EXTRACT(ISODOW FROM clm_from_dt::DATE)
                     IN (6, 7) THEN 1 ELSE 0 END                    AS is_weekend
            FROM injected.carrier_claims
            WHERE at_physn_npi IS NOT NULL
        ),

        daily_max AS (
            SELECT at_physn_npi, claim_year,
                   MAX(day_cnt) AS max_claims_in_single_day
            FROM (
                SELECT at_physn_npi, claim_year, claim_date, COUNT(*) AS day_cnt
                FROM base
                GROUP BY at_physn_npi, claim_year, claim_date
            ) d
            GROUP BY at_physn_npi, claim_year
        ),

        base_agg AS (
            SELECT
                b.at_physn_npi,
                b.claim_year,
                COUNT(*)                                              AS total_carrier_claims,
                COUNT(DISTINCT b.desynpuf_id)                        AS distinct_beneficiaries,
                COUNT(DISTINCT b.primary_hcpcs_cd)                   AS distinct_hcpcs_codes,
                AVG(b.sub_ratio)                                     AS avg_submitted_to_allowed_ratio,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY b.sub_ratio)
                                                                     AS p95_submitted_to_allowed_ratio,
                AVG(b.is_weekend::FLOAT)                             AS pct_weekend_claims,
                dm.max_claims_in_single_day
            FROM base b
            JOIN daily_max dm
              ON dm.at_physn_npi = b.at_physn_npi
             AND dm.claim_year   = b.claim_year
            GROUP BY b.at_physn_npi, b.claim_year, dm.max_claims_in_single_day
        ),

        bene_counts AS (
            SELECT at_physn_npi, claim_year,
                   AVG(bc)::FLOAT AS avg_claims_per_beneficiary
            FROM (
                SELECT at_physn_npi, claim_year, desynpuf_id, COUNT(*) AS bc
                FROM base GROUP BY at_physn_npi, claim_year, desynpuf_id
            ) x
            GROUP BY at_physn_npi, claim_year
        ),

        hcpcs_stats AS (
            SELECT at_physn_npi, claim_year,
                   MAX(CASE WHEN rn = 1 THEN share END)   AS top_hcpcs_code_share,
                   SUM(share * share)                     AS hcpcs_concentration_score
            FROM (
                SELECT at_physn_npi, claim_year, primary_hcpcs_cd,
                       cnt::FLOAT / SUM(cnt) OVER (
                           PARTITION BY at_physn_npi, claim_year
                       )                                  AS share,
                       ROW_NUMBER() OVER (
                           PARTITION BY at_physn_npi, claim_year
                           ORDER BY cnt DESC
                       )                                  AS rn
                FROM (
                    SELECT at_physn_npi, claim_year,
                           primary_hcpcs_cd, COUNT(*) AS cnt
                    FROM base
                    GROUP BY at_physn_npi, claim_year, primary_hcpcs_cd
                ) c
            ) ranked
            GROUP BY at_physn_npi, claim_year
        ),

        exact_dups AS (
            SELECT at_physn_npi, claim_year,
                   COUNT(*) AS exact_duplicate_count
            FROM (
                SELECT at_physn_npi, claim_year,
                       desynpuf_id, primary_hcpcs_cd, claim_date
                FROM base
                GROUP BY at_physn_npi, claim_year,
                         desynpuf_id, primary_hcpcs_cd, claim_date
                HAVING COUNT(*) > 1
            ) e
            GROUP BY at_physn_npi, claim_year
        ),

        near_dups AS (
            SELECT at_physn_npi, claim_year,
                   SUM(is_nd) AS near_duplicate_count
            FROM (
                SELECT at_physn_npi, claim_year,
                       CASE
                           WHEN (claim_date
                                 - LAG(claim_date) OVER (
                                     PARTITION BY at_physn_npi, desynpuf_id,
                                                  primary_hcpcs_cd, claim_year
                                     ORDER BY claim_date
                                   )
                                ) BETWEEN 1 AND 3
                           THEN 1 ELSE 0
                       END AS is_nd
                FROM base
            ) lag_sub
            GROUP BY at_physn_npi, claim_year
        )

        SELECT
            ba.at_physn_npi,
            ba.claim_year                                            AS period_year,
            ba.total_carrier_claims,
            ba.distinct_beneficiaries,
            ba.distinct_hcpcs_codes,
            ba.avg_submitted_to_allowed_ratio,
            ba.p95_submitted_to_allowed_ratio,
            ba.pct_weekend_claims,
            ba.max_claims_in_single_day,
            COALESCE(bc.avg_claims_per_beneficiary, 0)              AS avg_claims_per_beneficiary,
            COALESCE(hs.top_hcpcs_code_share, 0)                   AS top_hcpcs_code_share,
            COALESCE(hs.hcpcs_concentration_score, 0)              AS hcpcs_concentration_score,
            COALESCE(ed.exact_duplicate_count, 0)                  AS exact_duplicate_count,
            COALESCE(nd.near_duplicate_count, 0)                   AS near_duplicate_count,
            CASE
                WHEN ba.total_carrier_claims > 0
                THEN (
                    COALESCE(ed.exact_duplicate_count, 0)
                    + COALESCE(nd.near_duplicate_count, 0)
                )::FLOAT / ba.total_carrier_claims
                ELSE 0
            END                                                     AS duplicate_rate
        FROM base_agg    ba
        LEFT JOIN bene_counts  bc USING (at_physn_npi, claim_year)
        LEFT JOIN hcpcs_stats  hs USING (at_physn_npi, claim_year)
        LEFT JOIN exact_dups   ed USING (at_physn_npi, claim_year)
        LEFT JOIN near_dups    nd USING (at_physn_npi, claim_year)
    """), conn)

    log.info("Injected feature query returned %d provider-year rows.", len(df))
    return df


# ── Main entry point ──────────────────────────────────────────────────────────

def run_injection(db_conn_str: str) -> pd.DataFrame:
    """
    Full pipeline: assign groups → copy data → inject anomalies →
    recompute features → attach labels.

    Returns:
        labeled_df — one row per (NPI, year) with:
            - all feature columns recomputed on injected data
            - anomaly_label  : int (0 = clean, 1 = anomaly)
            - anomaly_scenario: str ('clean' | 'upcoding' | 'phantom' | 'duplicate')
    """
    engine = create_engine(db_conn_str)

    with engine.connect() as conn:
        npis = _get_all_npis(conn)

    log.info("Total unique provider NPIs in analytics.carrier_claims: %d", len(npis))

    group_df = _assign_groups(npis)

    upcoding_npis  = group_df.loc[group_df.anomaly_group == 1, "at_physn_npi"].tolist()
    phantom_npis   = group_df.loc[group_df.anomaly_group == 2, "at_physn_npi"].tolist()
    duplicate_npis = group_df.loc[group_df.anomaly_group == 3, "at_physn_npi"].tolist()

    # All DDL + DML in a single transaction so partial injection never persists
    with engine.begin() as conn:
        _setup_injected_schema(conn)
        _inject_upcoding(conn, upcoding_npis)
        _inject_phantom_billing(conn, phantom_npis)
        _inject_duplicate_submission(conn, duplicate_npis)

    with engine.connect() as conn:
        features_df = _compute_injected_features(conn)

    # Attach labels
    labeled = features_df.merge(
        group_df[["at_physn_npi", "anomaly_group"]],
        on="at_physn_npi",
        how="left",
    )
    labeled["anomaly_group"] = labeled["anomaly_group"].fillna(0).astype(int)

    scenario_map = {0: "clean", 1: "upcoding", 2: "phantom", 3: "duplicate"}
    labeled["anomaly_scenario"] = labeled["anomaly_group"].map(scenario_map)
    labeled["anomaly_label"]    = (labeled["anomaly_group"] > 0).astype(int)

    log.info(
        "Labeled dataset ready: %d rows | %d anomalies (%.1f%%)",
        len(labeled),
        labeled["anomaly_label"].sum(),
        labeled["anomaly_label"].mean() * 100,
    )
    return labeled


if __name__ == "__main__":
    import os
    conn_str = (
        f"postgresql+psycopg2://{os.environ.get('FRAUD_DB_USER','airflow')}:"
        f"{os.environ.get('FRAUD_DB_PASSWORD','airflow')}@"
        f"{os.environ.get('FRAUD_DB_HOST','localhost')}:5432/"
        f"{os.environ.get('FRAUD_DB_NAME','fraud_claims')}"
    )
    df = run_injection(conn_str)
    print(df["anomaly_scenario"].value_counts())
