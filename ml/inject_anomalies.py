"""
ml/inject_anomalies.py

Controlled anomaly injection for the supervised demonstration layer.

Three fraud scenario types are injected into a copy of the analytics data:

    1. UPCODING — A subset of carrier claims has its primary HCPCS code
       replaced with a higher-reimbursing code in the same category.
       Provider-level features for affected providers shift measurably:
       avg_submitted_to_allowed_ratio and avg_submitted_charge increase.

    2. PHANTOM BILLING — A set of claims is duplicated with dates shifted
       by a random 1-to-5 day offset, simulating "same service, new date"
       phantom billing. Affects duplicate_rate and near_duplicate_count.

    3. DUPLICATE SUBMISSION — Exact duplicate claim lines are injected for
       a small percentage of providers, simulating resubmission of
       already-paid claims. Affects exact_duplicate_count.

Output:
    A labeled Parquet file at --output containing provider-year feature vectors
    with a binary 'label' column (0 = clean, 1 = anomaly) for each scenario type,
    plus a 'scenario' column for per-type evaluation.

Usage:
    python ml/inject_anomalies.py --output data/injected/ --seed 42

IMPORTANT:
    Injected data is a synthetic construct layered on top of already-synthetic
    DE-SynPUF data. Results from models trained here demonstrate the supervised
    architecture; they do not represent real fraud detection performance.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
from typing import Tuple

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Fraction of providers to inject into per scenario
UPCODING_RATE       = 0.04   # 4% of providers
PHANTOM_RATE        = 0.03   # 3%
DUPLICATE_RATE_INJ  = 0.03   # 3%

# HCPCS upcoding map: maps a common lower-paying code to a higher-paying substitute
# These are illustrative code pairs — the DE-SynPUF coarsens actual codes.
UPCODING_MAP = {
    "99213": "99215",  # Office visit, low complexity -> high complexity
    "99203": "99205",
    "99212": "99214",
    "93000": "93010",
    "71046": "71048",
}


def _get_db_conn_str() -> str:
    return (
        f"postgresql+psycopg2://{os.environ.get('FRAUD_DB_USER', 'airflow')}:"
        f"{os.environ.get('FRAUD_DB_PASSWORD', 'airflow')}@"
        f"{os.environ.get('FRAUD_DB_HOST', 'postgres')}:"
        f"{os.environ.get('FRAUD_DB_PORT', '5432')}/"
        f"{os.environ.get('FRAUD_DB_NAME', 'fraud_claims')}"
    )


def _load_base_data(engine) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads carrier claims and provider features from Postgres for injection.
    Returns (claims_df, features_df).
    """
    claims_df = pd.read_sql(
        text("""
            SELECT clm_id, desynpuf_id, at_physn_npi, clm_from_dt,
                   submitted_charge_amt, allowed_amt,
                   submitted_to_allowed_ratio, primary_hcpcs_cd,
                   is_weekend_claim, claim_year
            FROM analytics.carrier_claims
            ORDER BY clm_from_dt ASC
            LIMIT 2000000
        """),
        engine,
    )

    features_df = pd.read_sql(
        text("SELECT * FROM features.provider_features"),
        engine,
    )
    return claims_df, features_df


def inject_upcoding(
    claims_df: pd.DataFrame,
    features_df: pd.DataFrame,
    rate: float = UPCODING_RATE,
    rng: np.random.Generator = None,
) -> pd.DataFrame:
    """
    Injects upcoding anomaly: replaces primary HCPCS codes with higher-paying
    substitutes for a random sample of providers. Recomputes affected features.

    Returns a modified copy of features_df with 'label' and 'scenario' columns.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    all_npis     = features_df["at_physn_npi"].unique()
    target_npis  = set(rng.choice(all_npis, size=max(1, int(len(all_npis) * rate)), replace=False))

    claims_copy  = claims_df.copy()
    mask         = claims_copy["at_physn_npi"].isin(target_npis)
    upcoded_mask = mask & claims_copy["primary_hcpcs_cd"].isin(UPCODING_MAP)

    claims_copy.loc[upcoded_mask, "primary_hcpcs_cd"] = \
        claims_copy.loc[upcoded_mask, "primary_hcpcs_cd"].map(UPCODING_MAP)

    # Inflate submitted_charge_amt for upcoded claims by 20-40%
    multipliers = rng.uniform(1.20, 1.40, size=upcoded_mask.sum())
    claims_copy.loc[upcoded_mask, "submitted_charge_amt"] *= multipliers

    # Recompute affected provider-level features
    provider_stats = (
        claims_copy[claims_copy["at_physn_npi"].isin(target_npis)]
        .groupby("at_physn_npi")
        .agg(
            avg_submitted_charge=("submitted_charge_amt", "mean"),
            avg_submitted_to_allowed_ratio=("submitted_to_allowed_ratio", "mean"),
        )
        .reset_index()
    )

    features_copy = features_df.copy()
    for _, row in provider_stats.iterrows():
        npi = row["at_physn_npi"]
        idx = features_copy["at_physn_npi"] == npi
        features_copy.loc[idx, "avg_submitted_to_allowed_ratio"] = row["avg_submitted_to_allowed_ratio"]
        features_copy.loc[idx, "avg_submitted_charge"]           = row["avg_submitted_charge"]

    features_copy["label"]    = features_copy["at_physn_npi"].isin(target_npis).astype(int)
    features_copy["scenario"] = "upcoding"
    log.info("Upcoding injection: %d providers affected.", len(target_npis))
    return features_copy


def inject_phantom_billing(
    claims_df: pd.DataFrame,
    features_df: pd.DataFrame,
    rate: float = PHANTOM_RATE,
    rng: np.random.Generator = None,
) -> pd.DataFrame:
    """
    Injects phantom billing: duplicates claims with a 1-5 day date shift.
    Recomputes near_duplicate_count and duplicate_rate for affected providers.
    """
    if rng is None:
        rng = np.random.default_rng(43)

    all_npis    = features_df["at_physn_npi"].unique()
    target_npis = set(rng.choice(all_npis, size=max(1, int(len(all_npis) * rate)), replace=False))

    target_claims = claims_df[claims_df["at_physn_npi"].isin(target_npis)].copy()

    # Sample 30% of each targeted provider's claims to phantom-duplicate
    phantom_rows = []
    for npi, group in target_claims.groupby("at_physn_npi"):
        n_phantom = max(1, int(len(group) * 0.30))
        sample    = group.sample(n=min(n_phantom, len(group)), random_state=0)
        sample    = sample.copy()
        shifts    = rng.integers(1, 6, size=len(sample))
        sample["clm_from_dt"] = pd.to_datetime(sample["clm_from_dt"]) + \
                                 pd.to_timedelta(shifts, unit="D")
        phantom_rows.append(sample)

    if phantom_rows:
        phantoms = pd.concat(phantom_rows, ignore_index=True)

        # Recompute near_duplicate_count for targeted providers
        phantom_counts = phantoms.groupby("at_physn_npi").size().reset_index(name="phantom_count")

        features_copy = features_df.copy()
        for _, row in phantom_counts.iterrows():
            npi = row["at_physn_npi"]
            idx = features_copy["at_physn_npi"] == npi
            current_near_dup = features_copy.loc[idx, "near_duplicate_count"].values
            current_total    = features_copy.loc[idx, "total_carrier_claims"].values
            new_near_dup     = (current_near_dup + row["phantom_count"]).clip(0)
            new_total        = (current_total + row["phantom_count"]).clip(1)
            features_copy.loc[idx, "near_duplicate_count"] = new_near_dup
            features_copy.loc[idx, "duplicate_rate"]       = new_near_dup / new_total
    else:
        features_copy = features_df.copy()

    features_copy["label"]    = features_copy["at_physn_npi"].isin(target_npis).astype(int)
    features_copy["scenario"] = "phantom_billing"
    log.info("Phantom billing injection: %d providers affected.", len(target_npis))
    return features_copy


def inject_duplicate_submission(
    claims_df: pd.DataFrame,
    features_df: pd.DataFrame,
    rate: float = DUPLICATE_RATE_INJ,
    rng: np.random.Generator = None,
) -> pd.DataFrame:
    """
    Injects exact duplicate submissions: adds identical claim lines for a
    subset of providers. Recomputes exact_duplicate_count and duplicate_rate.
    """
    if rng is None:
        rng = np.random.default_rng(44)

    all_npis    = features_df["at_physn_npi"].unique()
    target_npis = set(rng.choice(all_npis, size=max(1, int(len(all_npis) * rate)), replace=False))

    target_claims = claims_df[claims_df["at_physn_npi"].isin(target_npis)].copy()
    dup_counts    = {}

    for npi, group in target_claims.groupby("at_physn_npi"):
        n_dups = max(1, int(len(group) * 0.15))
        dup_counts[npi] = n_dups

    features_copy = features_df.copy()
    for npi, n_dups in dup_counts.items():
        idx = features_copy["at_physn_npi"] == npi
        current_exact = features_copy.loc[idx, "exact_duplicate_count"].values
        current_total = features_copy.loc[idx, "total_carrier_claims"].values
        new_exact     = (current_exact + n_dups).clip(0)
        new_total     = (current_total + n_dups).clip(1)
        features_copy.loc[idx, "exact_duplicate_count"] = new_exact
        features_copy.loc[idx, "duplicate_rate"]        = new_exact / new_total

    features_copy["label"]    = features_copy["at_physn_npi"].isin(target_npis).astype(int)
    features_copy["scenario"] = "duplicate_submission"
    log.info("Duplicate submission injection: %d providers affected.", len(target_npis))
    return features_copy


def run_all_injections(output_dir: str, seed: int = 42) -> None:
    """
    Runs all three injection scenarios and writes the combined labeled dataset
    to output_dir as a Parquet file.
    """
    os.makedirs(output_dir, exist_ok=True)
    engine = create_engine(_get_db_conn_str())
    rng    = np.random.default_rng(seed)

    log.info("Loading base data from Postgres...")
    claims_df, features_df = _load_base_data(engine)

    if features_df.empty:
        raise ValueError(
            "features.provider_features is empty. "
            "Run DAG 1 and at least one DAG 2 cycle before injecting anomalies."
        )

    log.info("Running injection scenarios...")
    scenarios = [
        inject_upcoding(claims_df, features_df, rng=rng),
        inject_phantom_billing(claims_df, features_df, rng=rng),
        inject_duplicate_submission(claims_df, features_df, rng=rng),
    ]

    combined = pd.concat(scenarios, ignore_index=True)
    out_path = os.path.join(output_dir, "injected_labeled_features.parquet")
    combined.to_parquet(out_path, index=False)

    total_anomalies = combined["label"].sum()
    log.info(
        "Injection complete. %d total rows | %d anomalies (%.1f%%) | saved to %s",
        len(combined), total_anomalies,
        total_anomalies / len(combined) * 100,
        out_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject anomaly scenarios into DE-SynPUF features.")
    parser.add_argument("--output", required=True, help="Output directory for labeled Parquet")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    run_all_injections(args.output, seed=args.seed)
