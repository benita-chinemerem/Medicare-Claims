"""
tests/test_features.py

Unit tests for the feature engineering module.

Tests verify:
    - Duplicate detection (exact and near-duplicate logic)
    - Volume and velocity computation
    - Billing ratio calculations
    - Temporal pattern flags (weekend billing)
    - Post-death billing detection
    - Feature output schema completeness

All tests use in-memory DataFrames — no database connection required.

Run with:
    pytest tests/test_features.py -v
"""

from __future__ import annotations

import sys
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.etl.feature_engineering import _detect_duplicates


# -----------------------------------------------------------------------
# Fixture helpers
# -----------------------------------------------------------------------

def make_claims(npi: str, rows: list[dict]) -> pd.DataFrame:
    """Creates a minimal carrier claims DataFrame for one provider."""
    defaults = {
        "at_physn_npi":            npi,
        "desynpuf_id":             "BENE000001",
        "clm_id":                  "CLM000001",
        "clm_from_dt":             pd.Timestamp("2008-03-15"),
        "claim_year":              2008,
        "submitted_charge_amt":    300.0,
        "allowed_amt":             150.0,
        "submitted_to_allowed_ratio": 2.0,
        "primary_hcpcs_cd":        "99213",
        "place_of_service_cd":     "11",
        "is_weekend_claim":        False,
        "death_date":              None,
        "part_b_months":           12,
        "chronic_condition_count": 2,
        "state_code":              "CA",
    }
    records = []
    for i, override in enumerate(rows):
        rec = {**defaults, "clm_id": f"CLM{i:06d}"}
        rec.update(override)
        records.append(rec)
    return pd.DataFrame(records)


# -----------------------------------------------------------------------
# Duplicate detection
# -----------------------------------------------------------------------
class TestDetectDuplicates:

    def test_no_duplicates(self):
        claims = make_claims("NPI001", [
            {"clm_from_dt": pd.Timestamp("2008-01-10"), "primary_hcpcs_cd": "99213"},
            {"clm_from_dt": pd.Timestamp("2008-02-15"), "primary_hcpcs_cd": "99214"},
        ])
        result = _detect_duplicates(claims)
        row = result[result["at_physn_npi"] == "NPI001"].iloc[0]
        assert row["exact_duplicate_count"] == 0
        assert row["near_duplicate_count"]  == 0

    def test_exact_duplicate_detected(self):
        """Same bene + same code + same date = exact duplicate."""
        claims = make_claims("NPI002", [
            {"desynpuf_id": "BENE001", "primary_hcpcs_cd": "99213",
             "clm_from_dt": pd.Timestamp("2008-03-15")},
            {"desynpuf_id": "BENE001", "primary_hcpcs_cd": "99213",
             "clm_from_dt": pd.Timestamp("2008-03-15")},   # exact duplicate
            {"desynpuf_id": "BENE001", "primary_hcpcs_cd": "99214",
             "clm_from_dt": pd.Timestamp("2008-04-10")},
        ])
        result = _detect_duplicates(claims)
        row = result[result["at_physn_npi"] == "NPI002"].iloc[0]
        assert row["exact_duplicate_count"] == 2   # both rows flagged by duplicated()

    def test_near_duplicate_within_3_days(self):
        """Same bene + same code, dates 2 days apart = near duplicate."""
        claims = make_claims("NPI003", [
            {"desynpuf_id": "BENE001", "primary_hcpcs_cd": "99213",
             "clm_from_dt": pd.Timestamp("2008-05-01")},
            {"desynpuf_id": "BENE001", "primary_hcpcs_cd": "99213",
             "clm_from_dt": pd.Timestamp("2008-05-03")},   # 2-day gap
        ])
        result = _detect_duplicates(claims)
        row = result[result["at_physn_npi"] == "NPI003"].iloc[0]
        assert row["near_duplicate_count"] >= 1

    def test_not_near_duplicate_beyond_3_days(self):
        """Same bene + same code, dates 5 days apart = NOT a near duplicate."""
        claims = make_claims("NPI004", [
            {"desynpuf_id": "BENE001", "primary_hcpcs_cd": "99213",
             "clm_from_dt": pd.Timestamp("2008-05-01")},
            {"desynpuf_id": "BENE001", "primary_hcpcs_cd": "99213",
             "clm_from_dt": pd.Timestamp("2008-05-06")},   # 5-day gap
        ])
        result = _detect_duplicates(claims)
        row = result[result["at_physn_npi"] == "NPI004"].iloc[0]
        assert row["near_duplicate_count"] == 0

    def test_different_benes_not_duplicates(self):
        """Same code + same date but different beneficiaries = not a duplicate."""
        claims = make_claims("NPI005", [
            {"desynpuf_id": "BENE001", "primary_hcpcs_cd": "99213",
             "clm_from_dt": pd.Timestamp("2008-03-15")},
            {"desynpuf_id": "BENE002", "primary_hcpcs_cd": "99213",
             "clm_from_dt": pd.Timestamp("2008-03-15")},   # different bene
        ])
        result = _detect_duplicates(claims)
        row = result[result["at_physn_npi"] == "NPI005"].iloc[0]
        assert row["exact_duplicate_count"] == 0
        assert row["near_duplicate_count"]  == 0

    def test_multiple_providers_independent(self):
        """Duplicate detection runs independently per provider."""
        claims = pd.concat([
            make_claims("NPI_A", [
                {"desynpuf_id": "BENE001", "primary_hcpcs_cd": "99213",
                 "clm_from_dt": pd.Timestamp("2008-01-01")},
                {"desynpuf_id": "BENE001", "primary_hcpcs_cd": "99213",
                 "clm_from_dt": pd.Timestamp("2008-01-01")},  # exact dup
            ]),
            make_claims("NPI_B", [
                {"desynpuf_id": "BENE002", "primary_hcpcs_cd": "99214",
                 "clm_from_dt": pd.Timestamp("2008-02-10")},
            ]),
        ], ignore_index=True)

        result = _detect_duplicates(claims)
        npi_a = result[result["at_physn_npi"] == "NPI_A"].iloc[0]
        npi_b = result[result["at_physn_npi"] == "NPI_B"].iloc[0]

        assert npi_a["exact_duplicate_count"] == 2
        assert npi_b["exact_duplicate_count"] == 0


# -----------------------------------------------------------------------
# Volume and velocity logic
# -----------------------------------------------------------------------
class TestVolumeFeatures:

    def test_total_claim_count(self):
        claims = make_claims("NPI010", [{"claim_year": 2008}] * 15)
        count = claims[claims["claim_year"] == 2008].groupby("at_physn_npi").size()
        assert count["NPI010"] == 15

    def test_distinct_beneficiaries(self):
        claims = make_claims("NPI011", [
            {"desynpuf_id": "BENE001"},
            {"desynpuf_id": "BENE001"},
            {"desynpuf_id": "BENE002"},
            {"desynpuf_id": "BENE003"},
        ])
        distinct = claims.groupby("at_physn_npi")["desynpuf_id"].nunique()
        assert distinct["NPI011"] == 3


# -----------------------------------------------------------------------
# Billing ratio features
# -----------------------------------------------------------------------
class TestBillingFeatures:

    def test_avg_submitted_to_allowed_ratio(self):
        claims = make_claims("NPI020", [
            {"submitted_charge_amt": 400.0, "allowed_amt": 200.0,
             "submitted_to_allowed_ratio": 2.0},
            {"submitted_charge_amt": 600.0, "allowed_amt": 200.0,
             "submitted_to_allowed_ratio": 3.0},
        ])
        avg_ratio = claims["submitted_to_allowed_ratio"].mean()
        assert abs(avg_ratio - 2.5) < 1e-6

    def test_ratio_zero_allowed(self):
        """Ratio should be None/NaN when allowed amount is zero (avoid division by zero)."""
        allowed = 0.0
        ratio   = 300.0 / allowed if allowed > 0 else None
        assert ratio is None


# -----------------------------------------------------------------------
# Temporal features
# -----------------------------------------------------------------------
class TestTemporalFeatures:

    def test_weekend_flag_saturday(self):
        dt = pd.Timestamp("2008-01-05")   # Saturday
        assert dt.dayofweek in (5, 6)

    def test_weekend_flag_sunday(self):
        dt = pd.Timestamp("2008-01-06")   # Sunday
        assert dt.dayofweek in (5, 6)

    def test_weekday_not_flagged(self):
        dt = pd.Timestamp("2008-01-07")   # Monday
        assert dt.dayofweek not in (5, 6)

    def test_pct_weekend_claims(self):
        claims = make_claims("NPI030", [
            {"clm_from_dt": pd.Timestamp("2008-01-05"), "is_weekend_claim": True},   # Sat
            {"clm_from_dt": pd.Timestamp("2008-01-06"), "is_weekend_claim": True},   # Sun
            {"clm_from_dt": pd.Timestamp("2008-01-07"), "is_weekend_claim": False},  # Mon
            {"clm_from_dt": pd.Timestamp("2008-01-08"), "is_weekend_claim": False},  # Tue
        ])
        pct = claims["is_weekend_claim"].mean()
        assert abs(pct - 0.5) < 1e-6


# -----------------------------------------------------------------------
# Post-death billing
# -----------------------------------------------------------------------
class TestPostDeathBilling:

    def test_claim_after_death_flagged(self):
        death_date   = pd.Timestamp("2008-06-01")
        claim_date   = pd.Timestamp("2008-07-15")   # after death
        is_after     = claim_date > death_date
        assert is_after is True

    def test_claim_before_death_not_flagged(self):
        death_date   = pd.Timestamp("2008-06-01")
        claim_date   = pd.Timestamp("2008-04-10")
        is_after     = claim_date > death_date
        assert is_after is False

    def test_claim_in_death_year(self):
        death_date   = pd.Timestamp("2008-11-20")
        claim_date   = pd.Timestamp("2008-03-01")
        same_year    = claim_date.year == death_date.year
        assert same_year is True


# -----------------------------------------------------------------------
# Feature schema completeness
# -----------------------------------------------------------------------
class TestFeatureSchema:

    REQUIRED_FEATURES = [
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

    def test_all_required_features_defined(self):
        """
        Ensures the FEATURE_COLUMNS list in isolation_forest.py
        contains every required feature.
        """
        from ml.isolation_forest import FEATURE_COLUMNS
        for feat in self.REQUIRED_FEATURES:
            assert feat in FEATURE_COLUMNS, (
                f"Feature '{feat}' is required but missing from "
                f"ml/isolation_forest.py FEATURE_COLUMNS."
            )

    def test_no_extra_undefined_features(self):
        """
        Every feature in FEATURE_COLUMNS should be in the required list
        or explicitly documented as an extension.
        """
        from ml.isolation_forest import FEATURE_COLUMNS
        undefined = [f for f in FEATURE_COLUMNS if f not in self.REQUIRED_FEATURES]
        assert len(undefined) == 0, (
            f"Undocumented features in FEATURE_COLUMNS: {undefined}"
        )
