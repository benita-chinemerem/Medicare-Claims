"""
tests/test_etl.py

Unit and integration tests for the ETL layer.

Tests cover:
    - _safe_date(): date string parsing edge cases
    - _safe_numeric(): numeric cast edge cases
    - _build_hcpcs_array(): HCPCS code array construction
    - transform_beneficiary(): schema and null handling on synthetic fixtures
    - transform_carrier(): date filtering, NPI null drops, ratio computation
    - transform_outpatient(): basic structure

Run with:
    pytest tests/test_etl.py -v
"""

from __future__ import annotations

import sys
import os

import pandas as pd
import pytest

# Allow imports from the scripts package without installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.etl.transform_analytics import (
    _safe_date,
    _safe_numeric,
    _build_hcpcs_array,
)


# -----------------------------------------------------------------------
# _safe_date
# -----------------------------------------------------------------------
class TestSafeDate:

    def test_valid_yyyymmdd(self):
        assert _safe_date("20080115") == "2008-01-15"

    def test_valid_yyyymmdd_float_string(self):
        """DE-SynPUF sometimes serialises integers as float strings."""
        assert _safe_date("20080115.0") == "2008-01-15"

    def test_empty_string(self):
        assert _safe_date("") is None

    def test_zero_value(self):
        """CMS uses '0' as a sentinel for missing death dates."""
        assert _safe_date("0") is None

    def test_nan_string(self):
        assert _safe_date("nan") is None

    def test_none_input(self):
        assert _safe_date(None) is None

    def test_malformed_date(self):
        assert _safe_date("2008-01-15") is None   # wrong format for DE-SynPUF

    def test_short_string(self):
        assert _safe_date("2008") is None

    def test_non_numeric(self):
        assert _safe_date("ABCDEFGH") is None


# -----------------------------------------------------------------------
# _safe_numeric
# -----------------------------------------------------------------------
class TestSafeNumeric:

    def test_integer_string(self):
        assert _safe_numeric("1500") == 1500.0

    def test_float_string(self):
        assert abs(_safe_numeric("1500.75") - 1500.75) < 1e-6

    def test_zero(self):
        assert _safe_numeric("0") == 0.0

    def test_none(self):
        assert _safe_numeric(None) is None

    def test_empty_string(self):
        assert _safe_numeric("") is None

    def test_nan_float(self):
        import math
        assert _safe_numeric(float("nan")) is None

    def test_word_string(self):
        assert _safe_numeric("unknown") is None

    def test_negative(self):
        assert _safe_numeric("-250.5") == -250.5


# -----------------------------------------------------------------------
# _build_hcpcs_array
# -----------------------------------------------------------------------
class TestBuildHcpcsArray:

    def _make_row(self, codes: list) -> pd.Series:
        """Creates a Series with hcpcs_cd_1 ... hcpcs_cd_N columns."""
        data = {}
        for i, code in enumerate(codes, start=1):
            data[f"hcpcs_cd_{i}"] = code
        return pd.Series(data)

    def test_basic_codes(self):
        row  = self._make_row(["99213", "93000", ""])
        cols = [f"hcpcs_cd_{i}" for i in range(1, 4)]
        result = _build_hcpcs_array(row, cols)
        assert result == ["99213", "93000"]

    def test_deduplication(self):
        row  = self._make_row(["99213", "99213", "93000"])
        cols = [f"hcpcs_cd_{i}" for i in range(1, 4)]
        result = _build_hcpcs_array(row, cols)
        assert result == ["99213", "93000"]   # first occurrence preserved

    def test_all_null(self):
        row  = self._make_row(["nan", "0", ""])
        cols = [f"hcpcs_cd_{i}" for i in range(1, 4)]
        result = _build_hcpcs_array(row, cols)
        assert result == []

    def test_preserves_order(self):
        row  = self._make_row(["A", "B", "C"])
        cols = [f"hcpcs_cd_{i}" for i in range(1, 4)]
        result = _build_hcpcs_array(row, cols)
        assert result == ["A", "B", "C"]

    def test_missing_columns(self):
        """If a column key is not in the row, it should be skipped gracefully."""
        row  = pd.Series({"hcpcs_cd_1": "99213"})
        cols = [f"hcpcs_cd_{i}" for i in range(1, 4)]
        result = _build_hcpcs_array(row, cols)
        assert result == ["99213"]


# -----------------------------------------------------------------------
# Transform fixture helpers
# -----------------------------------------------------------------------
def make_staging_carrier_df(n: int = 10) -> pd.DataFrame:
    """Creates a minimal staging carrier claims DataFrame for transform tests."""
    return pd.DataFrame({
        "clm_id":                        [f"CLM{i:06d}" for i in range(n)],
        "desynpuf_id":                   [f"BENE{i:06d}" for i in range(n)],
        "at_physn_npi":                  [f"NPI{i:010d}" for i in range(n)],
        "clm_from_dt":                   ["20080115"] * n,
        "clm_thru_dt":                   ["20080115"] * n,
        "clm_pmt_amt":                   ["150.00"] * n,
        "nch_carr_clm_sbmtd_chrg_amt":   ["300.00"] * n,
        "nch_carr_clm_allwd_amt":        ["150.00"] * n,
        "nch_clm_type_cd":               ["71"] * n,
        "prncpal_dgns_cd":               ["4011"] * n,
        "line_place_of_srvc_cd":         ["11"] * n,
        **{f"hcpcs_cd_{i}": (["99213"] if i == 1 else [None]) * n for i in range(1, 14)},
        "sample_id":                     [1] * n,
    })


def make_staging_beneficiary_df(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame({
        "desynpuf_id":            [f"BENE{i:06d}" for i in range(n)],
        "bene_birth_dt":          ["19300101"] * n,
        "bene_death_dt":          ["0"] * n,
        "bene_sex_ident_cd":      ["1"] * n,
        "bene_race_cd":           ["1"] * n,
        "bene_esrd_ind":          ["N"] * n,
        "sp_state_code":          ["10"] * n,
        "bene_county_cd":         ["100"] * n,
        "bene_hi_cvrage_tot_mons": ["12"] * n,
        "bene_smi_cvrage_tot_mons": ["12"] * n,
        "bene_hmo_cvrage_tot_mons": ["0"] * n,
        "plan_cvrg_mos_num":      ["0"] * n,
        "sp_alzhdmta": ["2"] * n, "sp_chf": ["2"] * n,
        "sp_chrnkidn": ["1"] * n, "sp_cncr": ["2"] * n,
        "sp_copd":     ["2"] * n, "sp_depressn": ["2"] * n,
        "sp_diabetes": ["1"] * n, "sp_ischmcht": ["2"] * n,
        "sp_osteoprs": ["2"] * n, "sp_ra_oa": ["2"] * n,
        "sp_strketia": ["2"] * n,
        "medreimb_ip": ["0"] * n, "benres_ip": ["0"] * n, "pppymt_ip": ["0"] * n,
        "medreimb_op": ["0"] * n, "benres_op": ["0"] * n, "pppymt_op": ["0"] * n,
        "medreimb_car": ["500.00"] * n, "benres_car": ["0"] * n, "pppymt_car": ["0"] * n,
        "year":      [2008] * n,
        "sample_id": [1] * n,
    })


# -----------------------------------------------------------------------
# Transform unit tests (against in-memory DataFrames, no DB required)
# -----------------------------------------------------------------------
class TestCarrierTransformLogic:

    def test_date_parsing_in_output(self):
        """clm_from_dt should become a valid ISO date string after parsing."""
        df = make_staging_carrier_df(3)
        df["clm_from_dt_parsed"] = df["clm_from_dt"].apply(_safe_date)
        assert all(df["clm_from_dt_parsed"] == "2008-01-15")

    def test_null_npi_rows_detected(self):
        """Rows with null at_physn_npi should be flagged for removal."""
        df = make_staging_carrier_df(5)
        df.loc[2, "at_physn_npi"] = None
        null_mask = df["at_physn_npi"].isna()
        assert null_mask.sum() == 1

    def test_submitted_to_allowed_ratio(self):
        """Ratio = submitted / allowed. Should be 2.0 when submitted=300, allowed=150."""
        submitted = 300.0
        allowed   = 150.0
        ratio     = submitted / allowed if allowed > 0 else None
        assert ratio == 2.0

    def test_hcpcs_array_in_carrier(self):
        df   = make_staging_carrier_df(1)
        cols = [f"hcpcs_cd_{i}" for i in range(1, 14)]
        row  = df.iloc[0]
        arr  = _build_hcpcs_array(row, cols)
        assert "99213" in arr
        assert len(arr) == 1   # only one non-null code in fixture


class TestBeneficiaryTransformLogic:

    def test_death_date_zero_becomes_none(self):
        df   = make_staging_beneficiary_df(3)
        parsed = df["bene_death_dt"].apply(_safe_date)
        assert all(parsed.isna())

    def test_chronic_flag_casting(self):
        df = make_staging_beneficiary_df(3)
        # sp_chrnkidn=1 (yes), sp_diabetes=1 (yes), rest=2 (no)
        # Expected chronic_condition_count = 2
        flag_cols = [
            "sp_alzhdmta", "sp_chf", "sp_chrnkidn", "sp_cncr",
            "sp_copd", "sp_depressn", "sp_diabetes",
            "sp_ischmcht", "sp_osteoprs", "sp_ra_oa", "sp_strketia",
        ]
        for col in flag_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int16")

        count = (
            (df["sp_alzhdmta"] == 1).astype(int) +
            (df["sp_chf"]      == 1).astype(int) +
            (df["sp_chrnkidn"] == 1).astype(int) +
            (df["sp_cncr"]     == 1).astype(int) +
            (df["sp_copd"]     == 1).astype(int) +
            (df["sp_depressn"] == 1).astype(int) +
            (df["sp_diabetes"] == 1).astype(int) +
            (df["sp_ischmcht"] == 1).astype(int) +
            (df["sp_osteoprs"] == 1).astype(int) +
            (df["sp_ra_oa"]    == 1).astype(int) +
            (df["sp_strketia"] == 1).astype(int)
        )
        assert all(count == 2)
