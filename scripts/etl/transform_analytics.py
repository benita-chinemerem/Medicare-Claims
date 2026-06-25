"""
scripts/etl/transform_analytics.py

Transforms staging tables into the typed analytics schema.

KEY SCHEMA FACTS (actual DE-SynPUF carrier claims):
  - No AT_PHYSN_NPI field. Provider identified by PRF_PHYSN_NPI_1 (line 1 NPI).
  - No aggregate NCH_CARR_CLM_SBMTD_CHRG_AMT. Billing is at line level:
      LINE_ALOWD_CHRG_AMT_1..13  = allowed charge per line
      LINE_NCH_PMT_AMT_1..13     = Medicare payment per line
  - We derive:
      at_physn_npi    = PRF_PHYSN_NPI_1
      allowed_amt     = SUM(LINE_ALOWD_CHRG_AMT_1..13)
      clm_pmt_amt     = SUM(LINE_NCH_PMT_AMT_1..13)
      payment_to_allowed_ratio = clm_pmt_amt / allowed_amt
        (replaces submitted-to-allowed; captures underpayment/overpayment signal)
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text

log = logging.getLogger(__name__)

CARRIER_HCPCS_COLS    = [f"hcpcs_cd_{i}"             for i in range(1, 14)]
CARRIER_NPI_COLS      = [f"prf_physn_npi_{i}"         for i in range(1, 14)]
CARRIER_ALOWD_COLS    = [f"line_alowd_chrg_amt_{i}"   for i in range(1, 14)]
CARRIER_PMT_COLS      = [f"line_nch_pmt_amt_{i}"      for i in range(1, 14)]
OUTPATIENT_HCPCS_COLS = [f"hcpcs_cd_{i}"              for i in range(1, 46)]
OUTPATIENT_REV_COLS   = [f"revenue_cd_{i}"             for i in range(1, 6)]

CHUNK_SIZE = 50_000


def _safe_date(val) -> Optional[str]:
    if not val or str(val).strip() in ("", "nan", "NaN", "0"):
        return None
    val = str(val).strip().split(".")[0]
    if len(val) == 8 and val.isdigit():
        return f"{val[:4]}-{val[4:6]}-{val[6:8]}"
    return None


def _safe_numeric(val) -> Optional[float]:
    try:
        f = float(val)
        return None if (f != f) else f
    except (TypeError, ValueError):
        return None


def _build_array(row: pd.Series, cols: list[str]) -> list[str]:
    codes = []
    for col in cols:
        val = str(row.get(col, "")).strip()
        if val and val.lower() not in ("nan", "none", "0", ""):
            codes.append(val)
    return list(dict.fromkeys(codes))


def _sum_line_amounts(row: pd.Series, cols: list[str]) -> float:
    """Sums numeric values across line-level amount columns, ignoring nulls."""
    total = 0.0
    for col in cols:
        v = _safe_numeric(row.get(col))
        if v is not None:
            total += v
    return total


def transform_beneficiary(engine) -> int:
    log.info("Transforming beneficiary_summary...")
    df = pd.read_sql("SELECT * FROM staging.beneficiary_summary", engine)
    if df.empty:
        log.warning("staging.beneficiary_summary is empty.")
        return 0

    for col in ("bene_birth_dt", "bene_death_dt"):
        df[col] = df[col].apply(_safe_date)

    int_cols = [
        "bene_hi_cvrage_tot_mons", "bene_smi_cvrage_tot_mons",
        "bene_hmo_cvrage_tot_mons", "plan_cvrg_mos_num",
        "sp_alzhdmta", "sp_chf", "sp_chrnkidn", "sp_cncr",
        "sp_copd", "sp_depressn", "sp_diabetes",
        "sp_ischmcht", "sp_osteoprs", "sp_ra_oa", "sp_strketia",
    ]
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int16")

    for col in ("medreimb_ip", "benres_ip", "pppymt_ip",
                "medreimb_op", "benres_op", "pppymt_op",
                "medreimb_car", "benres_car", "pppymt_car"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    out = pd.DataFrame({
        "desynpuf_id":              df["desynpuf_id"],
        "year":                     df["year"].astype("Int16"),
        "sample_id":                df["sample_id"].astype("Int16"),
        "birth_date":               df["bene_birth_dt"],
        "death_date":               df["bene_death_dt"],
        "sex_cd":                   df["bene_sex_ident_cd"].astype(str).str.strip().replace("nan", None),
        "race_cd":                  df["bene_race_cd"].astype(str).str.strip().replace("nan", None),
        "esrd_ind":                 df["bene_esrd_ind"].astype(str).str.strip().replace("nan", None),
        "state_code":               df["sp_state_code"].astype(str).str.strip().replace("nan", None),
        "county_cd":                df["bene_county_cd"].astype(str).str.strip().replace("nan", None),
        "part_a_months":            df["bene_hi_cvrage_tot_mons"],
        "part_b_months":            df["bene_smi_cvrage_tot_mons"],
        "hmo_months":               df["bene_hmo_cvrage_tot_mons"],
        "flag_alzheimer":           df["sp_alzhdmta"],
        "flag_chf":                 df["sp_chf"],
        "flag_chronic_kidney":      df["sp_chrnkidn"],
        "flag_cancer":              df["sp_cncr"],
        "flag_copd":                df["sp_copd"],
        "flag_depression":          df["sp_depressn"],
        "flag_diabetes":            df["sp_diabetes"],
        "flag_ischemic_heart":      df["sp_ischmcht"],
        "flag_osteoporosis":        df["sp_osteoprs"],
        "flag_ra_oa":               df["sp_ra_oa"],
        "flag_stroke":              df["sp_strketia"],
        "reimbursement_inpatient":  df["medreimb_ip"],
        "reimbursement_outpatient": df["medreimb_op"],
        "reimbursement_carrier":    df["medreimb_car"],
    })

    out = out.dropna(subset=["desynpuf_id", "year"])
    rows_written = 0
    for i in range(0, len(out), CHUNK_SIZE):
        out.iloc[i:i+CHUNK_SIZE].to_sql(
            "beneficiary_summary", engine, schema="analytics",
            if_exists="append", index=False, method="multi",
        )
        rows_written += len(out.iloc[i:i+CHUNK_SIZE])

    log.info("beneficiary_summary: %d rows written.", rows_written)
    return rows_written


def transform_carrier(engine) -> int:
    """
    Reads staging.carrier_claims (wide/line-level format) and writes to
    analytics.carrier_claims.

    Provider NPI:    PRF_PHYSN_NPI_1 (first performing physician)
    Allowed amount:  SUM of LINE_ALOWD_CHRG_AMT_1..13
    Payment amount:  SUM of LINE_NCH_PMT_AMT_1..13
    Ratio:           payment / allowed (replaces submitted-to-allowed)
    """
    log.info("Transforming carrier_claims (wide/line-level schema)...")
    df = pd.read_sql("SELECT * FROM staging.carrier_claims", engine)
    if df.empty:
        log.warning("staging.carrier_claims is empty.")
        return 0

    # Primary provider NPI: first non-null PRF_PHYSN_NPI across lines
    npi_cols_present = [c for c in CARRIER_NPI_COLS if c in df.columns]
    df["at_physn_npi"] = df[npi_cols_present].bfill(axis=1).iloc[:, 0]

    # Drop rows with no NPI or no claim ID
    df = df.dropna(subset=["clm_id", "at_physn_npi"])
    df = df[df["at_physn_npi"].str.strip().replace("", float("nan")).notna()]

    # Date parsing
    df["clm_from_dt"] = df["clm_from_dt"].apply(_safe_date)
    df["clm_thru_dt"] = df["clm_thru_dt"].apply(_safe_date)
    df = df.dropna(subset=["clm_from_dt"])

    # Sum line-level amounts
    alowd_cols_present = [c for c in CARRIER_ALOWD_COLS if c in df.columns]
    pmt_cols_present   = [c for c in CARRIER_PMT_COLS   if c in df.columns]

    for col in alowd_cols_present + pmt_cols_present:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["allowed_amt"]  = df[alowd_cols_present].sum(axis=1)
    df["clm_pmt_amt"]  = df[pmt_cols_present].sum(axis=1)

    # Payment-to-allowed ratio (fraud signal: unusual payment rates)
    df["payment_to_allowed_ratio"] = df.apply(
        lambda r: r["clm_pmt_amt"] / r["allowed_amt"]
        if r["allowed_amt"] > 0 else None,
        axis=1,
    )

    # HCPCS code array
    hcpcs_present = [c for c in CARRIER_HCPCS_COLS if c in df.columns]
    df["hcpcs_codes"]      = df.apply(lambda r: _build_array(r, hcpcs_present), axis=1)
    df["primary_hcpcs_cd"] = df["hcpcs_codes"].apply(lambda c: c[0] if c else None)

    out = pd.DataFrame({
        "clm_id":                    df["clm_id"].str.strip(),
        "desynpuf_id":               df["desynpuf_id"].str.strip(),
        "at_physn_npi":              df["at_physn_npi"].str.strip(),
        "clm_from_dt":               df["clm_from_dt"],
        "clm_thru_dt":               df["clm_thru_dt"],
        "clm_pmt_amt":               df["clm_pmt_amt"],
        "allowed_amt":               df["allowed_amt"],
        # payment_to_allowed_ratio maps to submitted_to_allowed_ratio column
        # in analytics schema (column kept for compatibility)
        "submitted_to_allowed_ratio": df["payment_to_allowed_ratio"],
        "submitted_charge_amt":       df["allowed_amt"],   # best proxy available
        "primary_hcpcs_cd":          df["primary_hcpcs_cd"],
        "hcpcs_codes":               df["hcpcs_codes"].apply(
                                         lambda x: "{" + ",".join(x) + "}" if x else "{}"
                                     ),
        "prncpal_dgns_cd":           df.get("icd9_dgns_cd_1", pd.Series([None]*len(df)))
                                        .astype(str).str.strip().replace("nan", None),
        "nch_clm_type_cd":           None,
        "place_of_service_cd":       None,   # not present in DE-SynPUF carrier claims
        "sample_id":                 df["sample_id"].astype("Int16"),
    })

    rows_written = 0
    for i in range(0, len(out), CHUNK_SIZE):
        out.iloc[i:i+CHUNK_SIZE].to_sql(
            "carrier_claims", engine, schema="analytics",
            if_exists="append", index=False, method="multi",
        )
        rows_written += len(out.iloc[i:i+CHUNK_SIZE])

    log.info("carrier_claims: %d rows written to analytics schema.", rows_written)
    return rows_written


def transform_outpatient(engine) -> int:
    log.info("Transforming outpatient_claims...")
    df = pd.read_sql("SELECT * FROM staging.outpatient_claims", engine)
    if df.empty:
        log.warning("staging.outpatient_claims is empty.")
        return 0

    df = df.dropna(subset=["clm_id"])
    df["clm_from_dt"] = df["clm_from_dt"].apply(_safe_date)
    df["clm_thru_dt"] = df["clm_thru_dt"].apply(_safe_date)
    df = df.dropna(subset=["clm_from_dt"])
    df["clm_pmt_amt"] = pd.to_numeric(df["clm_pmt_amt"], errors="coerce")

    hcpcs_present = [c for c in OUTPATIENT_HCPCS_COLS if c in df.columns]
    rev_present   = [c for c in OUTPATIENT_REV_COLS   if c in df.columns]

    df["hcpcs_codes"]  = df.apply(lambda r: _build_array(r, hcpcs_present), axis=1)
    df["revenue_codes"] = df.apply(lambda r: _build_array(r, rev_present), axis=1)

    out = pd.DataFrame({
        "clm_id":          df["clm_id"].str.strip(),
        "desynpuf_id":     df["desynpuf_id"].str.strip(),
        "prvdr_num":       df.get("prvdr_num", pd.Series([None]*len(df))).astype(str).str.strip().replace("nan", None),
        "at_physn_npi":    df.get("at_physn_npi", pd.Series([None]*len(df))).astype(str).str.strip().replace("nan", None),
        "clm_from_dt":     df["clm_from_dt"],
        "clm_thru_dt":     df["clm_thru_dt"],
        "clm_pmt_amt":     df["clm_pmt_amt"],
        "clm_fac_type_cd": df.get("clm_fac_type_cd", pd.Series([None]*len(df))).astype(str).str.strip().replace("nan", None),
        "prncpal_dgns_cd": df.get("icd9_dgns_cd_1", pd.Series([None]*len(df))).astype(str).str.strip().replace("nan", None),
        "hcpcs_codes":     df["hcpcs_codes"].apply(lambda x: "{" + ",".join(x) + "}" if x else "{}"),
        "revenue_codes":   df["revenue_codes"].apply(lambda x: "{" + ",".join(x) + "}" if x else "{}"),
        "sample_id":       df["sample_id"].astype("Int16"),
    })

    rows_written = 0
    for i in range(0, len(out), CHUNK_SIZE):
        out.iloc[i:i+CHUNK_SIZE].to_sql(
            "outpatient_claims", engine, schema="analytics",
            if_exists="append", index=False, method="multi",
        )
        rows_written += len(out.iloc[i:i+CHUNK_SIZE])

    log.info("outpatient_claims: %d rows written.", rows_written)
    return rows_written


def run_all_transforms(db_conn_str: str) -> None:
    """
    Entry point called by DAG 1. Runs all three transforms in order.
    """
    engine = create_engine(db_conn_str)
    total  = 0
    total += transform_beneficiary(engine)
    total += transform_carrier(engine)
    total += transform_outpatient(engine)
    log.info("All transforms complete. Total rows written: %d", total)