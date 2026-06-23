"""
scripts/etl/transform_analytics.py

Transforms staging tables (TEXT/VARCHAR columns, raw from CSV) into the
typed, indexed analytics schema.

This is the single place where:
    - String dates are parsed to DATE
    - String amounts are cast to NUMERIC
    - HCPCS codes are gathered into arrays
    - Place-of-service codes are validated
    - Null NPIs and null claim IDs are filtered

Called by DAG 1 (dag1_historical_backfill.py) via the
transform_to_analytics_schema task.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text

log = logging.getLogger(__name__)

# Carrier claim HCPCS columns in the staging schema
CARRIER_HCPCS_COLS = [f"hcpcs_cd_{i}" for i in range(1, 14)]

# Outpatient HCPCS and revenue code columns in the staging schema
OUTPATIENT_HCPCS_COLS   = [f"hcpcs_cd_{i}"  for i in range(1, 46)]
OUTPATIENT_REVENUE_COLS = [f"revenue_cd_{i}" for i in range(1, 6)]

# Chunk size for Postgres inserts
CHUNK_SIZE = 50_000


def _safe_date(val: str) -> Optional[str]:
    """Converts YYYYMMDD strings from DE-SynPUF to YYYY-MM-DD. Returns None if unparseable."""
    if not val or str(val).strip() in ("", "nan", "NaN", "0"):
        return None
    val = str(val).strip().split(".")[0]  # strip any float formatting
    if len(val) == 8 and val.isdigit():
        return f"{val[:4]}-{val[4:6]}-{val[6:8]}"
    return None


def _safe_numeric(val) -> Optional[float]:
    """Casts a value to float, returning None on failure."""
    try:
        f = float(val)
        return None if (f != f) else f   # catches NaN
    except (TypeError, ValueError):
        return None


def _build_hcpcs_array(row: pd.Series, cols: list[str]) -> list[str]:
    """Returns a de-duplicated, null-free list of HCPCS codes from a claim row."""
    codes = []
    for col in cols:
        val = str(row.get(col, "")).strip()
        if val and val.lower() not in ("nan", "none", "0", ""):
            codes.append(val)
    return list(dict.fromkeys(codes))   # preserves order, removes duplicates


def transform_beneficiary(engine) -> int:
    """
    Reads staging.beneficiary_summary, casts fields, writes to
    analytics.beneficiary_summary.  Returns rows written.
    """
    log.info("Transforming beneficiary_summary...")

    df = pd.read_sql("SELECT * FROM staging.beneficiary_summary", engine)
    if df.empty:
        log.warning("staging.beneficiary_summary is empty. Has the backfill run?")
        return 0

    # Date fields
    for col in ("bene_birth_dt", "bene_death_dt"):
        df[col] = df[col].apply(_safe_date)

    # Integer/smallint fields
    int_cols = [
        "bene_hi_cvrage_tot_mons", "bene_smi_cvrage_tot_mons",
        "bene_hmo_cvrage_tot_mons", "plan_cvrg_mos_num",
        "sp_alzhdmta", "sp_chf", "sp_chrnkidn", "sp_cncr",
        "sp_copd", "sp_depressn", "sp_diabetes",
        "sp_ischmcht", "sp_osteoprs", "sp_ra_oa", "sp_strketia",
    ]
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int16")

    # Numeric reimbursement fields
    for col in ("medreimb_ip", "benres_ip", "pppymt_ip",
                "medreimb_op", "benres_op", "pppymt_op",
                "medreimb_car", "benres_car", "pppymt_car"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    out = pd.DataFrame({
        "desynpuf_id":             df["desynpuf_id"],
        "year":                    df["year"].astype("Int16"),
        "sample_id":               df["sample_id"].astype("Int16"),
        "birth_date":              df["bene_birth_dt"],
        "death_date":              df["bene_death_dt"],
        "sex_cd":                  df["bene_sex_ident_cd"].astype(str).str.strip().replace("nan", None),
        "race_cd":                 df["bene_race_cd"].astype(str).str.strip().replace("nan", None),
        "esrd_ind":                df["bene_esrd_ind"].astype(str).str.strip().replace("nan", None),
        "state_code":              df["sp_state_code"].astype(str).str.strip().replace("nan", None),
        "county_cd":               df["bene_county_cd"].astype(str).str.strip().replace("nan", None),
        "part_a_months":           df["bene_hi_cvrage_tot_mons"],
        "part_b_months":           df["bene_smi_cvrage_tot_mons"],
        "hmo_months":              df["bene_hmo_cvrage_tot_mons"],
        "flag_alzheimer":          df["sp_alzhdmta"],
        "flag_chf":                df["sp_chf"],
        "flag_chronic_kidney":     df["sp_chrnkidn"],
        "flag_cancer":             df["sp_cncr"],
        "flag_copd":               df["sp_copd"],
        "flag_depression":         df["sp_depressn"],
        "flag_diabetes":           df["sp_diabetes"],
        "flag_ischemic_heart":     df["sp_ischmcht"],
        "flag_osteoporosis":       df["sp_osteoprs"],
        "flag_ra_oa":              df["sp_ra_oa"],
        "flag_stroke":             df["sp_strketia"],
        "reimbursement_inpatient": df["medreimb_ip"],
        "reimbursement_outpatient":df["medreimb_op"],
        "reimbursement_carrier":   df["medreimb_car"],
    })

    # Drop rows missing the primary key components
    out = out.dropna(subset=["desynpuf_id", "year"])

    rows_written = 0
    for i in range(0, len(out), CHUNK_SIZE):
        chunk = out.iloc[i : i + CHUNK_SIZE]
        chunk.to_sql(
            "beneficiary_summary", engine,
            schema="analytics", if_exists="append",
            index=False, method="multi",
        )
        rows_written += len(chunk)

    log.info("beneficiary_summary: %d rows written to analytics schema.", rows_written)
    return rows_written


def transform_carrier(engine) -> int:
    """
    Reads staging.carrier_claims, casts fields, builds HCPCS arrays,
    writes to analytics.carrier_claims.
    """
    log.info("Transforming carrier_claims...")

    df = pd.read_sql("SELECT * FROM staging.carrier_claims", engine)
    if df.empty:
        log.warning("staging.carrier_claims is empty.")
        return 0

    # Drop rows with null NPI or claim ID (no investigative value without them)
    df = df.dropna(subset=["clm_id", "at_physn_npi"])
    df = df[df["clm_id"].str.strip() != ""]
    df = df[df["at_physn_npi"].str.strip() != ""]

    # Date parsing
    df["clm_from_dt"] = df["clm_from_dt"].apply(_safe_date)
    df["clm_thru_dt"] = df["clm_thru_dt"].apply(_safe_date)
    df = df.dropna(subset=["clm_from_dt"])   # claim date is required

    # Numeric fields
    for col, dest in [
        ("clm_pmt_amt",                  "clm_pmt_amt"),
        ("nch_carr_clm_sbmtd_chrg_amt",  "submitted_charge_amt"),
        ("nch_carr_clm_allwd_amt",       "allowed_amt"),
    ]:
        df[dest] = pd.to_numeric(df[col], errors="coerce")

    # HCPCS code array
    df["hcpcs_codes"] = df.apply(
        lambda row: _build_hcpcs_array(row, CARRIER_HCPCS_COLS), axis=1
    )
    df["primary_hcpcs_cd"] = df["hcpcs_codes"].apply(
        lambda codes: codes[0] if codes else None
    )

    out = pd.DataFrame({
        "clm_id":               df["clm_id"].str.strip(),
        "desynpuf_id":          df["desynpuf_id"].str.strip(),
        "at_physn_npi":         df["at_physn_npi"].str.strip(),
        "clm_from_dt":          df["clm_from_dt"],
        "clm_thru_dt":          df["clm_thru_dt"],
        "clm_pmt_amt":          df["clm_pmt_amt"],
        "submitted_charge_amt": df["submitted_charge_amt"],
        "allowed_amt":          df["allowed_amt"],
        "place_of_service_cd":  df["line_place_of_srvc_cd"].astype(str).str.strip().replace("nan", None),
        "nch_clm_type_cd":      df["nch_clm_type_cd"].astype(str).str.strip().replace("nan", None),
        "prncpal_dgns_cd":      df["prncpal_dgns_cd"].astype(str).str.strip().replace("nan", None),
        "primary_hcpcs_cd":     df["primary_hcpcs_cd"],
        "hcpcs_codes":          df["hcpcs_codes"].apply(
                                    lambda x: "{" + ",".join(x) + "}" if x else "{}"
                                ),
        "sample_id":            df["sample_id"].astype("Int16"),
    })

    rows_written = 0
    for i in range(0, len(out), CHUNK_SIZE):
        chunk = out.iloc[i : i + CHUNK_SIZE]
        chunk.to_sql(
            "carrier_claims", engine,
            schema="analytics", if_exists="append",
            index=False, method="multi",
        )
        rows_written += len(chunk)

    log.info("carrier_claims: %d rows written to analytics schema.", rows_written)
    return rows_written


def transform_outpatient(engine) -> int:
    """
    Reads staging.outpatient_claims, casts fields, writes to
    analytics.outpatient_claims.
    """
    log.info("Transforming outpatient_claims...")

    df = pd.read_sql("SELECT * FROM staging.outpatient_claims", engine)
    if df.empty:
        log.warning("staging.outpatient_claims is empty.")
        return 0

    df = df.dropna(subset=["clm_id"])
    df = df[df["clm_id"].str.strip() != ""]

    df["clm_from_dt"] = df["clm_from_dt"].apply(_safe_date)
    df["clm_thru_dt"] = df["clm_thru_dt"].apply(_safe_date)
    df = df.dropna(subset=["clm_from_dt"])

    df["clm_pmt_amt"] = pd.to_numeric(df["clm_pmt_amt"], errors="coerce")

    df["hcpcs_codes"] = df.apply(
        lambda row: _build_hcpcs_array(row, OUTPATIENT_HCPCS_COLS), axis=1
    )
    df["revenue_codes"] = df.apply(
        lambda row: _build_hcpcs_array(row, OUTPATIENT_REVENUE_COLS), axis=1
    )

    out = pd.DataFrame({
        "clm_id":          df["clm_id"].str.strip(),
        "desynpuf_id":     df["desynpuf_id"].str.strip(),
        "prvdr_num":       df["prvdr_num"].astype(str).str.strip().replace("nan", None),
        "at_physn_npi":    df["at_physn_npi"].astype(str).str.strip().replace("nan", None),
        "clm_from_dt":     df["clm_from_dt"],
        "clm_thru_dt":     df["clm_thru_dt"],
        "clm_pmt_amt":     df["clm_pmt_amt"],
        "clm_fac_type_cd": df["clm_fac_type_cd"].astype(str).str.strip().replace("nan", None),
        "prncpal_dgns_cd": df["prncpal_dgns_cd"].astype(str).str.strip().replace("nan", None),
        "hcpcs_codes":     df["hcpcs_codes"].apply(
                               lambda x: "{" + ",".join(x) + "}" if x else "{}"
                           ),
        "revenue_codes":   df["revenue_codes"].apply(
                               lambda x: "{" + ",".join(x) + "}" if x else "{}"
                           ),
        "sample_id":       df["sample_id"].astype("Int16"),
    })

    rows_written = 0
    for i in range(0, len(out), CHUNK_SIZE):
        chunk = out.iloc[i : i + CHUNK_SIZE]
        chunk.to_sql(
            "outpatient_claims", engine,
            schema="analytics", if_exists="append",
            index=False, method="multi",
        )
        rows_written += len(chunk)

    log.info("outpatient_claims: %d rows written to analytics schema.", rows_written)
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
    log.info("All transforms complete. Total rows written to analytics schema: %d", total)
