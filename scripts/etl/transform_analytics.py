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
"""

from __future__ import annotations

import csv
import logging
import os
from io import StringIO
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

# Chunk sizes optimized for table widths to balance network and memory speed
BENEFICIARY_CHUNK_SIZE = 50_000
CLAIMS_CHUNK_SIZE      = 20_000


def psql_insert_copy(table, conn, keys, data_iter):
    """
    Executes a high-performance bulk insertion using the PostgreSQL COPY engine.
    Dramatically reduces database CPU, RAM, and WAL overhead compared to standard INSERTs.
    """
    dbapi_conn = conn.connection
    with dbapi_conn.cursor() as cur:
        s_buf = StringIO()
        writer = csv.writer(s_buf)
        writer.writerows(data_iter)
        s_buf.seek(0)

        columns = ', '.join([f'"{k}"' for k in keys])
        table_name = f'"{table.schema}"."{table.name}"' if table.schema else f'"{table.name}"'
        
        sql = f'COPY {table_name} ({columns}) FROM STDIN WITH CSV'
        cur.copy_expert(sql=sql, file=s_buf)


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


def transform_beneficiary(engine) -> int:
    log.info("Transforming beneficiary_summary in memory-safe chunks...")
    rows_written = 0
    
    # Create an intermediate unlogged staging table to safely hold bulk COPY chunks without constraint friction
    with engine.begin() as init_conn:
        init_conn.execute(text("DROP TABLE IF EXISTS analytics.stage_beneficiary_summary;"))
        init_conn.execute(text(
            "CREATE UNLOGGED TABLE analytics.stage_beneficiary_summary "
            "(LIKE analytics.beneficiary_summary EXCLUDING CONSTRAINTS);"
        ))

    with engine.connect() as conn:
        streaming_conn = conn.execution_options(stream_results=True)
        try:
            chunks = pd.read_sql("SELECT * FROM staging.beneficiary_summary", streaming_conn, chunksize=BENEFICIARY_CHUNK_SIZE)
        except Exception as e:
            log.error("Failed to read staging.beneficiary_summary: %s", e)
            return 0

        for df in chunks:
            if df.empty:
                continue

            df.columns = df.columns.str.lower()

            for col in ("bene_birth_dt", "bene_death_dt"):
                if col in df.columns:
                    df[col] = df[col].apply(_safe_date)

            int_cols = [
                "bene_hi_cvrage_tot_mons", "bene_smi_cvrage_tot_mons",
                "bene_hmo_cvrage_tot_mons", "plan_cvrg_mos_num",
                "sp_alzhdmta", "sp_chf", "sp_chrnkidn", "sp_cncr",
                "sp_copd", "sp_depressn", "sp_diabetes",
                "sp_ischmcht", "sp_osteoprs", "sp_ra_oa", "sp_strketia",
            ]
            for col in int_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int16")

            float_cols = [
                "medreimb_ip", "benres_ip", "pppymt_ip",
                "medreimb_op", "benres_op", "pppymt_op",
                "medreimb_car", "benres_car", "pppymt_car"
            ]
            for col in float_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            out = pd.DataFrame({
                "desynpuf_id":              df["desynpuf_id"].astype(str).str.strip().replace("nan", None) if "desynpuf_id" in df.columns else None,
                "year":                     df["year"].astype("Int16") if "year" in df.columns else None,
                "sample_id":                df["sample_id"].astype("Int16") if "sample_id" in df.columns else None,
                "birth_date":               df.get("bene_birth_dt"),
                "death_date":               df.get("bene_death_dt"),
                "sex_cd":                   df["bene_sex_ident_cd"].astype(str).str.strip().replace("nan", None) if "bene_sex_ident_cd" in df.columns else None,
                "race_cd":                  df["bene_race_cd"].astype(str).str.strip().replace("nan", None) if "bene_race_cd" in df.columns else None,
                "esrd_ind":                 df["bene_esrd_ind"].astype(str).str.strip().replace("nan", None) if "bene_esrd_ind" in df.columns else None,
                "state_code":               df["sp_state_code"].astype(str).str.strip().replace("nan", None) if "sp_state_code" in df.columns else None,
                "county_cd":                df["bene_county_cd"].astype(str).str.strip().replace("nan", None) if "bene_county_cd" in df.columns else None,
                "part_a_months":            df.get("bene_hi_cvrage_tot_mons"),
                "part_b_months":            df.get("bene_smi_cvrage_tot_mons"),
                "hmo_months":               df.get("bene_hmo_cvrage_tot_mons"),
                "flag_alzheimer":           df.get("sp_alzhdmta"),
                "flag_chf":                 df.get("sp_chf"),
                "flag_chronic_kidney":      df.get("sp_chrnkidn"),
                "flag_cancer":              df.get("sp_cncr"),
                "flag_copd":                df.get("sp_copd"),
                "flag_depression":          df.get("sp_depressn"),
                "flag_diabetes":            df.get("sp_diabetes"),
                "flag_ischemic_heart":      df.get("sp_ischmcht"),
                "flag_osteoporosis":        df.get("sp_osteoprs"),
                "flag_ra_oa":               df.get("sp_ra_oa"),
                "flag_stroke":              df.get("sp_strketia"),
                "reimbursement_inpatient":  df.get("medreimb_ip"),
                "reimbursement_outpatient": df.get("medreimb_op"),
                "reimbursement_carrier":    df.get("medreimb_car"),
            })

            out = out.dropna(subset=["desynpuf_id", "year"])
            
            # Deduplicate cleanly within the pandas dataframe chunk
            out = out.drop_duplicates(subset=["desynpuf_id", "year"], keep="first")
            
            if out.empty:
                continue

            # Write directly to intermediate staging table
            out.to_sql(
                "stage_beneficiary_summary", engine, schema="analytics",
                if_exists="append", index=False, method=psql_insert_copy
            )

    # FIX: Explicitly target columns and omit the 'chronic_condition_count' generated field
    log.info("Performing final cross-chunk deduplication and merging to production...")
    merge_sql = """
        INSERT INTO analytics.beneficiary_summary (
            desynpuf_id, year, sample_id, birth_date, death_date, sex_cd, race_cd, esrd_ind, 
            state_code, county_cd, part_a_months, part_b_months, hmo_months, 
            flag_alzheimer, flag_chf, flag_chronic_kidney, flag_cancer, flag_copd, 
            flag_depression, flag_diabetes, flag_ischemic_heart, flag_osteoporosis, 
            flag_ra_oa, flag_stroke, reimbursement_inpatient, reimbursement_outpatient, 
            reimbursement_carrier
        )
        SELECT DISTINCT ON (desynpuf_id, year) 
            desynpuf_id, year, sample_id, birth_date, death_date, sex_cd, race_cd, esrd_ind, 
            state_code, county_cd, part_a_months, part_b_months, hmo_months, 
            flag_alzheimer, flag_chf, flag_chronic_kidney, flag_cancer, flag_copd, 
            flag_depression, flag_diabetes, flag_ischemic_heart, flag_osteoporosis, 
            flag_ra_oa, flag_stroke, reimbursement_inpatient, reimbursement_outpatient, 
            reimbursement_carrier
        FROM analytics.stage_beneficiary_summary
        ORDER BY desynpuf_id, year
        ON CONFLICT (desynpuf_id, year) DO NOTHING;
    """
    with engine.begin() as merge_conn:
        result = merge_conn.execute(text(merge_sql))
        rows_written = result.rowcount if result.rowcount is not None else 0
        merge_conn.execute(text("DROP TABLE IF EXISTS analytics.stage_beneficiary_summary;"))

    log.info("beneficiary_summary: %d rows written.", rows_written)
    return rows_written


def transform_carrier(engine) -> int:
    log.info("Transforming carrier_claims in memory-safe chunks...")
    rows_written = 0

    with engine.connect() as conn:
        streaming_conn = conn.execution_options(stream_results=True)
        try:
            chunks = pd.read_sql("SELECT * FROM staging.carrier_claims", streaming_conn, chunksize=CLAIMS_CHUNK_SIZE)
        except Exception as e:
            log.error("Failed to read staging.carrier_claims: %s", e)
            return 0

        for df in chunks:
            if df.empty:
                continue

            df.columns = df.columns.str.lower()

            npi_cols_present = [c for c in CARRIER_NPI_COLS if c in df.columns]
            if not npi_cols_present or "clm_id" not in df.columns:
                continue

            df["at_physn_npi"] = df[npi_cols_present].bfill(axis=1).iloc[:, 0]
            df = df.dropna(subset=["clm_id", "at_physn_npi"])
            df = df[df["at_physn_npi"].str.strip().replace("", float("nan")).notna()]
            if df.empty:
                continue

            df["clm_from_dt"] = df["clm_from_dt"].apply(_safe_date)
            df["clm_thru_dt"] = df["clm_thru_dt"].apply(_safe_date)
            df = df.dropna(subset=["clm_from_dt"])
            if df.empty:
                continue

            alowd_cols_present = [c for c in CARRIER_ALOWD_COLS if c in df.columns]
            pmt_cols_present   = [c for c in CARRIER_PMT_COLS   if c in df.columns]

            for col in alowd_cols_present + pmt_cols_present:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

            df["allowed_amt"]  = df[alowd_cols_present].sum(axis=1)
            df["clm_pmt_amt"]  = df[pmt_cols_present].sum(axis=1)

            df["payment_to_allowed_ratio"] = df.apply(
                lambda r: r["clm_pmt_amt"] / r["allowed_amt"]
                if r["allowed_amt"] > 0 else None,
                axis=1,
            )

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
                "submitted_to_allowed_ratio": df["payment_to_allowed_ratio"],
                "submitted_charge_amt":       df["allowed_amt"],
                "primary_hcpcs_cd":          df["primary_hcpcs_cd"],
                "hcpcs_codes":               df["hcpcs_codes"].apply(
                                                 lambda x: "{" + ",".join(x) + "}" if x else "{}"
                                             ),
                "prncpal_dgns_cd":           df.get("icd9_dgns_cd_1", pd.Series([None]*len(df)))
                                                .astype(str).str.strip().replace("nan", None),
                "nch_clm_type_cd":           None,
                "place_of_service_cd":       None,
                "sample_id":                 df["sample_id"].astype("Int16") if "sample_id" in df.columns else None,
            })

            out = out.drop_duplicates(subset=["clm_id"], keep="first")

            out.to_sql(
                "carrier_claims", engine, schema="analytics",
                if_exists="append", index=False, method=psql_insert_copy
            )
            rows_written += len(out)

    log.info("carrier_claims: %d rows written to analytics schema.", rows_written)
    return rows_written


def transform_outpatient(engine) -> int:
    log.info("Transforming outpatient_claims in memory-safe chunks...")
    rows_written = 0

    with engine.connect() as conn:
        streaming_conn = conn.execution_options(stream_results=True)
        try:
            chunks = pd.read_sql("SELECT * FROM staging.outpatient_claims", streaming_conn, chunksize=CLAIMS_CHUNK_SIZE)
        except Exception as e:
            log.error("Failed to read staging.outpatient_claims: %s", e)
            return 0

        for df in chunks:
            if df.empty:
                continue

            df.columns = df.columns.str.lower()

            if "clm_id" not in df.columns:
                continue

            df = df.dropna(subset=["clm_id"])
            df["clm_from_dt"] = df["clm_from_dt"].apply(_safe_date)
            df["clm_thru_dt"] = df["clm_thru_dt"].apply(_safe_date)
            df = df.dropna(subset=["clm_from_dt"])
            if df.empty:
                continue

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
                "sample_id":       df["sample_id"].astype("Int16") if "sample_id" in df.columns else None,
            })

            out = out.drop_duplicates(subset=["clm_id"], keep="first")

            out.to_sql(
                "outpatient_claims", engine, schema="analytics",
                if_exists="append", index=False, method=psql_insert_copy
            )
            rows_written += len(out)

    log.info("outpatient_claims: %d rows written.", rows_written)
    return rows_written


def run_all_transforms(db_conn_str: str) -> None:
    """
    Entry point called by DAG 1. Runs all three transforms in order.
    """
    engine = create_engine(db_conn_str)
    
    log.info("Ensuring 'analytics' schema exists in PostgreSQL...")
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS analytics;"))
        
    log.info("Truncating analytics tables safely if they exist...")
    conditional_truncate_script = """
    DO $$ 
    BEGIN 
        IF EXISTS (SELECT FROM pg_tables WHERE schemaname = 'analytics' AND tablename = 'beneficiary_summary') THEN
            TRUNCATE TABLE analytics.beneficiary_summary RESTART IDENTITY CASCADE;
        END IF;
        IF EXISTS (SELECT FROM pg_tables WHERE schemaname = 'analytics' AND tablename = 'carrier_claims') THEN
            TRUNCATE TABLE analytics.carrier_claims RESTART IDENTITY CASCADE;
        END IF;
        IF EXISTS (SELECT FROM pg_tables WHERE schemaname = 'analytics' AND tablename = 'outpatient_claims') THEN
            TRUNCATE TABLE analytics.outpatient_claims RESTART IDENTITY CASCADE;
        END IF;
    END $$;
    """
    with engine.begin() as conn:
        conn.execute(text(conditional_truncate_script))

    total  = 0
    total += transform_beneficiary(engine)
    total += transform_carrier(engine)
    total += transform_outpatient(engine)
    log.info("All transforms complete. Total rows written: %d", total)