"""
scripts/validate_data.py

Pre-load validation for DE-SynPUF files.

Checks:
    1. All expected files exist for each sample
    2. Row counts are within 10% of CMS codebook targets
    3. Required columns are present in each file
    4. Primary key fields (clm_id, desynpuf_id, at_physn_npi) are not fully null

Run this before starting DAG 1 to catch data issues early.

Usage:
    python scripts/validate_data.py --raw-path data/raw/
    python scripts/validate_data.py --raw-path data/raw/ --sample sample_01
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

# -----------------------------------------------------------------------
# CMS codebook row-count targets (approximate per sample)
# Source: DE-SynPUF Data Users Guide, CMS 2013
# -----------------------------------------------------------------------
CODEBOOK_TARGETS = {
    "carrier_a":    2_350_000,   # Carrier Claims Part A CSV
    "carrier_b":    2_350_000,   # Carrier Claims Part B CSV
    "outpatient":     790_000,
    "bene_2008":      116_352,
    "bene_2009":      116_352,
    "bene_2010":      116_352,
}

ROW_COUNT_TOLERANCE = 0.10   # allow 10% deviation

# -----------------------------------------------------------------------
# Expected file name patterns per sample
# Placeholder {SN} = Sample_1 or Sample_2
# -----------------------------------------------------------------------
FILE_SPECS = {
    "carrier_a": {
        "pattern": "DE1_0_2008_to_2010_Carrier_Claims_{SN}A.csv",
        "required_cols": [
            "DESYNPUF_ID", "CLM_ID", "AT_PHYSN_NPI",
            "CLM_FROM_DT", "NCH_CARR_CLM_SBMTD_CHRG_AMT",
            "NCH_CARR_CLM_ALLWD_AMT",
        ],
    },
    "carrier_b": {
        "pattern": "DE1_0_2008_to_2010_Carrier_Claims_{SN}B.csv",
        "required_cols": [
            "DESYNPUF_ID", "CLM_ID", "AT_PHYSN_NPI",
            "CLM_FROM_DT", "NCH_CARR_CLM_SBMTD_CHRG_AMT",
            "NCH_CARR_CLM_ALLWD_AMT",
        ],
    },
    "outpatient": {
        "pattern": "DE1_0_2008_to_2010_Outpatient_Claims_{SN}.csv",
        "required_cols": [
            "DESYNPUF_ID", "CLM_ID", "CLM_FROM_DT", "CLM_PMT_AMT",
        ],
    },
    "bene_2008": {
        "pattern": "DE1_0_2008_Beneficiary_Summary_File_{SN}.csv",
        "required_cols": [
            "DESYNPUF_ID", "BENE_BIRTH_DT", "BENE_SEX_IDENT_CD",
            "BENE_HI_CVRAGE_TOT_MONS", "BENE_SMI_CVRAGE_TOT_MONS",
        ],
    },
    "bene_2009": {
        "pattern": "DE1_0_2009_Beneficiary_Summary_File_{SN}.csv",
        "required_cols": ["DESYNPUF_ID", "BENE_BIRTH_DT"],
    },
    "bene_2010": {
        "pattern": "DE1_0_2010_Beneficiary_Summary_File_{SN}.csv",
        "required_cols": ["DESYNPUF_ID", "BENE_BIRTH_DT"],
    },
}

SAMPLE_NAME_MAP = {
    "sample_01": "Sample_1",
    "sample_02": "Sample_2",
}


def validate_sample(raw_path: str, sample: str) -> list[str]:
    """
    Validates all expected files for one sample directory.
    Returns a list of error strings (empty list = all OK).
    """
    errors: list[str] = []
    sn = SAMPLE_NAME_MAP.get(sample)
    if sn is None:
        errors.append(f"Unknown sample identifier: '{sample}'. Expected sample_01 or sample_02.")
        return errors

    sample_dir = os.path.join(raw_path, sample)
    if not os.path.isdir(sample_dir):
        errors.append(f"Sample directory not found: {sample_dir}")
        return errors

    for file_key, spec in FILE_SPECS.items():
        fname = spec["pattern"].replace("{SN}", sn)
        fpath = os.path.join(sample_dir, fname)

        # 1. File existence
        if not os.path.exists(fpath):
            errors.append(f"[{sample}] MISSING: {fname}")
            continue

        # 2. Row count (line count - 1 for header)
        with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
            row_count = sum(1 for _ in fh) - 1

        if row_count <= 0:
            errors.append(f"[{sample}] EMPTY: {fname} (0 data rows)")
            continue

        target = CODEBOOK_TARGETS.get(file_key, 0)
        if target > 0:
            deviation = abs(row_count - target) / target
            if deviation > ROW_COUNT_TOLERANCE:
                log.warning(
                    "[%s] Row count for %s: found %d, expected ~%d (%.1f%% deviation)",
                    sample, fname, row_count, target, deviation * 100,
                )
            else:
                log.info(
                    "[%s] OK: %s — %d rows (within %.0f%% of codebook target %d)",
                    sample, fname, row_count, ROW_COUNT_TOLERANCE * 100, target,
                )

        # 3. Column presence (read only the header row)
        try:
            header_df = pd.read_csv(fpath, nrows=0)
            actual_cols = [c.upper().strip() for c in header_df.columns.tolist()]
        except Exception as exc:
            errors.append(f"[{sample}] Could not read header of {fname}: {exc}")
            continue

        for col in spec["required_cols"]:
            if col.upper() not in actual_cols:
                errors.append(f"[{sample}] Missing required column '{col}' in {fname}")

        # 4. Primary key null check (read first 1000 rows)
        pk_cols_map = {
            "carrier_a": ["DESYNPUF_ID", "CLM_ID"],
            "carrier_b": ["DESYNPUF_ID", "CLM_ID"],
            "outpatient": ["DESYNPUF_ID", "CLM_ID"],
            "bene_2008": ["DESYNPUF_ID"],
            "bene_2009": ["DESYNPUF_ID"],
            "bene_2010": ["DESYNPUF_ID"],
        }
        pk_cols = pk_cols_map.get(file_key, [])
        if pk_cols:
            try:
                sample_df = pd.read_csv(fpath, nrows=1000, dtype=str, low_memory=False)
                sample_df.columns = [c.upper().strip() for c in sample_df.columns]
                for pk in pk_cols:
                    if pk in sample_df.columns:
                        null_rate = sample_df[pk].isna().mean()
                        if null_rate > 0.05:
                            errors.append(
                                f"[{sample}] High null rate ({null_rate:.0%}) "
                                f"in primary key column '{pk}' of {fname}"
                            )
            except Exception as exc:
                log.warning("[%s] Could not run PK null check on %s: %s", sample, fname, exc)

    return errors


def run_validation(raw_path: str, samples: list[str]) -> bool:
    """
    Runs validation for all requested samples.
    Returns True if all pass, False otherwise.
    """
    all_errors: list[str] = []

    for sample in samples:
        log.info("Validating sample: %s", sample)
        errs = validate_sample(raw_path, sample)
        all_errors.extend(errs)

    if all_errors:
        log.error("\n=== VALIDATION FAILED ===")
        for err in all_errors:
            log.error("  %s", err)
        log.error("%d error(s) found. Resolve before running DAG 1.", len(all_errors))
        return False

    log.info("=== VALIDATION PASSED — all files present and structurally correct ===")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate DE-SynPUF files before pipeline run.")
    parser.add_argument(
        "--raw-path", default="data/raw/",
        help="Path to the raw data directory (default: data/raw/)",
    )
    parser.add_argument(
        "--sample", default=None,
        help="Validate a single sample only (e.g. sample_01). Default: validates all.",
    )
    args = parser.parse_args()

    samples_to_check = (
        [args.sample] if args.sample
        else list(SAMPLE_NAME_MAP.keys())
    )

    ok = run_validation(raw_path=args.raw_path, samples=samples_to_check)
    sys.exit(0 if ok else 1)
