"""
scripts/etl/convert_to_parquet.py

Converts DE-SynPUF raw CSV files to Parquet format, implementing the
two-tier raw zone pattern described in Section 4.2 of the project spec.

The raw zone holds:
  Tier 1 — original CSV files in /opt/airflow/data/raw/
  Tier 2 — Parquet copies in  /opt/airflow/data/parquet/

Why Parquet alongside CSVs?
  - Parquet is columnar and compressed — reading a subset of columns
    for model development is 5-10x faster than scanning a full CSV.
  - Re-reads during feature engineering iteration do not require
    re-parsing the original CSVs.
  - Parquet files carry schema metadata (column types, nullability),
    which makes downstream reads predictable and type-safe.

This script is idempotent: it checks whether the Parquet file already
exists before converting, so re-running it never overwrites completed files.

Usage (run inside the Airflow container or locally):
    python /opt/airflow/scripts/etl/convert_to_parquet.py

Or from the host (one-time, to cover the existing raw files from DAG 1):
    docker exec fraud_anomaly_airflow_scheduler \
        python /opt/airflow/scripts/etl/convert_to_parquet.py
"""

from __future__ import annotations

import logging
import os

import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

RAW_DIR     = os.environ.get("DATA_RAW_PATH",     "/opt/airflow/data/raw")
PARQUET_DIR = os.environ.get("DATA_PARQUET_PATH", "/opt/airflow/data/parquet")


def convert_csv_to_parquet(
    raw_dir:     str = RAW_DIR,
    parquet_dir: str = PARQUET_DIR,
) -> int:
    """
    Convert every CSV in raw_dir to a Parquet file in parquet_dir.
    Skips files whose Parquet counterpart already exists.

    Returns:
        Number of files actually converted (0 if all already exist).
    """
    os.makedirs(parquet_dir, exist_ok=True)

    csv_files = [f for f in os.listdir(raw_dir) if f.lower().endswith(".csv")]
    if not csv_files:
        log.warning("No CSV files found in %s", raw_dir)
        return 0

    converted = 0
    for fname in sorted(csv_files):
        parquet_name = os.path.splitext(fname)[0] + ".parquet"
        csv_path     = os.path.join(raw_dir, fname)
        parquet_path = os.path.join(parquet_dir, parquet_name)

        if os.path.exists(parquet_path):
            log.info("SKIP  %s  (Parquet already exists)", fname)
            continue

        log.info("Converting %s ...", fname)
        try:
            # Read as string first to avoid pandas making wrong type inferences
            # on CMS fields like NPI (leading zeros) or date strings.
            df = pd.read_csv(csv_path, dtype=str, low_memory=False)
            df.to_parquet(parquet_path, index=False, compression="snappy")
            size_mb = os.path.getsize(parquet_path) / (1024 * 1024)
            log.info(
                "  %-60s → %-60s  (%.1f MB)",
                fname, parquet_name, size_mb,
            )
            converted += 1
        except Exception as exc:
            log.error("Failed to convert %s: %s", fname, exc)

    log.info(
        "Parquet conversion complete. %d file(s) converted. "
        "Output directory: %s",
        converted, parquet_dir,
    )
    return converted


# ── Airflow task wrapper ──────────────────────────────────────────────────────
def convert_raw_to_parquet_task(**context) -> int:
    """
    Airflow PythonOperator callable for DAG 1.
    Returns the count of files converted for XCom.
    """
    return convert_csv_to_parquet()


if __name__ == "__main__":
    convert_csv_to_parquet()
