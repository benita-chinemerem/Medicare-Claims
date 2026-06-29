"""
dag1_historical_backfill.py

One-time historical backfill DAG.
Validates DE-SynPUF files, converts CSVs to Parquet, loads staging tables,
and transforms to the typed analytics schema.

Run order:
    validate_files >> convert_to_parquet >> load_staging_beneficiary
    >> load_staging_carrier >> load_staging_outpatient
    >> transform_analytics_beneficiary >> transform_analytics_carrier
    >> transform_analytics_outpatient >> backfill_complete
"""

from __future__ import annotations

import os
import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default args
# ---------------------------------------------------------------------------
DEFAULT_ARGS = {
    "owner": "fraud-anomaly-ai",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
RAW_PATH      = os.environ.get("DATA_RAW_PATH",     "/opt/airflow/data/raw")
PARQUET_PATH  = os.environ.get("DATA_PARQUET_PATH", "/opt/airflow/data/parquet")
DB_CONN_STR   = (
    f"postgresql+psycopg2://{os.environ.get('FRAUD_DB_USER', 'airflow')}:"
    f"{os.environ.get('FRAUD_DB_PASSWORD', 'airflow')}@"
    f"{os.environ.get('FRAUD_DB_HOST', 'postgres')}:"
    f"{os.environ.get('FRAUD_DB_PORT', '5432')}/"
    f"{os.environ.get('FRAUD_DB_NAME', 'fraud_claims')}"
)

SAMPLES = ["sample_01", "sample_02"]

# CMS codebook row-count targets (approximate; used for validation)
CODEBOOK_ROW_COUNTS = {
    "carrier":     4_700_000,
    "outpatient":    790_000,
    "beneficiary":   115_000,  # per year
}
VALIDATION_TOLERANCE = 0.10  # allow 10% variance from codebook targets


# ---------------------------------------------------------------------------
# Task functions
# ---------------------------------------------------------------------------

def validate_files(**context):
    """
    Check that expected DE-SynPUF files exist and that row counts are within
    10% of the CMS codebook targets. Raises if any file is missing or empty.
    """
    import pandas as pd

    errors = []

    file_patterns = {
        "carrier": [
            "DE1_0_2008_to_2010_Carrier_Claims_{sample_upper}A.csv",
            "DE1_0_2008_to_2010_Carrier_Claims_{sample_upper}B.csv",
        ],
        "outpatient": [
            "DE1_0_2008_to_2010_Outpatient_Claims_{sample_upper}.csv",
        ],
        "beneficiary": [
            "DE1_0_2008_Beneficiary_Summary_File_{sample_upper}.csv",
            "DE1_0_2009_Beneficiary_Summary_File_{sample_upper}.csv",
            "DE1_0_2010_Beneficiary_Summary_File_{sample_upper}.csv",
        ],
    }

    for sample in SAMPLES:
        sample_upper = sample.replace("sample_0", "Sample_")
        sample_dir = os.path.join(RAW_PATH, sample)

        for file_type, patterns in file_patterns.items():
            for pattern in patterns:
                fname = pattern.format(sample_upper=sample_upper)
                fpath = os.path.join(sample_dir, fname)

                if not os.path.exists(fpath):
                    errors.append(f"MISSING: {fpath}")
                    continue

                # Row count check (count lines rather than loading full file)
                with open(fpath, "r") as f:
                    row_count = sum(1 for _ in f) - 1  # subtract header

                if row_count == 0:
                    errors.append(f"EMPTY: {fpath}")
                    continue

                target = CODEBOOK_ROW_COUNTS.get(file_type, 0)
                if target > 0:
                    deviation = abs(row_count - target) / target
                    if deviation > VALIDATION_TOLERANCE:
                        log.warning(
                            "Row count deviation for %s: found %d, expected ~%d (%.1f%% off)",
                            fname, row_count, target, deviation * 100
                        )

                log.info("OK: %s (%d rows)", fname, row_count)

    if errors:
        raise ValueError(
            f"File validation failed with {len(errors)} error(s):\n" +
            "\n".join(errors)
        )

    log.info("All DE-SynPUF files validated successfully.")


def convert_to_parquet(**context):
    """
    Convert all raw CSVs to Parquet format and write to the Parquet zone.
    Parquet copies dramatically speed up subsequent feature engineering reads.
    """
    import pandas as pd

    os.makedirs(PARQUET_PATH, exist_ok=True)

    all_files = []
    for sample in SAMPLES:
        sample_dir = os.path.join(RAW_PATH, sample)
        for fname in os.listdir(sample_dir):
            if fname.endswith(".csv"):
                all_files.append((sample, os.path.join(sample_dir, fname)))

    for sample, csv_path in all_files:
        fname_stem = os.path.splitext(os.path.basename(csv_path))[0]
        out_dir = os.path.join(PARQUET_PATH, sample)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{fname_stem}.parquet")

        if os.path.exists(out_path):
            log.info("Parquet already exists, skipping: %s", out_path)
            continue

        log.info("Converting %s -> %s", csv_path, out_path)
        df = pd.read_csv(csv_path, dtype=str, low_memory=False)
        df.to_parquet(out_path, index=False, engine="pyarrow")
        log.info("Written %d rows to %s", len(df), out_path)


def load_staging_table(file_type: str, **context):
    """
    Generic staging loader. Streams data from Parquet using PyArrow batches,
    appends sample_id, and bulk-inserts to the appropriate staging table
    using database-safe chunk sizes to prevent Postgres container crashes.
    """
    import pandas as pd
    import pyarrow.parquet as pq
    from sqlalchemy import create_engine, text

    engine = create_engine(DB_CONN_STR)
    
    # Ensure the staging schema space AND the load_log table exist
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS staging;"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS staging.load_log (
                id SERIAL PRIMARY KEY,
                table_name TEXT,
                sample_id INTEGER,
                file_name TEXT,
                rows_loaded INTEGER,
                loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))

    table_map = {
        "beneficiary": "staging.beneficiary_summary",
        "carrier":     "staging.carrier_claims",
        "outpatient":  "staging.outpatient_claims",
    }
    target_table = table_map[file_type]

    for sample_idx, sample in enumerate(SAMPLES, start=1):
        sample_parquet_dir = os.path.join(PARQUET_PATH, sample)

        matching_files = [
            f for f in os.listdir(sample_parquet_dir)
            if file_type.replace("beneficiary", "Beneficiary")
                        .replace("carrier",    "Carrier")
                        .replace("outpatient", "Outpatient").lower() in f.lower()
            and f.endswith(".parquet")
        ]

        for pq_file in sorted(matching_files):
            pq_path = os.path.join(sample_parquet_dir, pq_file)
            log.info("Streaming %s -> %s", pq_path, target_table)

            pfile = pq.ParquetFile(pq_path)
            rows_loaded = 0
            
            # SAFEGUARD: Drop chunk size to 20k to avoid Postgres query parameter limits
            chunk_size = 20_000
            for batch in pfile.iter_batches(batch_size=chunk_size):
                chunk_df = batch.to_pandas()
                chunk_df["sample_id"] = sample_idx

                if file_type == "beneficiary":
                    year = int([p for p in pq_file.split("_") if p.isdigit() and len(p) == 4][0])
                    chunk_df["year"] = year

                # SAFEGUARD: Removed method="multi" to protect DB memory execution
                chunk_df.to_sql(
                    target_table.split(".")[1],
                    engine,
                    schema=target_table.split(".")[0],
                    if_exists="append",
                    index=False,
                )
                rows_loaded += len(chunk_df)

            # Log execution data to load_log
            with engine.begin() as conn:
                conn.execute(
                    text("""
                        INSERT INTO staging.load_log
                            (table_name, sample_id, file_name, rows_loaded)
                        VALUES (:tbl, :sid, :fname, :rows)
                    """),
                    {"tbl": target_table, "sid": sample_idx,
                     "fname": pq_file, "rows": rows_loaded},
                )

            log.info("Successfully loaded %d total rows from %s", rows_loaded, pq_file)

def transform_analytics(**context):
    """
    Calls the transform script which reads from staging and writes to
    the typed analytics schema. Import done inside function so Airflow
    worker does not need the module at DAG parse time.
    """
    import sys
    sys.path.insert(0, "/opt/airflow/scripts")
    from etl.transform_analytics import run_all_transforms
    run_all_transforms(db_conn_str=DB_CONN_STR)


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id="dag1_historical_backfill",
    default_args=DEFAULT_ARGS,
    description=(
        "One-time historical backfill: validate DE-SynPUF files, convert to Parquet, "
        "load staging tables, transform to analytics schema."
    ),
    schedule_interval=None,       # triggered manually; does not recur
    start_date=days_ago(1),
    catchup=False,
    tags=["fraud-anomaly-ai", "backfill", "etl"],
    doc_md=__doc__,
) as dag:

    start = EmptyOperator(task_id="start")

    t_validate = PythonOperator(
        task_id="validate_files",
        python_callable=validate_files,
        doc_md="Validates that all expected DE-SynPUF CSVs exist and have plausible row counts.",
    )

    t_parquet = PythonOperator(
        task_id="convert_to_parquet",
        python_callable=convert_to_parquet,
        doc_md="Converts raw CSVs to Parquet for faster downstream reads (raw zone dual storage).",
    )

    t_stage_bene = PythonOperator(
        task_id="load_staging_beneficiary",
        python_callable=load_staging_table,
        op_kwargs={"file_type": "beneficiary"},
    )

    t_stage_carrier = PythonOperator(
        task_id="load_staging_carrier",
        python_callable=load_staging_table,
        op_kwargs={"file_type": "carrier"},
    )

    t_stage_op = PythonOperator(
        task_id="load_staging_outpatient",
        python_callable=load_staging_table,
        op_kwargs={"file_type": "outpatient"},
    )

    t_transform = PythonOperator(
        task_id="transform_to_analytics_schema",
        python_callable=transform_analytics,
        doc_md=(
            "Transforms all staging tables to the typed analytics schema. "
            "Applies type casting, date parsing, HCPCS array construction, "
            "partitioning, and index creation."
        ),
        execution_timeout=timedelta(hours=3),
    )

    end = EmptyOperator(task_id="backfill_complete")

    # Dependency chain
    (
        start
        >> t_validate
        >> t_parquet
        >> [t_stage_bene, t_stage_carrier, t_stage_op]
        >> t_transform
        >> end
    )
