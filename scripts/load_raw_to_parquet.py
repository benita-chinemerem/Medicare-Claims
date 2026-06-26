"""
scripts/load_raw_to_parquet.py

Converts all DE-SynPUF CSV files in the raw zone to Parquet format.

Why both formats?
    - CSV originals are the canonical source, exactly as downloaded from CMS.
      They serve as the permanent audit record.
    - Parquet copies reduce feature-engineering and model-retraining read time
      by 5-10x compared to CSV parsing (columnar format, built-in compression,
      faster null handling for wide files with 50+ columns).

This implements the two-tier raw-zone pattern used in production data
engineering environments. Skips files that already have a Parquet copy.

Usage:
    python scripts/load_raw_to_parquet.py
    python scripts/load_raw_to_parquet.py --raw-path data/raw/ --parquet-path data/parquet/
    python scripts/load_raw_to_parquet.py --force   # re-convert even if Parquet exists
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

DEFAULT_RAW_PATH     = os.environ.get("DATA_RAW_PATH",     "data/raw")
DEFAULT_PARQUET_PATH = os.environ.get("DATA_PARQUET_PATH", "data/parquet")


def convert_file(
    csv_path: str,
    parquet_path: str,
    force: bool = False,
    chunksize: int = 100_000,
) -> dict:
    """
    Converts a single CSV file to Parquet using memory-efficient streaming chunks.
    Returns a summary dict with timing and row count.
    """
    if os.path.exists(parquet_path) and not force:
        log.info("Skipping (Parquet exists): %s", parquet_path)
        return {"status": "skipped", "path": parquet_path}

    log.info("Converting: %s", csv_path)
    t0 = time.time()

    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)

    total_rows = 0
    writer = None

    try:
        # Read the file in chunks instead of loading it all into memory
        chunks = pd.read_csv(csv_path, dtype=str, low_memory=False, chunksize=chunksize)
        
        for chunk in chunks:
            total_rows += len(chunk)
            
            # Convert Pandas DataFrame chunk to PyArrow Table
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            
            # Initialize the ParquetWriter with the first chunk's schema
            if writer is None:
                writer = pq.ParquetWriter(
                    parquet_path, 
                    table.schema, 
                    compression="snappy"
                )
            
            # Write this chunk to the file
            writer.write_table(table)
            
    except Exception as e:
        log.error("Failed to convert %s: %s", csv_path, str(e))
        # Clean up partial file if it fails mid-stream
        if os.path.exists(parquet_path):
            os.remove(parquet_path)
        raise e
    finally:
        # Always close the writer to finalize the Parquet file
        if writer is not None:
            writer.close()

    elapsed = time.time() - t0
    size_mb  = os.path.getsize(parquet_path) / 1_048_576

    log.info(
        "Done: %s — %d rows | %.1f MB | %.1fs",
        os.path.basename(parquet_path),
        total_rows,
        size_mb,
        elapsed,
    )
    return {
        "status":   "converted",
        "path":     parquet_path,
        "rows":     total_rows,
        "size_mb":  round(size_mb, 1),
        "elapsed_s": round(elapsed, 1),
    }


def convert_all(
    raw_path: str = DEFAULT_RAW_PATH,
    parquet_path: str = DEFAULT_PARQUET_PATH,
    force: bool = False,
) -> list[dict]:
    """
    Walks the raw directory tree and converts every CSV to Parquet,
    preserving the sample sub-directory structure.
    """
    if not os.path.isdir(raw_path):
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_path}\n"
            "Download DE-SynPUF Samples 1 and 2 from CMS and place them in data/raw/."
        )

    results = []
    for root, _dirs, files in os.walk(raw_path):
        for fname in sorted(files):
            if not fname.lower().endswith(".csv"):
                continue

            csv_full  = os.path.join(root, fname)
            # Mirror the directory structure under parquet_path
            rel_path  = os.path.relpath(csv_full, raw_path)
            pq_full   = os.path.join(
                parquet_path,
                os.path.splitext(rel_path)[0] + ".parquet",
            )

            result = convert_file(csv_full, pq_full, force=force)
            results.append(result)

    converted = sum(1 for r in results if r["status"] == "converted")
    skipped   = sum(1 for r in results if r["status"] == "skipped")

    log.info(
        "Conversion complete. %d converted | %d skipped (already existed).",
        converted, skipped,
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert DE-SynPUF CSVs to Parquet (raw-zone dual storage)."
    )
    parser.add_argument(
        "--raw-path", default=DEFAULT_RAW_PATH,
        help=f"Source CSV directory (default: {DEFAULT_RAW_PATH})",
    )
    parser.add_argument(
        "--parquet-path", default=DEFAULT_PARQUET_PATH,
        help=f"Parquet output directory (default: {DEFAULT_PARQUET_PATH})",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-convert even if Parquet file already exists",
    )
    args = parser.parse_args()
    convert_all(
        raw_path=args.raw_path,
        parquet_path=args.parquet_path,
        force=args.force,
    )
