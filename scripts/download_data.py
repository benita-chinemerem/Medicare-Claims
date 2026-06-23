"""
scripts/download_data.py

Helper for downloading CMS DE-SynPUF Sample files.

CMS does not provide a direct programmatic download API for DE-SynPUF.
The files are hosted on the CMS website and require manual download via
browser. This script:

    1. Prints the exact URLs and file names to download
    2. Verifies that files are present after manual download
    3. Confirms expected file sizes are plausible

Manual download steps:
    1. Go to https://www.cms.gov/data-research/statistics-trends-and-reports/
       medicare-claims-synthetic-public-use-files/
       cms-2008-2010-data-entrepreneurs-synthetic-public-use-file-de-synpuf
    2. Download the Data Users Guide (codebook PDF) — required for feature engineering
    3. Download Sample 1 and Sample 2 ZIP files for each claim type
    4. Extract ZIPs into data/raw/sample_01/ and data/raw/sample_02/
    5. Run this script to verify the download

Usage:
    python scripts/download_data.py --check           # verify files are present
    python scripts/download_data.py --print-urls      # print CMS download URLs
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

# -----------------------------------------------------------------------
# Expected files per sample after extraction
# -----------------------------------------------------------------------
EXPECTED_FILES = {
    "sample_01": [
        "DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv",
        "DE1_0_2009_Beneficiary_Summary_File_Sample_1.csv",
        "DE1_0_2010_Beneficiary_Summary_File_Sample_1.csv",
        "DE1_0_2008_to_2010_Carrier_Claims_Sample_1A.csv",
        "DE1_0_2008_to_2010_Carrier_Claims_Sample_1B.csv",
        "DE1_0_2008_to_2010_Outpatient_Claims_Sample_1.csv",
        "DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv",
        "DE1_0_2008_to_2010_Prescription_Drug_Events_Sample_1.csv",
    ],
    "sample_02": [
        "DE1_0_2008_Beneficiary_Summary_File_Sample_2.csv",
        "DE1_0_2009_Beneficiary_Summary_File_Sample_2.csv",
        "DE1_0_2010_Beneficiary_Summary_File_Sample_2.csv",
        "DE1_0_2008_to_2010_Carrier_Claims_Sample_2A.csv",
        "DE1_0_2008_to_2010_Carrier_Claims_Sample_2B.csv",
        "DE1_0_2008_to_2010_Outpatient_Claims_Sample_2.csv",
        "DE1_0_2008_to_2010_Inpatient_Claims_Sample_2.csv",
        "DE1_0_2008_to_2010_Prescription_Drug_Events_Sample_2.csv",
    ],
}

# Note: Prescription Drug Events and Inpatient Claims are listed here
# for completeness but are outside the scope of this version of the framework.
OUT_OF_SCOPE_PATTERNS = [
    "Prescription_Drug_Events",
    "Inpatient_Claims",
]

CMS_PAGE_URL = (
    "https://www.cms.gov/data-research/statistics-trends-and-reports/"
    "medicare-claims-synthetic-public-use-files/"
    "cms-2008-2010-data-entrepreneurs-synthetic-public-use-file-de-synpuf"
)

CODEBOOK_URL = (
    "https://www.cms.gov/Research-Statistics-Data-and-Systems/"
    "Downloadable-Public-Use-Files/SynPUFs/Downloads/SynPUF_DUG.pdf"
)

# Approximate minimum file sizes in MB (anything smaller is likely corrupt)
MIN_FILE_SIZES_MB = {
    "Carrier_Claims":         50,
    "Outpatient_Claims":      20,
    "Beneficiary_Summary":     3,
    "Inpatient_Claims":        5,
    "Prescription_Drug_Events": 80,
}


def print_download_instructions() -> None:
    print()
    print("=" * 70)
    print("  CMS DE-SynPUF — Download Instructions")
    print("=" * 70)
    print()
    print("Step 1: Open the CMS DE-SynPUF page in your browser:")
    print(f"        {CMS_PAGE_URL}")
    print()
    print("Step 2: Download the Data Users Guide (codebook) — REQUIRED:")
    print(f"        {CODEBOOK_URL}")
    print()
    print("Step 3: Download these ZIP files from the CMS page:")
    print()
    print("  For Sample 1 (place in data/raw/sample_01/):")
    for f in EXPECTED_FILES["sample_01"]:
        tag = "  [out of scope]" if any(p in f for p in OUT_OF_SCOPE_PATTERNS) else ""
        print(f"    - {f}{tag}")
    print()
    print("  For Sample 2 (place in data/raw/sample_02/):")
    for f in EXPECTED_FILES["sample_02"]:
        tag = "  [out of scope]" if any(p in f for p in OUT_OF_SCOPE_PATTERNS) else ""
        print(f"    - {f}{tag}")
    print()
    print("Step 4: After extraction, run:")
    print("        python scripts/download_data.py --check")
    print()
    print("Step 5: Then run:")
    print("        python scripts/validate_data.py")
    print("        python scripts/load_raw_to_parquet.py")
    print()
    print("Out-of-scope files (Inpatient Claims, Prescription Drug Events)")
    print("are not used by this version of the pipeline but can be downloaded")
    print("for future extension.")
    print()


def check_files(raw_path: str) -> bool:
    """
    Checks that all expected in-scope files are present and exceed minimum size.
    Returns True if all OK, False otherwise.
    """
    all_ok = True
    for sample, files in EXPECTED_FILES.items():
        sample_dir = os.path.join(raw_path, sample)
        log.info("Checking %s...", sample_dir)

        for fname in files:
            if any(p in fname for p in OUT_OF_SCOPE_PATTERNS):
                continue   # skip out-of-scope files

            fpath = os.path.join(sample_dir, fname)
            if not os.path.exists(fpath):
                log.error("  MISSING: %s", fpath)
                all_ok = False
                continue

            size_mb = os.path.getsize(fpath) / 1_048_576
            # Determine minimum size check
            min_mb = next(
                (v for k, v in MIN_FILE_SIZES_MB.items() if k in fname), 1
            )
            if size_mb < min_mb:
                log.error(
                    "  SUSPECT SIZE: %s — %.1f MB (expected >= %d MB). "
                    "File may be incomplete.",
                    fname, size_mb, min_mb,
                )
                all_ok = False
            else:
                log.info("  OK: %s (%.1f MB)", fname, size_mb)

    if all_ok:
        log.info("All expected files present and have plausible sizes.")
        log.info("Next step: run python scripts/validate_data.py")
    else:
        log.error("Some files are missing or undersized. Re-download from CMS.")

    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CMS DE-SynPUF download helper."
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Check that downloaded files are present and have plausible sizes.",
    )
    parser.add_argument(
        "--print-urls", action="store_true",
        help="Print CMS download page URL and instructions.",
    )
    parser.add_argument(
        "--raw-path", default="data/raw/",
        help="Path to the raw data directory (default: data/raw/)",
    )
    args = parser.parse_args()

    if args.print_urls or not args.check:
        print_download_instructions()

    if args.check:
        ok = check_files(args.raw_path)
        sys.exit(0 if ok else 1)
