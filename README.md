# Medicare Claims Reimbursement-Integrity Framework

> An AI-driven anomaly-detection framework built on CMS DE-SynPUF synthetic Medicare claims data, demonstrating automated data pipelines, provider-level risk scoring, and explainable AI for healthcare payment integrity.

---

## Overview

This project is a fully functional prototype of a Medicare claims reimbursement-integrity system. It ingests over 11 million synthetic claim records from two samples of the publicly available CMS 2008-2010 Data Entrepreneurs' Synthetic Public Use File (DE-SynPUF), runs them through a structured ETL pipeline orchestrated by Apache Airflow, engineers provider-level behavioral features, scores providers using Isolation Forest unsupervised anomaly detection, and surfaces risk scores with SHAP-based reason codes in a Power BI dashboard.

The system was designed explicitly around the purpose CMS stated for the DE-SynPUF dataset: to allow developers to build software "that may eventually be applied to actual CMS claims data." Every component in this repository is production-architecture-aligned. The ETL structure, DAG design, feature engineering logic, and ML layer translate directly to real payer or CMS-contractor environments with minimal modification.

**This is a demonstration framework on synthetic data. No real Medicare beneficiary data, restricted-access claims, or identifiable provider information is used anywhere in this project.**

---

## Table of Contents

- [Architecture](#architecture)
- [Data Source](#data-source)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Setup and Installation](#setup-and-installation)
- [Running the Pipeline](#running-the-pipeline)
- [Dashboard Setup](#dashboard-setup)
- [ML Models](#ml-models)
- [Project Scope](#project-scope)
- [References](#references)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                             │
│         CMS DE-SynPUF CSVs (Samples 1 & 2)                     │
│         ~11M records: Carrier + Outpatient + Beneficiary        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RAW ZONE (local disk)                        │
│         CSV originals  +  Parquet copies                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│             APACHE AIRFLOW (Docker Compose)                     │
│                                                                 │
│  DAG 1: Historical Backfill  (one-time)                         │
│  DAG 2: Weekly Incremental Scoring  (recurring, scheduled)      │
│  DAG 3: Monthly Model Retraining  (recurring, scheduled)        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              POSTGRESQL (Docker)                                │
│                                                                 │
│  staging schema   → raw ingested tables                         │
│  analytics schema → cleaned, typed, indexed claim tables        │
│  features schema  → provider-level feature vectors              │
│  scores schema    → risk scores + SHAP values + model registry  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              PYTHON ML LAYER                                    │
│                                                                 │
│  Isolation Forest    → unsupervised provider anomaly scoring    │
│  XGBoost             → supervised demo on injected scenarios    │
│  SHAP                → per-prediction feature attribution        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              POWER BI DASHBOARD                                 │
│                                                                 │
│  Page 1: Overview            (volume, flags, at-risk dollars)   │
│  Page 2: Provider Risk Rank  (sortable scored provider table)   │
│  Page 3: Claim Explanation   (SHAP reason codes per claim)      │
│  Page 4: Pipeline Health     (run history, model version)       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Source

**Dataset:** CMS 2008-2010 Data Entrepreneurs' Synthetic Public Use File (DE-SynPUF)

**Download:** [CMS DE-SynPUF Page](https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files/cms-2008-2010-data-entrepreneurs-synthetic-public-use-file-de-synpuf) — free, no registration required.

**Also download:** The DE-SynPUF Data Users Guide (codebook) from the same page. It is required reading before running feature engineering and is cited throughout the white paper.

**Samples used:** Sample 1 and Sample 2. Each sample is a self-contained 0.25% slice of Medicare. All claims for a beneficiary stay in the same sample, linkable via `DESYNPUF_ID`.

| File | Volume (per sample) | Role |
|---|---|---|
| Carrier Claims (2 CSVs) | ~4.7M claim lines | Primary fraud-detection target |
| Outpatient Claims (1 CSV) | ~790K claims | Secondary target |
| Beneficiary Summary (3 CSVs, one per year) | ~115K beneficiaries/year | Join table — demographics, chronic conditions |
| Inpatient Claims (1 CSV) | ~67K claims | Optional module |
| Prescription Drug Events (1 CSV) | ~5.5M events | Out of scope — designated as future extension |

---

## Repository Structure

```
fraud-anomaly-ai/
├── README.md
├── docker-compose.yml          # Airflow + PostgreSQL services
├── .env.example                # Environment variable template
├── .gitignore
│
├── dags/
│   ├── dag1_historical_backfill.py     # One-time full data load
│   ├── dag2_weekly_scoring.py          # Recurring incremental scoring
│   └── dag3_monthly_retraining.py      # Recurring model retraining
│
├── sql/
│   ├── 01_create_staging_tables.sql
│   ├── 02_create_analytics_tables.sql
│   ├── 03_create_feature_tables.sql
│   └── 04_create_scores_tables.sql
│
├── scripts/
│   ├── download_data.py        # Download DE-SynPUF files
│   ├── validate_data.py        # Row count + null checks vs codebook
│   ├── load_raw_to_parquet.py  # CSV → Parquet conversion
│   └── etl/
│       ├── load_staging.py         # Staging table load
│       ├── transform_analytics.py  # Analytics schema transforms
│       └── feature_engineering.py  # Provider-level feature computation
│
├── ml/
│   ├── isolation_forest.py     # Unsupervised anomaly scoring
│   ├── xgboost_supervised.py   # Supervised demo on injected anomalies
│   ├── shap_explainer.py       # SHAP value computation + reason codes
│   ├── inject_anomalies.py     # Controlled anomaly injection for supervised layer
│   └── model_registry.py       # Model versioning + promotion logic
│
├── dashboard/
│   └── README.md               # Power BI connection + page setup instructions
│
├── tests/
│   ├── test_etl.py
│   └── test_features.py
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
└── docs/
    ├── whitepaper.md           # Technical white paper
    └── images/
        └── architecture_diagram.png
```

---

## Prerequisites

- **Docker Desktop** (v4.0+) — for Airflow and PostgreSQL
- **Docker Compose** (v2.0+) — included with Docker Desktop
- **Python 3.10+** — for local script execution and ML layer
- **Power BI Desktop** — for dashboard (free download from Microsoft)
- **~15 GB free disk space** — for two DE-SynPUF samples + Parquet copies + PostgreSQL data volume

---

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/Princeleo400/fraud-anomaly-ai.git
cd fraud-anomaly-ai
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and set your values. The defaults work for a local development run.

### 3. Download DE-SynPUF data

Create a data directory and download Samples 1 and 2 from the CMS page:

```bash
mkdir -p data/raw/sample_01 data/raw/sample_02
```

Place the downloaded ZIP files in the appropriate sample directories and extract them. The expected folder structure is:

```
data/
└── raw/
    ├── sample_01/
    │   ├── DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv
    │   ├── DE1_0_2008_to_2010_Carrier_Claims_Sample_1A.csv
    │   ├── DE1_0_2008_to_2010_Carrier_Claims_Sample_1B.csv
    │   ├── DE1_0_2008_to_2010_Outpatient_Claims_Sample_1.csv
    │   ├── DE1_0_2009_Beneficiary_Summary_File_Sample_1.csv
    │   └── DE1_0_2010_Beneficiary_Summary_File_Sample_1.csv
    └── sample_02/
        └── [same structure, Sample_2 suffix]
```

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 5. Start Docker services

```bash
docker-compose up -d
```

This starts PostgreSQL and Apache Airflow (webserver + scheduler + worker). Wait approximately 60 seconds for Airflow to initialize on first run.

### 6. Access Airflow UI

Open [http://localhost:8080](http://localhost:8080) in your browser.

- **Username:** `airflow`
- **Password:** `airflow`

### 7. Initialize the database schema

```bash
docker exec -i fraud_anomaly_postgres psql -U airflow -d fraud_claims < sql/01_create_staging_tables.sql
docker exec -i fraud_anomaly_postgres psql -U airflow -d fraud_claims < sql/02_create_analytics_tables.sql
docker exec -i fraud_anomaly_postgres psql -U airflow -d fraud_claims < sql/03_create_feature_tables.sql
docker exec -i fraud_anomaly_postgres psql -U airflow -d fraud_claims < sql/04_create_scores_tables.sql
```

---

## Running the Pipeline

### DAG 1: Historical Backfill (run once)

In the Airflow UI, find `dag1_historical_backfill` and trigger it manually. This DAG:

1. Validates CSV file presence and row counts against codebook targets
2. Converts CSVs to Parquet (raw zone dual storage)
3. Loads all files to PostgreSQL staging tables
4. Transforms staging tables to the analytics schema (typed, indexed, partitioned)

Expected runtime: 30-60 minutes depending on hardware.

### DAG 2: Weekly Incremental Scoring (recurring)

`dag2_weekly_scoring` is scheduled to run every Sunday at 02:00 UTC. On each run it:

1. Ingests the next batch of held-back claim records (simulating a new weekly claims feed)
2. Refreshes provider-level feature aggregations
3. Scores providers using the trained Isolation Forest model
4. Writes risk scores and SHAP values to the `scores` schema
5. Updates the dashboard data

To simulate multiple weeks of history quickly during initial setup, you can trigger the DAG manually several times with the `batch_id` parameter incremented.

### DAG 3: Monthly Model Retraining (recurring)

`dag3_monthly_retraining` runs on the first day of each month. It retrains the Isolation Forest on accumulated feature data, compares the new model against the prior version, logs metrics, and promotes the new model if performance improves. All model versions are tracked in the `scores.model_registry` table.

---

## Dashboard Setup

See [`dashboard/README.md`](dashboard/README.md) for step-by-step instructions on connecting Power BI Desktop to PostgreSQL and configuring the four dashboard pages.

The PostgreSQL connection string for Power BI:

```
Server:   localhost
Port:     5432
Database: fraud_claims
Username: airflow
Password: (from your .env file)
```

---

## ML Models

### Isolation Forest (Primary)

The unsupervised anomaly-detection model. Runs over provider-level feature vectors (15-20 features) and outputs a continuous risk score per provider. Scores are normalized to a 0-100 scale and ranked by decile in the dashboard.

```bash
python ml/isolation_forest.py --sample sample_01
```

### XGBoost (Supervised Demonstration)

Trained on a copy of the data with controlled anomaly injection. Demonstrates the supervised component of the architecture for environments with labeled fraud data.

```bash
python ml/inject_anomalies.py --output data/injected/
python ml/xgboost_supervised.py --data data/injected/
```

### SHAP Explainer

Generates SHAP values for all scored providers. Output is written to the `scores.shap_values` table and consumed by the dashboard's Claim Explanation page.

```bash
python ml/shap_explainer.py --model isolation_forest --run-id latest
```

---

## Project Scope

**In scope:**
- Carrier claims (physician/supplier Part B billing)
- Outpatient facility claims
- Beneficiary summary (demographics, chronic conditions, coverage months)
- Inpatient claims (optional module, not enabled by default)

**Out of scope:**
- Prescription Drug Events / Part D analytics (designated as future extension)
- Cloud deployment (no BigQuery, no AWS, no Azure — local infrastructure only)
- Real or restricted-access Medicare claims data of any kind
- Any claim that this system detects actual fraud — it is a demonstration framework on synthetic data, and all findings reflect synthetic data patterns only

---

## References

- CMS DE-SynPUF page: https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files/
- DE-SynPUF Data Users Guide (codebook): https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUFs/Downloads/SynPUF_DUG.pdf
- Liu, F.T., Ting, K.M., Zhou, Z-H. "Isolation Forest." IEEE ICDM 2008.
- Lundberg, S.M. and Lee, S-I. "A Unified Approach to Interpreting Model Predictions." NeurIPS 2017.
- Chen, T. and Guestrin, C. "XGBoost: A Scalable Tree Boosting System." KDD 2016.
- CMS FY2025 Improper Payments Fact Sheet: https://www.cms.gov/newsroom/fact-sheets/fiscal-year-2025-improper-payments-fact-sheet
- GAO Medicare and Medicaid Program Integrity: https://www.gao.gov/products/gao-24-107487

---

*Built on publicly available synthetic data released by CMS. No real patient or provider data was used at any stage of this project.*
