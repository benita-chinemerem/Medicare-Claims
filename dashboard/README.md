# Dashboard Setup Guide

This guide walks through connecting Power BI Desktop to the PostgreSQL analytics
database and configuring all four dashboard pages.

---

## Prerequisites

- **Power BI Desktop** (free): [Download from Microsoft](https://powerbi.microsoft.com/desktop/)
- Pipeline must have completed at least one full DAG 2 (weekly scoring) run
- PostgreSQL must be running via Docker Compose (`docker-compose up -d`)

---

## Connection Setup

### 1. Open Power BI Desktop

### 2. Connect to PostgreSQL

- Click **Get Data** (Home ribbon)
- Search for **PostgreSQL database** and select it
- Click **Connect**

Enter the connection details:

| Field    | Value       |
|----------|-------------|
| Server   | `localhost`  |
| Database | `fraud_claims` |
| Port     | `5432`       |

- Click **OK**
- Authentication: select **Database** tab
  - Username: `airflow`
  - Password: `airflow` (or your value from `.env`)

### 3. Select Tables

In the Navigator, select all tables under these schemas:

- `scores` — all tables and the `vw_dashboard_overview` view
- `features.provider_features`
- `analytics.carrier_claims` (optional, for claim-level drill-down)

Click **Load** (not Transform for now).

---

## Page 1 — Overview

**Purpose:** High-level summary of claims volume, flagged counts, and at-risk dollars
by week.

**Visuals to build:**

| Visual | Type | Data |
|---|---|---|
| Total claims processed (all time) | Card | SUM of `scoring_runs.carrier_claims_processed` |
| Total providers flagged | Card | SUM of `scoring_runs.providers_flagged` |
| Claims processed by run date | Line chart | X: `scoring_runs.completed_at`, Y: `carrier_claims_processed` |
| Providers flagged by run date | Column chart | X: `completed_at`, Y: `providers_flagged` |
| At-risk proxy trend | Line chart | X: `completed_at`, Y: `vw_dashboard_overview.at_risk_score_proxy` |

**Filters:** Add a date range slicer using `scoring_runs.completed_at`.

---

## Page 2 — Provider Risk Ranking

**Purpose:** Sortable, filterable table of all scored providers with risk decile,
top SHAP reason codes, and key metrics. This is the primary investigator-facing view.

**Visuals to build:**

| Visual | Type | Data |
|---|---|---|
| Provider risk table | Table | `provider_risk_scores`: npi, risk_score, risk_decile, top_reason_1/2/3, total_claims, distinct_benes, avg_submitted_to_allowed_ratio, duplicate_rate, pct_weekend_claims |
| Risk decile distribution | Bar chart | X: `risk_decile`, Y: COUNT of providers |
| Top 10 flagged providers | Table (filtered) | Same as risk table, filtered to `is_flagged = TRUE`, top 10 by risk_score |

**Filters:**
- Slicer: `is_flagged` (True/False toggle)
- Slicer: `risk_decile` (range selector, 1-10)
- Slicer: `period_year`

**Conditional formatting:**
- `risk_score` column: gradient from green (0) to red (100)
- `duplicate_rate` column: red if > 0.05

**Drillthrough:** Configure drillthrough from the provider table to Page 3
(Claim Explanation) using `at_physn_npi` as the drillthrough field.

---

## Page 3 — Claim Explanation (SHAP Reason Codes)

**Purpose:** Shows the SHAP-derived reason codes for one provider selected via
drillthrough from Page 2. Investigators use this page to understand why a
provider was flagged before opening a case.

**Visuals to build:**

| Visual | Type | Data |
|---|---|---|
| Provider summary card | Card | Selected NPI, risk_score, risk_decile |
| SHAP feature importance | Horizontal bar chart | X: `shap_values.shap_value` (absolute), Y: `feature_name`, sorted descending |
| Feature value table | Table | `shap_values`: feature_name, feature_value, shap_value, feature_rank |
| Reason codes text | Multi-row card | `provider_risk_scores.top_reason_1`, `top_reason_2`, `top_reason_3` |
| SHAP waterfall chart (optional) | Custom visual | Requires "Charticulator" or similar custom visual from AppSource |

**Note:** The SHAP waterfall chart visual is optional but highly effective for
presentations. The horizontal bar chart of feature importance achieves the same
communication goal with built-in Power BI visuals.

---

## Page 4 — Pipeline Health

**Purpose:** Operational view showing run history, processing volumes, and active
model version. Demonstrates continuous automated monitoring.

**Visuals to build:**

| Visual | Type | Data |
|---|---|---|
| Last successful run | Card | MAX(`scoring_runs.completed_at`) WHERE status = 'success' |
| Total runs completed | Card | COUNT of `scoring_runs` WHERE status = 'success' |
| Run history table | Table | `scoring_runs`: dag_run_id, completed_at, status, carrier_claims_processed, providers_scored, providers_flagged |
| Run status over time | Column chart | X: `completed_at`, Y: `providers_scored`, colored by `status` |
| Active model | Card | `model_registry.model_version` WHERE is_active = TRUE |
| Model training history | Table | `model_registry`: model_version, trained_at, training_samples, avg_anomaly_score, is_active, promoted_at |

**Tip:** Add a green/red conditional format on the `status` column:
- green for `success`
- red for `failed`
- amber for `running`

---

## Refreshing Data

Power BI Desktop refreshes data on demand. Click **Refresh** in the Home ribbon
after each DAG 2 run to pull the latest scores.

For automated refresh, publish the `.pbix` file to Power BI Service and configure
a scheduled refresh. This requires a Power BI Pro license and an on-premises
data gateway (since PostgreSQL is local). For this prototype, manual refresh on
demand is sufficient.

---

## Taking Screenshots for the Exhibit Packet

The exhibit packet requires screenshots from all four pages. For each:

1. Set all slicers to show the full available date range
2. Ensure at least one DAG 2 run has completed successfully (check Page 4)
3. Take full-page screenshots at 1920x1080 resolution
4. Save as PNG with descriptive names:
   - `dashboard_page1_overview.png`
   - `dashboard_page2_provider_risk_ranking.png`
   - `dashboard_page3_claim_explanation.png`
   - `dashboard_page4_pipeline_health.png`

Screenshots are embedded in the white paper (Section 7) and attached to the
exhibit packet as separate files.

---

## Troubleshooting

**"Cannot connect to server"**
- Confirm Docker is running: `docker ps` should show `fraud_anomaly_postgres`
- Confirm port 5432 is not blocked by a local firewall

**"Relation does not exist"**
- Run the four SQL schema scripts in order (see main README)
- Confirm DAG 1 has completed successfully

**"No data in scores tables"**
- DAG 2 must have completed at least one successful run
- Check Airflow UI at http://localhost:8080 for run status and logs

**"SHAP values table is empty"**
- DAG 2 runs SHAP computation only for flagged providers (risk_score >= 75)
- Lower `RISK_SCORE_THRESHOLD` in `.env` temporarily to flag more providers
  during initial testing
