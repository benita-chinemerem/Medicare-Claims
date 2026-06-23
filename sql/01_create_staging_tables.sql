-- =============================================================
-- 01_create_staging_tables.sql
-- Raw ingestion layer. Column types are deliberately permissive
-- (TEXT / VARCHAR) because DE-SynPUF has coarsened fields that
-- require inspection before casting. Type enforcement happens in
-- the analytics schema transform (02_create_analytics_tables.sql).
-- =============================================================

CREATE SCHEMA IF NOT EXISTS staging;

-- -------------------------------------------------------------
-- Beneficiary Summary (one table per year, unified here by year col)
-- -------------------------------------------------------------
DROP TABLE IF EXISTS staging.beneficiary_summary CASCADE;
CREATE TABLE staging.beneficiary_summary (
    desynpuf_id             VARCHAR(50),
    bene_birth_dt           VARCHAR(10),
    bene_death_dt           VARCHAR(10),
    bene_sex_ident_cd       VARCHAR(5),
    bene_race_cd            VARCHAR(5),
    bene_esrd_ind           VARCHAR(5),
    sp_state_code           VARCHAR(5),
    bene_county_cd          VARCHAR(10),
    bene_hi_cvrage_tot_mons VARCHAR(5),  -- Part A coverage months
    bene_smi_cvrage_tot_mons VARCHAR(5), -- Part B coverage months
    bene_hmo_cvrage_tot_mons VARCHAR(5),
    plan_cvrg_mos_num       VARCHAR(5),
    sp_alzhdmta             VARCHAR(5),  -- Alzheimer's flag
    sp_chf                  VARCHAR(5),  -- Congestive heart failure flag
    sp_chrnkidn             VARCHAR(5),  -- Chronic kidney disease flag
    sp_cncr                 VARCHAR(5),  -- Cancer flag
    sp_copd                 VARCHAR(5),  -- COPD flag
    sp_depressn             VARCHAR(5),  -- Depression flag
    sp_diabetes             VARCHAR(5),  -- Diabetes flag
    sp_ischmcht             VARCHAR(5),  -- Ischemic heart disease flag
    sp_osteoprs             VARCHAR(5),  -- Osteoporosis flag
    sp_ra_oa                VARCHAR(5),  -- RA/OA flag
    sp_strketia             VARCHAR(5),  -- Stroke/TIA flag
    medreimb_ip             VARCHAR(20), -- Inpatient Medicare reimbursement
    benres_ip               VARCHAR(20),
    pppymt_ip               VARCHAR(20),
    medreimb_op             VARCHAR(20), -- Outpatient Medicare reimbursement
    benres_op               VARCHAR(20),
    pppymt_op               VARCHAR(20),
    medreimb_car            VARCHAR(20), -- Carrier Medicare reimbursement
    benres_car              VARCHAR(20),
    pppymt_car              VARCHAR(20),
    year                    SMALLINT,   -- 2008, 2009, or 2010
    sample_id               SMALLINT    -- 1 or 2
);

-- -------------------------------------------------------------
-- Carrier Claims (Part B physician/supplier claims)
-- Segments A and B are loaded into the same table.
-- -------------------------------------------------------------
DROP TABLE IF EXISTS staging.carrier_claims CASCADE;
CREATE TABLE staging.carrier_claims (
    desynpuf_id             VARCHAR(50),
    clm_id                  VARCHAR(50),
    clm_from_dt             VARCHAR(10),
    clm_thru_dt             VARCHAR(10),
    clm_pmt_amt             VARCHAR(20),
    nch_clm_type_cd         VARCHAR(5),
    clm_disp_cd             VARCHAR(5),
    nch_carr_clm_sbmtd_chrg_amt VARCHAR(20),
    nch_carr_clm_allwd_amt  VARCHAR(20),
    carr_num                VARCHAR(10),
    carr_clm_cash_ddctbl_apld_amt VARCHAR(20),
    hcfa_priv_pay_amt       VARCHAR(20),
    carr_clm_rfrng_pin_num  VARCHAR(20),
    prncpal_dgns_cd         VARCHAR(10),
    admtng_icd9_dgns_cd     VARCHAR(10),
    -- Diagnosis codes (1-8)
    icd9_dgns_cd_1          VARCHAR(10),
    icd9_dgns_cd_2          VARCHAR(10),
    icd9_dgns_cd_3          VARCHAR(10),
    icd9_dgns_cd_4          VARCHAR(10),
    icd9_dgns_cd_5          VARCHAR(10),
    icd9_dgns_cd_6          VARCHAR(10),
    icd9_dgns_cd_7          VARCHAR(10),
    icd9_dgns_cd_8          VARCHAR(10),
    -- HCPCS procedure codes (1-13 lines on carrier claims)
    hcpcs_cd_1              VARCHAR(10),
    hcpcs_cd_2              VARCHAR(10),
    hcpcs_cd_3              VARCHAR(10),
    hcpcs_cd_4              VARCHAR(10),
    hcpcs_cd_5              VARCHAR(10),
    hcpcs_cd_6              VARCHAR(10),
    hcpcs_cd_7              VARCHAR(10),
    hcpcs_cd_8              VARCHAR(10),
    hcpcs_cd_9              VARCHAR(10),
    hcpcs_cd_10             VARCHAR(10),
    hcpcs_cd_11             VARCHAR(10),
    hcpcs_cd_12             VARCHAR(10),
    hcpcs_cd_13             VARCHAR(10),
    -- Line-level details
    line_nch_pmt_amt        VARCHAR(20),
    line_bene_ptb_ddctbl_amt VARCHAR(20),
    line_coinsrnc_amt       VARCHAR(20),
    line_bene_pmt_amt       VARCHAR(20),
    line_prvdr_pmt_amt      VARCHAR(20),
    line_bene_part_b_enrl_dt VARCHAR(10),
    line_cms_type_srvc_cd   VARCHAR(5),
    line_place_of_srvc_cd   VARCHAR(5),
    line_prcsg_ind_cd       VARCHAR(5),
    at_physn_npi            VARCHAR(20),   -- Attending physician NPI (primary fraud signal)
    op_physn_npi            VARCHAR(20),
    ot_physn_npi            VARCHAR(20),
    sample_id               SMALLINT
);

-- -------------------------------------------------------------
-- Outpatient Claims
-- -------------------------------------------------------------
DROP TABLE IF EXISTS staging.outpatient_claims CASCADE;
CREATE TABLE staging.outpatient_claims (
    desynpuf_id             VARCHAR(50),
    clm_id                  VARCHAR(50),
    segment                 VARCHAR(5),
    clm_from_dt             VARCHAR(10),
    clm_thru_dt             VARCHAR(10),
    prvdr_num               VARCHAR(20),
    clm_pmt_amt             VARCHAR(20),
    nch_prmry_pyr_clm_pd_amt VARCHAR(20),
    at_physn_npi            VARCHAR(20),
    op_physn_npi            VARCHAR(20),
    clm_fac_type_cd         VARCHAR(5),
    clm_srvc_clsfctn_type_cd VARCHAR(5),
    clm_freq_cd             VARCHAR(5),
    nch_clm_type_cd         VARCHAR(5),
    clm_mdcr_non_pmt_rsn_cd VARCHAR(5),
    nch_near_line_rec_ident_cd VARCHAR(5),
    clm_poa_ind_sw1         VARCHAR(5),
    icd9_dgns_cd_1          VARCHAR(10),
    icd9_dgns_cd_2          VARCHAR(10),
    icd9_dgns_cd_3          VARCHAR(10),
    icd9_dgns_cd_4          VARCHAR(10),
    icd9_dgns_cd_5          VARCHAR(10),
    icd9_dgns_cd_6          VARCHAR(10),
    icd9_dgns_cd_7          VARCHAR(10),
    icd9_dgns_cd_8          VARCHAR(10),
    icd9_dgns_cd_9          VARCHAR(10),
    icd9_dgns_cd_10         VARCHAR(10),
    hcpcs_cd_1              VARCHAR(10),
    hcpcs_cd_2              VARCHAR(10),
    hcpcs_cd_3              VARCHAR(10),
    hcpcs_cd_4              VARCHAR(10),
    hcpcs_cd_5              VARCHAR(10),
    hcpcs_cd_6              VARCHAR(10),
    hcpcs_cd_7              VARCHAR(10),
    hcpcs_cd_8              VARCHAR(10),
    hcpcs_cd_9              VARCHAR(10),
    hcpcs_cd_10             VARCHAR(10),
    hcpcs_cd_11             VARCHAR(10),
    hcpcs_cd_12             VARCHAR(10),
    hcpcs_cd_13             VARCHAR(10),
    hcpcs_cd_14             VARCHAR(10),
    hcpcs_cd_15             VARCHAR(10),
    hcpcs_cd_16             VARCHAR(10),
    hcpcs_cd_17             VARCHAR(10),
    hcpcs_cd_18             VARCHAR(10),
    hcpcs_cd_19             VARCHAR(10),
    hcpcs_cd_20             VARCHAR(10),
    hcpcs_cd_21             VARCHAR(10),
    hcpcs_cd_22             VARCHAR(10),
    hcpcs_cd_23             VARCHAR(10),
    hcpcs_cd_24             VARCHAR(10),
    hcpcs_cd_25             VARCHAR(10),
    hcpcs_cd_26             VARCHAR(10),
    hcpcs_cd_27             VARCHAR(10),
    hcpcs_cd_28             VARCHAR(10),
    hcpcs_cd_29             VARCHAR(10),
    hcpcs_cd_30             VARCHAR(10),
    hcpcs_cd_31             VARCHAR(10),
    hcpcs_cd_32             VARCHAR(10),
    hcpcs_cd_33             VARCHAR(10),
    hcpcs_cd_34             VARCHAR(10),
    hcpcs_cd_35             VARCHAR(10),
    hcpcs_cd_36             VARCHAR(10),
    hcpcs_cd_37             VARCHAR(10),
    hcpcs_cd_38             VARCHAR(10),
    hcpcs_cd_39             VARCHAR(10),
    hcpcs_cd_40             VARCHAR(10),
    hcpcs_cd_41             VARCHAR(10),
    hcpcs_cd_42             VARCHAR(10),
    hcpcs_cd_43             VARCHAR(10),
    hcpcs_cd_44             VARCHAR(10),
    hcpcs_cd_45             VARCHAR(10),
    revenue_cd_1            VARCHAR(10),
    revenue_cd_2            VARCHAR(10),
    revenue_cd_3            VARCHAR(10),
    revenue_cd_4            VARCHAR(10),
    revenue_cd_5            VARCHAR(10),
    sample_id               SMALLINT
);

-- Staging load tracking table
DROP TABLE IF EXISTS staging.load_log CASCADE;
CREATE TABLE staging.load_log (
    id              SERIAL PRIMARY KEY,
    table_name      VARCHAR(100),
    sample_id       SMALLINT,
    file_name       VARCHAR(255),
    rows_loaded     INTEGER,
    loaded_at       TIMESTAMP DEFAULT NOW(),
    status          VARCHAR(20) DEFAULT 'success'
);

COMMENT ON SCHEMA staging IS
    'Raw ingestion layer. Column types are TEXT/VARCHAR pending type enforcement in the analytics schema.';
