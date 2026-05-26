"""
Dataset configuration for DOSM and BNM data ingestion.
All URLs are public parquet endpoints from data.gov.my / OpenDosm
"""

DATASETS = {
    "cpi_state": {
        "url": "https://storage.dosm.gov.my/cpi/cpi_2d_state.parquet",
        "gcs_path": "cpi/cpi_2d_state.parquet",
        "bq_table": "raw.cpi_state",
        "description": "Monthly CPI state and division (13 cattegories x 16 stattes)"
    },"cpi_national": {
        "url": "https://storage.dosm.gov.my/cpi/cpi_2d.parquet",
        "gcs_path": "cpi/cpi_2d_national.parquet",
        "bq_table": "raw.cpi_national",
        "description": "Monthly CPI national level by division (13 categories)",
    },
    "ppi_headline": {
        "url": "https://storage.dosm.gov.my/ppi/ppi.parquet",
        "gcs_path": "ppi/ppi_headline.parquet",
        "bq_table": "raw.ppi_headline",
        "description": "Monthly headline PPI with seasonal adjustment",
    },
    "ppi_sitc": {
        "url": "https://storage.dosm.gov.my/ppi/ppi_sitc.parquet",
        "gcs_path": "ppi/ppi_sitc.parquet",
        "bq_table": "raw.ppi_sitc",
        "description": "Monthly PPI by SITC section (1-digit)",
    },
    "hies_state": {
        "url": "https://storage.dosm.gov.my/hies/hies_state.parquet",
        "gcs_path": "hies/hies_state.parquet",
        "bq_table": "raw.hies_state",
        "description": "Household income, expenditure, poverty by state (HIES 2022)",
    },
    "interest_rates": {
        "url": "https://storage.data.gov.my/finsector/interest_rates.parquet",
        "gcs_path": "bnm/interest_rates.parquet",
        "bq_table": "raw.interest_rates",
        "description": "Monthly interest rates including OPR (BNM)",
    },

}