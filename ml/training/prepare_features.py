"""
Feature preparation for Malaysia Inflation Intelligence ML training.
Pulls mart_cpi_features from BigQuery, handles nulls, and creates
walk-forward cross-validation splits for time series training.
"""

import os
import logging
from datetime import datetime

import pandas as pd
import numpy as np
from google.cloud import bigquery

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ID = "modular-rush-495414-i2"
DATASET = "dbt_"
TABLE = "mart_cpi_features"
TARGET_COL = "cpi_index"
DATE_COL = "date"

def load_features(project_id: str) -> pd.DataFrame:
    """Pull mart_cpi_features from BigQuery into a DataFrame."""
    client = bigquery.Client(project=project_id)
    query = f"""
        select *
        from `{project_id}.{DATASET}.{TABLE}`
        order by state, division, date
    """
    logger.info("Loading features from BigQuery...")
    df = client.query(query).to_dataframe()
    logger.info(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df

def handle_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle nulls in the feature table.

    Lag features are null for the first N months of each series
    (e.g. cpi_lag_12 is null for the first 12 months).
    OPR is null for recent months where BNM data lags.

    Strategy:
    - Lag/rolling features: drop rows where lag_1 is null
      (keeps only rows with at least 1 month of history)
    - OPR: forward-fill within each state/division group
    - Remaining nulls: fill with column median
    """
    logger.info(f"  Nulls before handling: {df.isnull().sum().sum()}")

    # Forward-fill OPR within date order (carry last known rate forward)
    df = df.sort_values(DATE_COL)
    df["opr_rate"] = df["opr_rate"].ffill()

    # Drop rows where cpi_lag_1 is null
    # These are the first months of each series with no history yet
    df = df.dropna(subset=["cpi_lag_1"])

    # Fill any remaining nulls with column median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    logger.info(f"  Nulls after handling: {df.isnull().sum().sum()}")
    logger.info(f"  Rows remaining: {len(df):,}")
    return df

def create_walk_forward_splits(
    df: pd.DataFrame,
    train_months: int = 36,
    gap_months: int = 1,
    test_months: int = 6,
    n_splits: int = 5,
) -> list:
    """
    Create walk-forward cross-validation splits for time series.

    Walk-forward CV respects temporal order — training data always
    comes before test data. This prevents data leakage.

    Parameters:
        train_months: how many months to train on
        gap_months: gap between train end and test start
          (simulates real-world prediction lag)
        test_months: how many months to test on
        n_splits: number of CV folds

    Returns:
        List of (train_df, test_df) tuples
    """
    dates = sorted(df[DATE_COL].unique())
    splits = []

    # Start from enough history to have a full training window
    start_idx = train_months

    for i in range(n_splits):
        # Each split moves forward by test_months
        split_start = start_idx + (i * test_months)

        if split_start + gap_months + test_months > len(dates):
            break

        train_end = dates[split_start - 1]
        test_start = dates[split_start + gap_months]
        test_end = dates[min(split_start + gap_months + test_months - 1, len(dates) - 1)]

        train_df = df[df[DATE_COL] <= train_end].copy()
        test_df = df[
            (df[DATE_COL] >= test_start) &
            (df[DATE_COL] <= test_end)
        ].copy()

        splits.append((train_df, test_df))
        logger.info(
            f"  Split {i+1}: train up to {train_end}, "
            f"test {test_start} to {test_end} "
            f"({len(train_df):,} train, {len(test_df):,} test rows)"
        )

    return splits


def get_feature_columns() -> list:
    """Return the list of feature columns used for ML training."""
    return [
        "cpi_lag_1",
        "cpi_lag_3",
        "cpi_lag_6",
        "cpi_lag_12",
        "cpi_ma_3",
        "cpi_ma_12",
        "ppi_index",
        "ppi_lag_1",
        "ppi_lag_2",
        "ppi_lag_3",
        "opr_rate",
    ]


def prepare(project_id: str = PROJECT_ID) -> tuple:
    """Full feature preparation pipeline."""
    df = load_features(project_id)
    df = handle_nulls(df)

    # Exclude COVID period from training
    # (anomalous patterns would confuse the model)
    df_train = df[df["is_covid_period"] == False].copy()
    logger.info(f"  Rows after excluding COVID period: {len(df_train):,}")

    splits = create_walk_forward_splits(df_train)
    features = get_feature_columns()

    logger.info(f"  Feature columns: {len(features)}")
    logger.info(f"  Walk-forward splits: {len(splits)}")

    return df_train, splits, features


if __name__ == "__main__":
    df, splits, features = prepare()
    logger.info("Feature preparation complete")
    logger.info(f"Final dataset: {df.shape}")
    logger.info(f"Features: {features}")
    logger.info(f"Splits: {len(splits)}")