"""
ML training pipeline for Malaysia Inflation Intelligence.

Trains Prophet (baseline) and XGBoost models per CPI division
using walk-forward cross-validation. Logs all experiments to
MLflow and saves the best model to GCS.

Usage:
    python train.py                          # train all divisions
    python train.py --division 01            # train one division
    python train.py --division 01 --state Selangor  # one series
"""

import os
import json
import logging
import argparse
import warnings
import tempfile
from datetime import datetime


warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
import xgboost as xgb
import shap
import joblib
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from google.cloud import storage

from prepare_features import prepare, get_feature_columns, PROJECT_ID

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

GCS_BUCKET = f"{PROJECT_ID}-ml-artifacts"
MLFLOW_EXPERIMENT = "malaysia-inflation-cpi-forecast"

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute evaluation metrics for a set of predictions.

    Returns:
        dict with MAE, RMSE, MAPE, and directional accuracy
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # MAPE — mean absolute percentage error
    # Add small epsilon to avoid division by zero
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    # Directional accuracy — did the model predict the right direction?
    # i.e. if actual went up, did the model also predict up?
    actual_direction = np.diff(y_true)
    pred_direction = np.diff(y_pred)
    directional_acc = np.mean(
        np.sign(actual_direction) == np.sign(pred_direction)
    ) * 100

    return {
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "mape": round(float(mape), 4),
        "directional_accuracy": round(float(directional_acc), 2),
    }


def evaluate_across_splits(
    splits: list,
    features: list,
    model_type: str,
    params: dict = None,
) -> dict:
    """
    Run a model across all walk-forward splits and average the metrics.
    """
    all_metrics = []

    for i, (train_df, test_df) in enumerate(splits):
        X_train = train_df[features].values
        y_train = train_df["cpi_index"].values
        X_test = test_df[features].values
        y_test = test_df["cpi_index"].values

        if model_type == "Prophet":
            y_pred = prophet_model_fn(
                X_train, y_train, X_test,
                train_df=train_df,
                test_df=test_df,
            )
        else:
            y_pred = xgboost_model_fn(X_train, y_train, X_test, params=params)

        metrics = compute_metrics(y_test, y_pred)
        all_metrics.append(metrics)

        logger.info(
            f"  [{model_type}] Split {i+1}: "
            f"MAE={metrics['mae']:.4f}, "
            f"RMSE={metrics['rmse']:.4f}, "
            f"MAPE={metrics['mape']:.2f}%, "
            f"DirAcc={metrics['directional_accuracy']:.1f}%"
        )

    avg_metrics = {
        key: round(float(np.mean([m[key] for m in all_metrics])), 4)
        for key in all_metrics[0].keys()
    }

    logger.info(
        f"  [{model_type}] Average: "
        f"MAE={avg_metrics['mae']:.4f}, "
        f"RMSE={avg_metrics['rmse']:.4f}, "
        f"MAPE={avg_metrics['mape']:.2f}%, "
        f"DirAcc={avg_metrics['directional_accuracy']:.1f}%"
    )

    return avg_metrics

def prophet_model_fn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> np.ndarray:
    """
    Prophet baseline model.

    Prophet only uses date and target value — it ignores the
    engineered features entirely. This is intentional — Prophet
    is the naive baseline. If XGBoost can't beat Prophet using
    all the features you engineered, something is wrong.

    Parameters:
        train_df: full training DataFrame (needs date column)
        test_df: full test DataFrame (needs date column)

    Returns:
        y_pred array aligned to test_df rows
    """
    # Prophet requires columns named 'ds' and 'y'
    prophet_train = pd.DataFrame({
        "ds": pd.to_datetime(train_df["date"]),
        "y": y_train,
    })

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
    )

    # Suppress Prophet's verbose output
    import logging as lg
    lg.getLogger("prophet").setLevel(lg.WARNING)
    lg.getLogger("cmdstanpy").setLevel(lg.WARNING)

    model.fit(prophet_train)

    # Make future dataframe for test period
    future = pd.DataFrame({"ds": pd.to_datetime(test_df["date"])})
    forecast = model.predict(future)

    return forecast["yhat"].values


def xgboost_model_fn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    params: dict = None,
) -> np.ndarray:
    """
    XGBoost model using engineered features.

    This is the main model. Unlike Prophet, it uses all 11
    feature columns — lag features, rolling averages, PPI leads,
    and OPR rate. The goal is for this to outperform Prophet
    by learning the economic relationships in the data.

    Parameters:
        params: XGBoost hyperparameters dict. If None uses defaults.

    Returns:
        y_pred array
    """
    if params is None:
        params = {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
        }

    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, np.zeros(len(X_test)))],
        verbose=False,
    )

    return model.predict(X_test)


def train_xgboost_final(
    df: pd.DataFrame,
    features: list,
    params: dict = None,
) -> xgb.XGBRegressor:
    """
    Train a final XGBoost model on the full non-COVID dataset.

    This is the production model — trained on all available
    data after walk-forward CV confirms the hyperparameters work.

    Returns:
        Fitted XGBRegressor
    """
    if params is None:
        params = {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
        }

    X = df[features].values
    y = df["cpi_index"].values

    model = xgb.XGBRegressor(**params)
    model.fit(X, y, verbose=False)

    logger.info(f"  Final model trained on {len(df):,} rows")
    return model


def compute_shap_values(
    model: xgb.XGBRegressor,
    X: np.ndarray,
    features: list,
) -> dict:
    """
    Compute SHAP feature importance for the trained XGBoost model.

    SHAP (SHapley Additive exPlanations) explains WHY the model
    made each prediction by attributing importance to each feature.

    Returns:
        dict mapping feature name to mean absolute SHAP value
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Mean absolute SHAP value per feature
    mean_shap = np.abs(shap_values).mean(axis=0)
    shap_dict = dict(zip(features, mean_shap.tolist()))

    # Sort by importance descending
    shap_dict = dict(
        sorted(shap_dict.items(), key=lambda x: x[1], reverse=True)
    )

    return shap_dict

def save_model_to_gcs(
    model: xgb.XGBRegressor,
    project_id: str,
    model_name: str,
) -> str:
    """
    Serialize model with joblib and upload to GCS ml-artifacts bucket.

    Returns:
        GCS URI of the saved model file
    """
    # Save locally first
    import tempfile
    local_path = os.path.join(tempfile.gettempdir(), f"{model_name}.joblib")
    joblib.dump(model, local_path)

    # Upload to GCS
    client = storage.Client(project=project_id)
    bucket = client.bucket(GCS_BUCKET)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gcs_path = f"models/{model_name}_{timestamp}.joblib"
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)

    gcs_uri = f"gs://{GCS_BUCKET}/{gcs_path}"
    logger.info(f"  Model saved to {gcs_uri}")
    return gcs_uri


def save_shap_to_gcs(
    shap_dict: dict,
    project_id: str,
    model_name: str,
) -> str:
    """
    Save SHAP feature importance dict as JSON to GCS.

    Returns:
        GCS URI of the SHAP JSON file
    """
    client = storage.Client(project=project_id)
    bucket = client.bucket(GCS_BUCKET)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gcs_path = f"shap/{model_name}_shap_{timestamp}.json"
    blob = bucket.blob(gcs_path)
    blob.upload_from_string(
        json.dumps(shap_dict, indent=2),
        content_type="application/json",
    )

    gcs_uri = f"gs://{GCS_BUCKET}/{gcs_path}"
    logger.info(f"  SHAP values saved to {gcs_uri}")
    return gcs_uri


def log_run_to_mlflow(
    run_name: str,
    model_type: str,
    metrics: dict,
    params: dict,
    gcs_uri: str,
    shap_dict: dict = None,
) -> str:
    """
    Log a training run to MLflow experiment tracking.

    Logs:
    - Parameters (model hyperparameters)
    - Metrics (MAE, RMSE, MAPE, directional accuracy)
    - Tags (model type, GCS URI)
    - SHAP values as a JSON artifact (if provided)

    Returns:
        MLflow run ID
    """
    with mlflow.start_run(run_name=run_name) as run:

        # Log hyperparameters
        mlflow.log_params(params)

        # Log evaluation metrics
        mlflow.log_metrics(metrics)

        # Log tags for easy filtering in MLflow UI
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("gcs_uri", gcs_uri)
        mlflow.set_tag("project", "malaysia-inflation-intel")

        # Log SHAP values as artifact if provided
        if shap_dict:
            shap_path = os.path.join(tempfile.gettempdir(), "shap_values.json")
            with open(shap_path, "w") as f:
                json.dump(shap_dict, f, indent=2)
            mlflow.log_artifact(shap_path, artifact_path="shap")

        run_id = run.info.run_id
        logger.info(f"  MLflow run ID: {run_id}")
        return run_id

def main():
    """
    Main training pipeline.

    1. Prepare features from BigQuery
    2. Train Prophet baseline across walk-forward splits
    3. Train XGBoost across walk-forward splits
    4. Compare models — log both to MLflow
    5. Train final XGBoost on full dataset
    6. Compute SHAP values
    7. Save model and SHAP to GCS
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--division",
        type=str,
        default=None,
        help="CPI division to train on (e.g. '01'). If None trains on all.",
    )
    parser.add_argument(
        "--state",
        type=str,
        default=None,
        help="State to train on (e.g. 'Selangor'). If None trains on all.",
    )
    args = parser.parse_args()

    # Step 1 — prepare features
    logger.info("=" * 60)
    logger.info("PHASE 4 — ML TRAINING PIPELINE")
    logger.info("=" * 60)

    df, splits, features = prepare(PROJECT_ID)

    # Filter to specific division or state if requested
    if args.division:
        df = df[df["division"] == args.division].copy()
        splits = [
            (
                train[train["division"] == args.division].copy(),
                test[test["division"] == args.division].copy(),
            )
            for train, test in splits
        ]
        logger.info(f"Filtered to division: {args.division} ({len(df):,} rows)")

    if args.state:
        df = df[df["state"] == args.state].copy()
        splits = [
            (
                train[train["state"] == args.state].copy(),
                test[test["state"] == args.state].copy(),
            )
            for train, test in splits
        ]
        logger.info(f"Filtered to state: {args.state} ({len(df):,} rows)")

    # Set up MLflow experiment
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # Step 2 — Prophet baseline
    logger.info("\nTraining Prophet baseline...")
    prophet_metrics = evaluate_across_splits(
        splits=splits,
        features=features,
        model_type="Prophet",
    )

    # Step 3 — XGBoost
    logger.info("\nTraining XGBoost...")
    xgb_params = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    }

    xgb_metrics = evaluate_across_splits(
        splits=splits,
        features=features,
        model_type="XGBoost",
        params=xgb_params,
    )

    # Step 4 — Compare and log both to MLflow
    logger.info("\nLogging runs to MLflow...")

    model_name = "cpi_forecast"
    if args.division:
        model_name += f"_div{args.division}"
    if args.state:
        model_name += f"_{args.state.lower().replace(' ', '_')}"

    log_run_to_mlflow(
        run_name=f"prophet_{model_name}",
        model_type="prophet",
        metrics=prophet_metrics,
        params={"model": "prophet", "seasonality_mode": "multiplicative"},
        gcs_uri="n/a",
    )

    # Step 5 — Train final XGBoost on full dataset
    logger.info("\nTraining final XGBoost on full dataset...")
    final_model = train_xgboost_final(df, features, params=xgb_params)

    # Step 6 — Compute SHAP values
    logger.info("\nComputing SHAP values...")
    X_sample = df[features].values[:1000]
    shap_dict = compute_shap_values(final_model, X_sample, features)

    logger.info("  Feature importance (SHAP):")
    for feat, val in list(shap_dict.items())[:5]:
        logger.info(f"    {feat}: {val:.4f}")

    # Step 7 — Save model and SHAP to GCS
    logger.info("\nSaving model and SHAP to GCS...")
    gcs_uri = save_model_to_gcs(final_model, PROJECT_ID, model_name)
    shap_uri = save_shap_to_gcs(shap_dict, PROJECT_ID, model_name)

    # Log final XGBoost run with model URI and SHAP
    log_run_to_mlflow(
        run_name=f"xgboost_{model_name}_final",
        model_type="xgboost",
        metrics=xgb_metrics,
        params=xgb_params,
        gcs_uri=gcs_uri,
        shap_dict=shap_dict,
    )

    # Step 8 — Final comparison summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Metric':<25} {'Prophet':>10} {'XGBoost':>10}")
    logger.info("-" * 45)
    for metric in ["mae", "rmse", "mape", "directional_accuracy"]:
        logger.info(
            f"{metric:<25} "
            f"{prophet_metrics[metric]:>10.4f} "
            f"{xgb_metrics[metric]:>10.4f}"
        )

    winner = "XGBoost" if xgb_metrics["mae"] < prophet_metrics["mae"] else "Prophet"
    logger.info(f"\nWinner by MAE: {winner}")
    logger.info(f"Model saved to: {gcs_uri}")
    logger.info(f"SHAP saved to:  {shap_uri}")
    logger.info("MLflow experiment: " + MLFLOW_EXPERIMENT)


if __name__ == "__main__":
    main()