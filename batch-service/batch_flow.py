"""Batch prediction and monitoring flow for wind production forecasting."""

# pylint: disable=import-error
# pylint: disable=line-too-long

import os
from datetime import datetime

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from prefect import flow, task
from prefect.deployments import run_deployment
from sqlalchemy import create_engine
from sqlalchemy_utils import create_database, database_exists

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", "http://experiment-tracking:5000"
)
MODEL_NAME = os.getenv("MODEL_NAME", "wind-production-model")
DATA_DIR = os.getenv("DATA_DIR", "/data")
BATCH_DATA_DIR = os.getenv("BATCH_DATA_DIR", "/batch-data")
RMSE_THRESHOLD = float(os.getenv("RMSE_THRESHOLD", "400000"))
RETRAIN_API_URL = os.getenv("RETRAIN_API_URL", "http://orchestration:4200/api")

DB_USER = os.getenv("POSTGRES_USER", "admin")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
DB_HOST = os.getenv("DB_HOST", "database")
DB_NAME = "metrics"
DB_URI = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"

REFERENCE_PATH = os.path.join(BATCH_DATA_DIR, "reference.parquet")


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def get_engine():
    """Create database and return SQLAlchemy engine."""
    if not database_exists(DB_URI):
        create_database(DB_URI)
    return create_engine(DB_URI)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


@task(name="load-latest-model")
def load_latest_model():
    """Load the latest registered model version from MLFlow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()
    versions = client.search_model_versions(
        f"name='{MODEL_NAME}'", order_by=["version_number DESC"], max_results=1
    )
    if not versions:
        raise RuntimeError(
            f"No registered versions found for model '{MODEL_NAME}'. Run the training pipeline first."
        )
    latest = versions[0]
    model = mlflow.sklearn.load_model(f"runs:/{latest.run_id}/model")
    print(f"Loaded model {MODEL_NAME} version {latest.version}")
    return model, latest.run_id


@task(name="load-and-prepare-data")
def load_and_prepare_data(data_dir: str) -> pd.DataFrame:
    """Load wind and production data, join and engineer features."""
    wind_df = pd.read_csv(os.path.join(data_dir, "wind.csv"), na_values=["NULL"])
    prod_df = pd.read_csv(os.path.join(data_dir, "production.csv"))

    wind_df["date"] = pd.to_datetime(wind_df["date"]).dt.date
    prod_df["tijd"] = pd.to_datetime(prod_df["tijd"], utc=True)
    prod_df["date"] = prod_df["tijd"].dt.date

    wind_df = wind_df[["date", "geo_windspeed_10m", "geo_windspeed_30m"]].dropna()
    df = prod_df.merge(wind_df, on="date", how="inner")
    df = df.dropna(subset=["vlaanderen wind kwh"])

    df["hour"] = df["tijd"].dt.hour
    df["day_of_week"] = df["tijd"].dt.dayofweek
    df["month"] = df["tijd"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["wind_speed_ratio"] = df["geo_windspeed_30m"] / (df["geo_windspeed_10m"] + 0.001)

    return df


@task(name="run-batch-predictions")
def run_batch_predictions(model, df: pd.DataFrame) -> pd.DataFrame:
    """Run predictions on the full dataset and return results DataFrame."""
    feature_cols = [
        "geo_windspeed_10m",
        "geo_windspeed_30m",
        "wind_speed_ratio",
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
    ]

    predictions = model.predict(df[feature_cols])
    predictions = np.maximum(predictions, 0)

    results = pd.DataFrame()
    results["tijd"] = df["tijd"].values
    results["geo_windspeed_10m"] = df["geo_windspeed_10m"].values
    results["geo_windspeed_30m"] = df["geo_windspeed_30m"].values
    results["actual_kwh"] = df["vlaanderen wind kwh"].values
    results["predicted_kwh"] = predictions
    results["error"] = results["actual_kwh"] - results["predicted_kwh"]
    results["predicted_at"] = datetime.utcnow()

    return results


@task(name="compute-metrics")
def compute_metrics(results: pd.DataFrame) -> dict:
    """Compute RMSE and MAE from batch results."""
    rmse = float(np.sqrt(np.mean(results["error"] ** 2)))
    mae = float(np.mean(np.abs(results["error"])))
    print(f"Batch RMSE: {rmse:.0f} kWh | MAE: {mae:.0f} kWh")
    return {"rmse": rmse, "mae": mae}


@task(name="save-predictions-parquet")
def save_predictions_parquet(results: pd.DataFrame):
    """Save predictions to parquet in the shared batch-data volume."""
    now = datetime.utcnow()
    path = os.path.join(BATCH_DATA_DIR, f"{now.year:04d}/{now.month:02d}")
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, f"{now.strftime('%Y%m%d_%H%M%S')}.parquet")
    results.to_parquet(filepath, index=False)
    print(f"Saved predictions to {filepath}")
    return filepath


@task(name="save-metrics-to-db")
def save_metrics_to_db(metrics: dict, run_id: str):
    """Save RMSE/MAE metrics to PostgreSQL for Grafana."""
    engine = get_engine()
    row = pd.DataFrame(
        [
            {
                "run_time": datetime.utcnow(),
                "run_id": run_id,
                "metric_name": "rmse",
                "value": str(metrics["rmse"]),
            },
            {
                "run_time": datetime.utcnow(),
                "run_id": run_id,
                "metric_name": "mae",
                "value": str(metrics["mae"]),
            },
        ]
    )
    row.to_sql("batch_metrics", engine, if_exists="append", index=False)
    print("Saved metrics to database")


@task(name="run-evidently-report")
def run_evidently_report(results: pd.DataFrame):
    """Generate Evidently drift report and save metrics to PostgreSQL."""
    feature_cols = ["geo_windspeed_10m", "geo_windspeed_30m", "predicted_kwh"]

    # Save current batch as reference if none exists yet
    if not os.path.exists(REFERENCE_PATH):
        os.makedirs(BATCH_DATA_DIR, exist_ok=True)
        results[feature_cols].to_parquet(REFERENCE_PATH, index=False)
        print("Saved reference dataset — skipping drift report for first run")
        return

    reference = pd.read_parquet(REFERENCE_PATH)[feature_cols]
    current = results[feature_cols]

    definition = DataDefinition()
    report = Report([DataDriftPreset()])
    run = report.run(
        Dataset.from_pandas(current, data_definition=definition),
        Dataset.from_pandas(reference, data_definition=definition),
    )

    # Save HTML report
    report_path = os.path.join(BATCH_DATA_DIR, "drift_report.html")
    run.save_html(report_path)
    print(f"Saved drift report to {report_path}")

    # Extract and save metrics to DB
    run_time = datetime.utcnow()
    json_data = run.dict()
    result_data = []
    for metric in json_data.get("metrics", []):
        metric_name = (
            metric.get("metric_id")
            or metric.get("id")
            or metric.get("name")
            or str(list(metric.keys()))
        )
        metric_value = metric.get("value") or metric.get("result") or ""
        result_data.append(
            {
                "run_time": run_time,
                "metric_name": str(metric_name),
                "value": str(metric_value),
            }
        )

    if result_data:
        engine = get_engine()
        pd.DataFrame(result_data).to_sql(
            "evidently_metrics", engine, if_exists="append", index=False
        )
        print(f"Saved {len(result_data)} Evidently metrics to database")

    # Update reference to current
    results[feature_cols].to_parquet(REFERENCE_PATH, index=False)


@task(name="check-retraining-trigger")
async def check_retraining_trigger(metrics: dict):
    """Trigger retraining flow if RMSE exceeds threshold."""
    if metrics["rmse"] > RMSE_THRESHOLD:
        print(
            f"RMSE {metrics['rmse']:.0f} exceeds threshold {RMSE_THRESHOLD:.0f} — triggering retraining"
        )
        # Log to DB as a retraining event
        engine = get_engine()
        pd.DataFrame(
            [
                {
                    "run_time": datetime.utcnow(),
                    "run_id": "retraining_trigger",
                    "metric_name": "retraining_triggered",
                    "value": str(metrics["rmse"]),
                }
            ]
        ).to_sql("batch_metrics", engine, if_exists="append", index=False)
        # Actually trigger the training deployment (timeout=0 = fire-and-forget)
        await run_deployment(
            "wind-production-training/wind-production-training", timeout=0
        )
        print("Retraining deployment triggered via Prefect")
    else:
        print(f"RMSE {metrics['rmse']:.0f} within threshold — no retraining needed")


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------


@flow(name="wind-batch-prediction")
def batch_flow():
    """Scheduled batch prediction, monitoring and retraining trigger."""
    model, run_id = load_latest_model()
    df = load_and_prepare_data(DATA_DIR)
    results = run_batch_predictions(model, df)
    metrics = compute_metrics(results)
    save_predictions_parquet(results)
    save_metrics_to_db(metrics, run_id)
    run_evidently_report(results)
    check_retraining_trigger(metrics)


if __name__ == "__main__":
    batch_flow.serve(
        name="wind-batch-daily",
        cron="0 6 * * *",  # runs every day at 6:00 AM
    )
