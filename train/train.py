import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from prefect import flow, task
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", "http://experiment-tracking:5000"
)
EXPERIMENT_NAME = "wind-production-forecasting"
MODEL_NAME = "wind-production-model"
DATA_DIR = os.getenv("DATA_DIR", "/data")
RMSE_THRESHOLD = float(os.getenv("RMSE_THRESHOLD", "150000"))


@task(name="load-and-join-data")
def load_and_join_data(data_dir: str) -> pd.DataFrame:
    wind_path = os.path.join(data_dir, "wind.csv")
    prod_path = os.path.join(data_dir, "production.csv")

    wind_df = pd.read_csv(wind_path, na_values=["NULL"])
    prod_df = pd.read_csv(prod_path)

    wind_df["date"] = pd.to_datetime(wind_df["date"]).dt.date
    prod_df["tijd"] = pd.to_datetime(prod_df["tijd"], utc=True)
    prod_df["date"] = prod_df["tijd"].dt.date

    wind_df = wind_df[["date", "geo_windspeed_10m", "geo_windspeed_30m"]].dropna()

    df = prod_df.merge(wind_df, on="date", how="inner")

    df = df.dropna(subset=["vlaanderen wind kwh"])

    return df


@task(name="engineer-features")
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["tijd"].dt.hour
    df["day_of_week"] = df["tijd"].dt.dayofweek
    df["month"] = df["tijd"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["wind_speed_ratio"] = df["geo_windspeed_30m"] / (df["geo_windspeed_10m"] + 0.001)

    return df


@task(name="split-data")
def split_data(df: pd.DataFrame):
    feature_cols = [
        "geo_windspeed_10m",
        "geo_windspeed_30m",
        "wind_speed_ratio",
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
    ]
    target_col = "vlaanderen wind kwh"

    x = df[feature_cols]
    y = df[target_col]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, shuffle=False
    )

    return x_train, x_test, y_train, y_test


@task(name="train-model")
def train_model(x_train, y_train, params: dict) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    return model


@task(name="evaluate-model")
def evaluate_model(model, x_test, y_test) -> dict:
    y_pred = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}


@task(name="register-best-model")
def register_best_model(run_id: str, metrics: dict):
    if metrics["rmse"] <= RMSE_THRESHOLD:
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
        print(f"Model registered: {MODEL_NAME} (RMSE={metrics['rmse']:.0f})")
    else:
        print(
            f"Model NOT registered: RMSE {metrics['rmse']:.0f} exceeds threshold {RMSE_THRESHOLD}"
        )


@flow(name="wind-production-training")
def train_flow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_and_join_data(DATA_DIR)
    df = engineer_features(df)
    x_train, x_test, y_train, y_test = split_data(df)

    print(f"Training on {len(x_train)} rows, testing on {len(x_test)} rows")

    param_grid = [
        {"n_estimators": 100, "max_depth": 10, "min_samples_split": 2},
        {"n_estimators": 200, "max_depth": 15, "min_samples_split": 2},
        {"n_estimators": 200, "max_depth": 20, "min_samples_split": 5},
        {"n_estimators": 300, "max_depth": None, "min_samples_split": 2},
    ]

    best_rmse = float("inf")
    best_run_id = None
    best_metrics = None

    for params in param_grid:
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.set_tag("model_type", "RandomForestRegressor")
            mlflow.set_tag("target", "vlaanderen wind kwh")

            model = train_model(x_train, y_train, params)
            metrics = evaluate_model(model, x_test, y_test)

            mlflow.log_metrics(metrics)

            signature = infer_signature(x_train, model.predict(x_train))
            mlflow.sklearn.log_model(model, "model", signature=signature)

            run_id = mlflow.active_run().info.run_id
            print(f"Run {run_id}: RMSE={metrics['rmse']:.0f}, R2={metrics['r2']:.3f}")

            if metrics["rmse"] < best_rmse:
                best_rmse = metrics["rmse"]
                best_run_id = run_id
                best_metrics = metrics

    print(f"\nBest run: {best_run_id} with RMSE={best_rmse:.0f}")
    register_best_model(best_run_id, best_metrics)


if __name__ == "__main__":
    train_flow()
    train_flow.serve(name="wind-production-training")
