"""Wind production forecast REST API."""

# pylint: disable=import-error

import os

import mlflow.sklearn
import mlflow.tracking
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", "http://experiment-tracking:5000"
)
MODEL_NAME = os.getenv("MODEL_NAME", "wind-production-model")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Load model via run ID so artifact is fetched through the tracking server HTTP API
client = mlflow.tracking.MlflowClient()
model_version_info = client.get_model_version(MODEL_NAME, MODEL_VERSION)
run_id = model_version_info.run_id
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")


def build_features(
    geo_windspeed_10m: float,
    geo_windspeed_30m: float,
    day_of_week: int = 0,
    month: int = 1,
    is_weekend: int = 0,
) -> pd.DataFrame:
    """Build a 24-row feature DataFrame for one day of hourly predictions."""
    rows = []
    for hour in range(24):
        rows.append(
            {
                "geo_windspeed_10m": geo_windspeed_10m,
                "geo_windspeed_30m": geo_windspeed_30m,
                "wind_speed_ratio": geo_windspeed_30m / (geo_windspeed_10m + 0.001),
                "hour": hour,
                "day_of_week": day_of_week,
                "month": month,
                "is_weekend": is_weekend,
            }
        )
    return pd.DataFrame(rows)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "model": MODEL_NAME, "version": MODEL_VERSION})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict wind production for the next 24 hours.

    Expected JSON body:
    {
        "geo_windspeed_10m": 5.2,
        "geo_windspeed_30m": 7.1,
        "day_of_week": 0,    (optional, 0=Monday ... 6=Sunday)
        "month": 3,          (optional, 1-12)
        "is_weekend": 0      (optional, 0 or 1)
    }

    Returns:
    {
        "predictions_kwh": [float, ...],  (24 values, one per hour)
        "total_kwh": float
    }
    """
    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    required = ["geo_windspeed_10m", "geo_windspeed_30m"]
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    features = build_features(
        geo_windspeed_10m=float(data["geo_windspeed_10m"]),
        geo_windspeed_30m=float(data["geo_windspeed_30m"]),
        day_of_week=int(data.get("day_of_week", 0)),
        month=int(data.get("month", 1)),
        is_weekend=int(data.get("is_weekend", 0)),
    )

    predictions = model.predict(features).tolist()
    predictions = [max(0.0, p) for p in predictions]

    return jsonify(
        {
            "predictions_kwh": predictions,
            "total_kwh": sum(predictions),
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696)
