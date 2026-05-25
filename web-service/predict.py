import os

import mlflow.sklearn
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", "http://experiment-tracking:5000"
)
MODEL_NAME = os.getenv("MODEL_NAME", "wind-production-model")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = mlflow.MlflowClient()
versions = client.search_model_versions(
    f"name='{MODEL_NAME}'", order_by=["version_number DESC"], max_results=1
)
latest = versions[0]
MODEL_VERSION = latest.version
model = mlflow.sklearn.load_model(f"runs:/{latest.run_id}/model")


def build_features(
    geo_windspeed_10m: float,
    geo_windspeed_30m: float,
    day_of_week: int = 0,
    month: int = 1,
    is_weekend: int = 0,
) -> pd.DataFrame:
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
    return jsonify({"status": "ok", "model": MODEL_NAME, "version": MODEL_VERSION})


@app.route("/predict", methods=["POST"])
def predict():
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
