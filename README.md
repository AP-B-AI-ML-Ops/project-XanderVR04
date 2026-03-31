# Renewable Energy Production Forecasting — Antwerp Region

## Dataset(s)

### Source Data
The dataset comes from the Data Engineering course and combines four sheets from multiple providers:

| Sheet | Sources | Key Features |
|---|---|---|
| `wind` | Open Meteo ECMWF, Geo.be, Kaggle (Uccle, Antwerpen) | Wind speed (km/h) per hour |
| `zon` | Open Meteo ECMWF, Geo.be, Kaggle (Uccle) | Solar radiation (W/m²) per hour |
| `productie` | Energie Vlaanderen, Elia | Solar & wind production (MW) per hour |
| `consumptie` | Energie Vlaanderen, Elia, Kaggle | Grid load & consumption (MW) per hour |

### How the Data Is Used
- **Training & validation**: Historical hourly records from the `wind`, `zon`, and `productie` sheets, joined on the `tijd` column. An 80/10/10 train/validation/test split is used, respecting time order.
- **Test data**: The held-out final 10% of the timeline, used only for final evaluation.
- **New / live data**: The ECMWF model provides multi-day-ahead hourly weather forecasts. The batch service fetches these live forecasts via the Open Meteo API to make real-world predictions — not just backtests.

---

## Project Explanation

### What Is Predicted?
This system predicts **solar and wind energy production (MW) for the Antwerp region over the next 24 hours**, one value per hour (24 output values).

### Inputs and Outputs

| | Description |
|---|---|
| **Inputs** | Hourly weather forecast features: wind speed (km/h), solar radiation (W/m²), and derived time features (hour of day, day of year, etc.) |
| **Outputs** | Predicted energy production in MW for each of the next 24 hours (solar and/or wind) |

### Why Is This Useful?
The renewable energy transition introduces grid instability because solar and wind production are weather-dependent. Reliable short-term forecasts allow:
- **Grid operators** to balance supply and demand in real time.
- **Energy traders** to make accurate bidding decisions on day-ahead markets.
- **Prosumers** to optimise their own consumption and storage.

### Application Architecture
The system exposes the model in two ways:

1. **Web Service**: a REST API that accepts weather forecast data and immediately returns predicted production for the next 24 hours.
2. **Batch Service**: a Prefect pipeline that periodically fetches live ECMWF forecasts, runs inference, stores predictions, and compares them against Elia actuals to monitor model performance.

---

## Flows & Actions

### 1. Data Preparation Flow
- **Action**: Join `wind`, `zon`, and `productie` sheets on `tijd`.
- **Action**: Clean and engineer features (lag features, time encodings, rolling means).
- **Action**: Split into train / validation / test sets (time-ordered).
- **Output**: Versioned dataset artifact.

### 2. Model Training & Experiment Tracking Flow
- **Action**: Train a regression model on the training set.
- **Action**: Log hyperparameters, metrics (RMSE, MAE), and artifacts to **MLflow**.
- **Action**: Register the best model in the **MLflow Model Registry**.
- **Output**: Registered, versioned model ready for deployment.

### 3. Web Service Deployment
- **Action**: Serve the registered MLflow model via a **FastAPI** REST endpoint.
- **Endpoint**: `POST /predict` accepts hourly weather forecast JSON for the next 24 hours, returns predicted MW values.
- **Containerised**: Packaged as a Docker image.

### 4. Batch Inference & Monitoring Flow
- **Action**: Fetch live ECMWF weather forecasts from Open Meteo API.
- **Action**: Run inference using the production model.
- **Action**: Store predictions to a database / file store.
- **Action**: When Elia actuals become available, compute error metrics (RMSE, MAE).
- **Action**: Send metrics to **Evidently** for drift/performance reporting.
- **Action**: Visualise metrics in **Grafana** dashboards.
- **Action**: If RMSE exceeds a defined threshold -> automatically trigger the **Model Training Flow**.
- **Schedule**: Runs on a configurable cron schedule.

### 5. Retraining Flow
- **Action**: Pull the latest available data (including recent actuals).
- **Action**: Retrain the model and log the new experiment to MLflow.
- **Action**: If the new model outperforms the current production model -> promote it in the registry.
- **Action**: Redeploy the web service with the updated model.

---

## Tech Stack

| Concern | Tool |
|---|---|
| Experiment tracking & model registry | MLflow |
| Workflow orchestration | Prefect |
| Monitoring | Evidently, Grafana |
| Web service | FastAPI |
| Containerisation | Docker / Docker Compose |
| Data & model versioning | DVC (or MLflow artifacts) |

---

## How to Run

> Instructions will be completed as the project is built out.

```bash
# 1. Clone the repo
git clone <repo-url>
cd project-XanderVR04

# 2. Start all services
docker compose up --build

# 3. Access services
#   MLflow UI:   http://localhost:5000
#   Prefect UI:  http://localhost:4200
#   Grafana:     http://localhost:3000
#   API docs:    http://localhost:8000/docs
```
