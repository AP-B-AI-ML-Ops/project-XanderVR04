# Wind Production Forecasting

This project predicts hourly wind energy production (in kWh) for Flanders over the next 24 hours using weather forecast data as input. It is built as a full end-to-end MLOps system covering training, deployment, scheduling, and monitoring.

**What is predicted:** total wind energy production per hour for Flanders (`vlaanderen wind kwh`).

**Inputs:** daily wind speed at 10m and 30m height, combined with time-based features (hour of day, day of week, month, weekend flag, wind speed ratio).

**Why this is useful:** grid operators and energy traders need short-term wind forecasts to make balancing decisions, such as when to activate backup capacity or trade excess energy.

---

## Project Components

### Training (`train/`)
Trains a Random Forest regressor across 4 hyperparameter configurations. Each run is tracked in MLflow with metrics and parameters logged. The best model is registered in the MLflow model registry. After the initial training run, the container stays alive as a Prefect deployment so it can be triggered automatically for retraining.

### Web Service (`web-service/`)
A Flask REST API running on port 9696. Accepts wind speed and date features as input and returns 24 hourly wind production predictions in kWh. The model is loaded from the MLflow registry on startup.

### Batch Service (`batch-service/`)
A Prefect flow scheduled to run daily at 06:00 UTC. On each run it loads the latest registered model, runs predictions over the full dataset, computes RMSE and MAE, saves results to PostgreSQL, generates an Evidently data drift report, and triggers retraining if RMSE exceeds 400,000 kWh.

### Experiment Tracking (`backend-service-experiment-tracking/`)
An MLflow server backed by PostgreSQL. Stores all training runs, metrics, parameters, and registered model versions.

### Orchestration (`backend-service-orchestration/`)
A Prefect server that manages and schedules the batch flow and retraining deployment.

### Monitoring (`grafana/`)
A Grafana dashboard connected to PostgreSQL. Displays RMSE and MAE over time, latest metric values, predicted vs actual production, and prediction error percentage per day.

---

## Services

| Service | URL | Credentials |
|---|---|---|
| MLflow | http://localhost:5000 | none |
| Prefect | http://localhost:4200 | none |
| Grafana | http://localhost:3400 | admin / admin |
| Web API | http://localhost:9696 | none |

---

## How to Run

### Prerequisites

- Docker Desktop (running)
- Git

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd project-xandervr04
```

### 2. Configure environment

```bash
cp backend-database/.env.example backend-database/.env
cp .env.example .env
```

Open both `.env` files and set `POSTGRES_USER` and `POSTGRES_PASSWORD` to the same values in both. Use only letters and numbers in the password.

### 3. Add data files

Place the following files in the `data/` folder:

- `wind.csv`
- `production.csv`

### 4. Start the infrastructure

```bash
docker compose up -d --build database experiment-tracking orchestration grafana
```

Wait until Prefect is ready:

```bash
docker compose logs orchestration --tail=20
```

Continue when you see `Check out the dashboard at http://0.0.0.0:4200`.

### 5. Train the model

```bash
docker compose up -d --build --force-recreate --no-deps train
```

Follow the logs to know when training is done:

```bash
docker compose logs train --follow
```

Continue when you see `Model registered: wind-production-model`. Press `Ctrl+C` to stop following.

### 6. Start the web service and batch service

```bash
docker compose up -d --build --force-recreate --no-deps web-service batch-service
```

Verify both started correctly:

```bash
docker compose logs web-service --tail=20
docker compose logs batch-service --tail=20
```

The web service is ready when you see `Running on http://0.0.0.0:9696`. The batch service is ready when you see `Worker started`.

### 7. Trigger the first batch run

Open http://localhost:4200 and go to **Deployments** -> `wind-batch-daily` -> **Run** -> **Quick Run**.

After triggering, refresh the Prefect page after about 30 seconds. The run will appear with a green status and marked as "Completed" when it finished successfully.

You can also verify via logs:

```bash
docker compose logs batch-service --tail=30
```

You should see `RMSE ... within threshold` or `Retraining deployment triggered`.

### 8. Test the web API

**Windows:**
```
curl -X POST http://localhost:9696/predict -H "Content-Type: application/json" -d "{\"geo_windspeed_10m\": 5.2, \"geo_windspeed_30m\": 7.1, \"month\": 3, \"day_of_week\": 0, \"is_weekend\": 0}"
```

**Linux / Mac:**
```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"geo_windspeed_10m": 5.2, "geo_windspeed_30m": 7.1, "month": 3, "day_of_week": 0, "is_weekend": 0}'
```

### 9. View the monitoring dashboard

Open Grafana at http://localhost:3400, log in with `admin` / `admin`, and open the **Wind Production Monitoring** dashboard. You can click and drag on any time series panel to zoom into a specific time range, and double-click to zoom back out.

---

## Pre-commit Hooks

Install locally to run linting and tests before each commit:

```bash
pip install pre-commit
pre-commit install
```
