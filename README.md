# Wind Production Forecasting — MLOps Project

## Dataset(s)

This project uses two datasets produced by the **Data Engineering** course pipelines:

| File | Source | Description |
|---|---|---|
| `data/wind.csv` | Open Meteo ECMWF, Geo.be | Daily wind speed (km/h) at 10m and 30m height for the Antwerp region |
| `data/production.csv` | Energie Vlaanderen, Elia | Hourly solar and wind energy production (kWh) for Flanders |

The two datasets are joined on date. Wind speed features (`geo_windspeed_10m`, `geo_windspeed_30m`) are forward-filled from daily to hourly granularity. The joined dataset covers **March 2025 to March 2026** (~9,000 hourly rows) — the period with the best data availability across all columns.

**Training data:** 80% of the joined dataset (chronological split, ~7,300 rows)
**Test data:** 20% of the joined dataset (~1,800 rows)
**New data for inference:** the same wind CSV is used by the batch service to simulate scheduled inference. In production, this would be replaced by live ECMWF forecast data from the Open Meteo API.

---

## Project Explanation

This project builds an **end-to-end MLOps system** that predicts hourly wind energy production (in kWh) for Flanders over the next 24 hours, using weather forecast data as input.

**What is predicted:** `vlaanderen wind kwh` — the total wind energy production in Flanders per hour.

**Inputs:** daily wind speed at 10m and 30m height, enriched with time-based features (hour of day, day of week, month, weekend flag, wind speed ratio).

**Output:** 24 hourly wind production predictions in kWh, plus a daily total.

**Why this is useful:** grid operators and energy traders need reliable short-term wind forecasts to make balancing decisions. By knowing how much wind energy will be produced in the next 24 hours, they can plan when to activate backup capacity or sell excess energy.

The system uses a **Random Forest regressor** trained with MLFlow experiment tracking and hyperparameter search across 4 configurations. The best model is registered in the MLFlow model registry and served via two deployment modes:

- **Web service:** a REST API that accepts weather forecast data and returns 24-hour predictions on demand
- **Batch service:** a scheduled Prefect pipeline that runs daily, generates predictions, computes error metrics, and monitors for model drift using Evidently

---

## Flows & Actions

### 1. Training Flow (`train/train.py`)
**Trigger:** manual (`docker compose up --force-recreate --no-deps train`)
**Steps:**
1. Load and join `wind.csv` and `production.csv` on date
2. Engineer features (hour, day of week, month, weekend flag, wind speed ratio)
3. Split data 80/20 (chronological)
4. Train 4 Random Forest configurations, tracking each run in MLFlow
5. Register the best model in the MLFlow model registry

### 2. Web Service (`web-service/predict.py`)
**Trigger:** HTTP POST request to `http://localhost:9696/predict`
**Input:**
```json
{
    "geo_windspeed_10m": 5.2,
    "geo_windspeed_30m": 7.1,
    "day_of_week": 0,
    "month": 3,
    "is_weekend": 0
}
```
**Output:**
```json
{
    "predictions_kwh": [578436.17, "..."],
    "total_kwh": 15375825.69
}
```

### 3. Batch Flow (`batch-service/batch_flow.py`)
**Trigger:** Prefect schedule (daily at 06:00) or manual via Prefect UI
**Steps:**
1. Load the latest registered model from MLFlow
2. Load wind and production data, engineer features
3. Run predictions across the full dataset
4. Compute RMSE and MAE against actual production values
5. Save predictions as parquet to `/batch-data/`
6. Save metrics to PostgreSQL `metrics.batch_metrics` table
7. Generate Evidently drift report (HTML + metrics to `metrics.evidently_metrics`)
8. Trigger retraining flag in DB if RMSE exceeds 400,000 kWh threshold

---

## Services

| Service | URL | Description |
|---|---|---|
| MLFlow | http://localhost:5000 | Experiment tracking and model registry |
| Prefect | http://localhost:4200 | Workflow orchestration and scheduling |
| Grafana | http://localhost:3400 | Monitoring dashboard (login: admin/admin) |
| Web API | http://localhost:9696 | On-demand forecast REST API |

---

## How to Run

### Prerequisites
- Docker Desktop
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
Edit both `.env` files and set `POSTGRES_USER` and `POSTGRES_PASSWORD` to the same values in both files. Use only letters and numbers in the password (no special characters).

### 3. Add data files
Place the following CSV files in the `data/` folder:
- `wind.csv`
- `production.csv`

### 4. Start core infrastructure
```bash
docker compose up -d --build database experiment-tracking orchestration grafana
```
Wait ~30 seconds for all services to be ready before continuing.

### 5. Train the model
```bash
docker compose up --force-recreate --no-deps train
```
Wait for `train-1 exited with code 0` before continuing.

### 6. Start the web service and batch service
```bash
docker compose up -d --force-recreate --no-deps web-service
docker compose up -d --force-recreate --no-deps batch-service
```

### 7. Trigger the first batch run
Go to http://localhost:4200/dashboard → **Deployments** → `wind-batch-daily` → **Run** → **Quick Run**.

### 8. Test the web API

**Windows (cmd):**
```
curl -X POST http://localhost:9696/predict -H "Content-Type: application/json" -d "{\"geo_windspeed_10m\": 5.2, \"geo_windspeed_30m\": 7.1, \"month\": 3, \"day_of_week\": 0, \"is_weekend\": 0}"
```

**Linux/Mac:**
```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"geo_windspeed_10m": 5.2, "geo_windspeed_30m": 7.1, "month": 3, "day_of_week": 0, "is_weekend": 0}'
```

### 9. View monitoring dashboard
Open Grafana at http://localhost:3400 and navigate to the **Wind Production Monitoring** dashboard.

---

## Dependencies

All dependencies are pinned inside each service's `requirements.txt`. See individual service folders for details.

Install pre-commit hooks locally:
```bash
pip install pre-commit pylint black isort pytest pandas
python -m pre_commit install
```
