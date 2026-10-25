# Energy Consumption Forecasting — End-to-End MLOps Pipeline

![CI](https://github.com/<your-username>/energy-forecasting-mlops/actions/workflows/ci.yml/badge.svg)

Forecasting hourly electricity consumption of buildings (offices, residential, retail), taken from experimentation to a production-style setup: reproducible training, experiment tracking, a containerised REST API, automated testing and deployment checks, and data drift monitoring.

## Architecture

```
 ┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐
 │ Data         │ →  │ Feature         │ →  │ Training         │ → MLflow (params,
 │ generation   │    │ engineering     │    │ + baseline       │    metrics, model)
 └──────────────┘    │ (lags, calendar)│    │ comparison       │
                     └─────────────────┘    └────────┬─────────┘
                                                     │ model.joblib
                    ┌──────────────────┐    ┌────────▼─────────┐
 New data ────────→ │ Drift monitoring │    │ FastAPI service  │ ← Docker
                    │ (PSI + KS test)  │    │ /predict /health │
                    └──────────────────┘    └──────────────────┘

 GitHub Actions: lint → tests → training + quality gate → Docker build + smoke test
                 weekly scheduled drift check
```

## Results

Evaluated on the last 30 days, held out chronologically (no shuffling, to avoid leakage in time series):

| Model | MAE (kWh) | RMSE (kWh) | MAPE |
|---|---|---|---|
| Seasonal naive baseline (same hour last week) | 5.34 | 7.07 | 9.7 % |
| Gradient boosting (HistGradientBoosting) | **4.51** | **5.70** | **9.3 %** |

The model reduces MAE by about **15 %** over the baseline. The CI pipeline enforces this: a model that does not beat the baseline fails the build.

## Key features

- **Leak-free feature engineering**: per-building lags (24h, 168h) and a rolling mean computed only from past values, plus cyclical calendar encodings.
- **Baseline-first evaluation**: every model is compared against a seasonal naive forecast, which is the honest benchmark for hourly energy data.
- **Experiment tracking with MLflow**: hyperparameters, metrics and the model artifact (with input example) are logged for every run.
- **Serving with FastAPI**: typed request validation with Pydantic, batch predictions, health and model-info endpoints.
- **Containerisation**: self-contained Docker image running as a non-root user, with a health check.
- **CI/CD with GitHub Actions**: linting, unit tests on two Python versions, training with a quality gate, Docker build and an end-to-end smoke test of the API.
- **Drift monitoring**: Population Stability Index and Kolmogorov–Smirnov test per feature against the training distribution, runnable on a weekly schedule, failing the job when drift is detected.

## Data

The dataset is synthetic but physically motivated, so the project runs anywhere without downloads. Consumption depends on building type and size, occupancy schedules (weekday/weekend), and heating and cooling demand driven by outdoor temperature with yearly and daily cycles. A `temperature_shift` option simulates a climate change scenario to demonstrate drift detection.

To use real data instead (for example the ASHRAE Great Energy Predictor dataset), provide a CSV with the columns `timestamp, building_id, building_type, temperature, consumption_kwh` at `data/raw/energy.csv`.

## Quickstart

```bash
git clone https://github.com/<your-username>/energy-forecasting-mlops.git
cd energy-forecasting-mlops
python -m venv .venv && source .venv/bin/activate
make install      # installs the package with dev, serving and tracking extras

make data         # generate the dataset
make train        # train, evaluate against the baseline, log to MLflow
make mlflow-ui    # browse experiments at http://localhost:5000
make test         # run the test suite
make serve        # start the API at http://localhost:8000/docs
make monitor      # simulate drifted production data and run the drift check
```

### With Docker

```bash
docker compose up --build
# API:    http://localhost:8000/docs
# MLflow: http://localhost:5000

# To log training runs to the MLflow server:
MLFLOW_TRACKING_URI=http://localhost:5000 make train
```

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '[{"timestamp": "2024-12-02T09:00:00", "building_type": "office",
        "temperature": 6.5, "lag_24": 85.0, "lag_168": 88.0, "rolling_mean_24": 52.0}]'
```

```json
[{"timestamp": "2024-12-02T09:00:00", "building_type": "office", "predicted_consumption_kwh": 92.8}]
```

## Project structure

```
├── configs/config.yaml          # all parameters in one place
├── src/energy_forecast/
│   ├── data.py                  # synthetic dataset generation
│   ├── features.py              # feature engineering and time-based split
│   ├── train.py                 # training, evaluation, MLflow tracking
│   ├── monitor.py               # data drift detection (PSI, KS test)
│   └── api.py                   # FastAPI serving
├── tests/                       # unit and API tests (pytest)
├── .github/workflows/           # CI pipeline and scheduled drift check
├── Dockerfile
├── docker-compose.yml           # API + MLflow tracking server
└── Makefile
```

## Possible next steps

- Register models in the MLflow Model Registry and have the API load the "Production" version
- Deploy the container to AWS (ECS Fargate or App Runner) with the image pushed to ECR from CI
- Automatic retraining triggered when drift is detected
- Probabilistic forecasts (quantile regression) to give prediction intervals
