# Energy Consumption Forecasting — End-to-End MLOps Pipeline

![CI](https://github.com/b-k-f/energy-forecasting-mlops/actions/workflows/ci.yml/badge.svg)

Forecasting hourly electricity consumption of buildings, on real meter data (BDG2) or synthetic data, taken from experimentation to a production-style setup: reproducible training, experiment tracking, a containerised REST API, automated testing and deployment checks, and data drift monitoring.

## What's inside

- **Real and synthetic data**: trains on the [Building Data Genome Project 2](https://github.com/buds-lab/building-data-genome-project-2) (real hourly meters from office, education and lodging buildings, with site weather), or on synthetic data for fast tests and CI.
- **Leak-free feature engineering**: per-building lags (24h, 168h), a rolling mean of past values, heating and cooling degree-hours, and cyclical calendar encodings.
- **Rigorous evaluation**: rolling-origin backtesting over several months, always compared to a seasonal naive baseline (same hour last week).
- **Hyperparameter tuning with Optuna**: optimises the mean backtest error, with every trial logged as a nested MLflow run.
- **Prediction intervals**: quantile models return an 80% interval around each forecast, with measured coverage.
- **Explainability**: permutation feature importance, saved as a chart with each run.
- **Experiment tracking with MLflow**: parameters, metrics, reports and the model artifact for every run.
- **Serving with FastAPI**: typed validation, batch predictions with intervals, health and model-info endpoints, in a non-root Docker container.
- **CI/CD with GitHub Actions**: lint, tests on two Python versions, training with a quality gate, Docker build with an API smoke test; plus a manual workflow that trains on the real dataset.
- **Drift monitoring**: Population Stability Index and Kolmogorov–Smirnov test per feature, on a weekly schedule.

## Architecture
cat > README.md <<'EOF'
# Energy Consumption Forecasting — End-to-End MLOps Pipeline

![CI](https://github.com/b-k-f/energy-forecasting-mlops/actions/workflows/ci.yml/badge.svg)

Forecasting hourly electricity consumption of buildings, on real meter data (BDG2) or synthetic data, taken from experimentation to a production-style setup: reproducible training, experiment tracking, a containerised REST API, automated testing and deployment checks, and data drift monitoring.

## What's inside

- **Real and synthetic data**: trains on the [Building Data Genome Project 2](https://github.com/buds-lab/building-data-genome-project-2) (real hourly meters from office, education and lodging buildings, with site weather), or on synthetic data for fast tests and CI.
- **Leak-free feature engineering**: per-building lags (24h, 168h), a rolling mean of past values, heating and cooling degree-hours, and cyclical calendar encodings.
- **Rigorous evaluation**: rolling-origin backtesting over several months, always compared to a seasonal naive baseline (same hour last week).
- **Hyperparameter tuning with Optuna**: optimises the mean backtest error, with every trial logged as a nested MLflow run.
- **Prediction intervals**: quantile models return an 80% interval around each forecast, with measured coverage.
- **Explainability**: permutation feature importance, saved as a chart with each run.
- **Experiment tracking with MLflow**: parameters, metrics, reports and the model artifact for every run.
- **Serving with FastAPI**: typed validation, batch predictions with intervals, health and model-info endpoints, in a non-root Docker container.
- **CI/CD with GitHub Actions**: lint, tests on two Python versions, training with a quality gate, Docker build with an API smoke test; plus a manual workflow that trains on the real dataset.
- **Drift monitoring**: Population Stability Index and Kolmogorov–Smirnov test per feature, on a weekly schedule.

## Architecture

~~~
 Synthetic data ─┐
                 ├→ Features → Backtesting ─→ Final model + quantile models → MLflow
 BDG2 (real) ────┘  (lags,     (4 monthly     (vs seasonal baseline,          (params, metrics,
                    degree-     windows)       80% intervals, importance)      reports, model)
                    hours)          ↑                     │
                              Optuna tuning               ▼
                                                  FastAPI service ← Docker
 New data → Drift monitoring (PSI + KS)           /predict /health /model-info

 GitHub Actions: lint → tests → training + quality gate → Docker build + smoke test
                 weekly drift check · manual training on real data
~~~

## Results (synthetic data)

| Metric | Seasonal baseline | Model |
|---|---|---|
| MAE on last 30 days (kWh) | 5.34 | **3.74** |
| Mean MAE over 4 monthly backtest folds (kWh) | 4.63 | **3.39 ± 0.58** |
| 80% interval coverage (target 0.80) | | **0.79** |

The model reduces MAE by about **30 %** over the baseline, and the prediction intervals are well calibrated. The CI pipeline blocks any model that does not beat the baseline.

On real BDG2 data, run `make data-real && make train-real` (or the *Train on real data* workflow) and compare the runs in MLflow; real meter data is noisier, so expect larger errors than on synthetic data.

## Data

**Real data (BDG2).** `make data-real` downloads the cleaned electricity meters, weather and metadata from the BDG2 repository (about 200 MB, once), selects up to 10 office, education and lodging buildings with fewer than 5 % missing readings, and joins each building with its site's air temperature. Missing readings stay as gaps on the hourly grid so lag features remain correct, then those rows are excluded from training. Data citation: Miller, C. et al., *The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III competition*, Scientific Data 7, 368 (2020).

**Synthetic data.** Physically motivated, so tests and CI run anywhere in seconds without downloads: consumption depends on building type and size, occupancy schedules, and heating and cooling demand driven by temperature. A `temperature_shift` option simulates a warmer climate to demonstrate drift detection.

## Quickstart

~~~bash
git clone https://github.com/b-k-f/energy-forecasting-mlops.git
cd energy-forecasting-mlops
python -m venv .venv && source .venv/bin/activate
make install      # installs the package with dev, serving, tracking and tuning extras

make data         # generate the synthetic dataset
make train        # backtest, train, intervals, importance, log to MLflow
make data-real    # download and prepare the real BDG2 subset (once)
make train-real   # same training on real data
make tune         # Optuna search (SOURCE=bdg2 for real data), then: make train-best
make mlflow-ui    # browse experiments at http://localhost:5000
make test         # run the test suite
make serve        # start the API at http://localhost:8000/docs
make monitor      # simulate drifted production data and run the drift check
~~~

### With Docker

~~~bash
docker compose up --build
# API:    http://localhost:8000/docs
# MLflow: http://localhost:5000

# To log training runs to the MLflow server:
MLFLOW_TRACKING_URI=http://localhost:5000 make train
~~~

### Example request

~~~bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '[{"timestamp": "2024-12-02T09:00:00", "building_type": "office",
        "temperature": 6.5, "lag_24": 85.0, "lag_168": 88.0, "rolling_mean_24": 52.0}]'
~~~

~~~json
[{"timestamp": "2024-12-02T09:00:00", "building_type": "office",
  "predicted_consumption_kwh": 91.28, "lower_kwh": 80.746, "upper_kwh": 99.166}]
~~~

## Project structure

~~~
├── configs/config.yaml          # all parameters in one place
├── src/energy_forecast/
│   ├── data.py                  # synthetic data and data source selection
│   ├── bdg2.py                  # real data: download and prepare BDG2
│   ├── features.py              # features, time split, backtest windows
│   ├── train.py                 # backtesting, training, intervals, importance, MLflow
│   ├── tune.py                  # Optuna hyperparameter search
│   ├── monitor.py               # data drift detection (PSI, KS test)
│   └── api.py                   # FastAPI serving
├── tests/                       # unit and API tests (pytest)
├── .github/workflows/           # CI, weekly drift check, real-data training
├── Dockerfile
├── docker-compose.yml           # API + MLflow tracking server
└── Makefile
~~~

## Possible next steps

- Register models in the MLflow Model Registry and have the API load the "champion" version
- Deploy the container to AWS (ECS Fargate or App Runner) with the image pushed to ECR from CI
- Monitor real accuracy once actual consumption is known (concept drift), and retrain automatically
- Add public holidays per site and compare with a deep learning model (LSTM)
