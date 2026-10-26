.PHONY: install data data-real train train-real tune train-best test lint serve monitor mlflow-ui docker-build docker-run clean

install:
	pip install -e ".[dev]"

data:
	python -m energy_forecast.data

# Download and prepare the real Building Data Genome 2 subset (~200 MB, once)
data-real:
	python -m energy_forecast.data --source bdg2

train:
	python -m energy_forecast.train

train-real:
	python -m energy_forecast.train --source bdg2

# Hyperparameter search with Optuna (add SOURCE=bdg2 to tune on real data)
tune:
	python -m energy_forecast.tune $(if $(SOURCE),--source $(SOURCE),)

train-best:
	python -m energy_forecast.train --params reports/best_params.yaml $(if $(SOURCE),--source $(SOURCE),)

test:
	pytest -v

lint:
	ruff check src tests

serve:
	uvicorn energy_forecast.api:app --reload --port 8000

# Simulate new production data with a +6°C shift and check for drift
monitor:
	python -m energy_forecast.data --start 2025-01-01 --temperature-shift 6 --output data/raw/new_data.csv
	python -m energy_forecast.monitor data/raw/new_data.csv

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

docker-build:
	docker build -t energy-forecast-api .

docker-run:
	docker run --rm -p 8000:8000 energy-forecast-api

# Keeps the downloaded BDG2 files (data/external) to avoid downloading again
clean:
	rm -rf data/raw data/processed models reports mlruns mlartifacts mlflow.db .pytest_cache
