.PHONY: install data train test lint serve monitor docker-build docker-run mlflow-ui clean

install:
	pip install -e ".[dev]"

data:
	python -m energy_forecast.data

train:
	python -m energy_forecast.train

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

clean:
	rm -rf data models reports mlruns mlartifacts mlflow.db .pytest_cache
