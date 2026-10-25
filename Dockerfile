FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir ".[serve]"

# Train at build time so the image is self-contained.
# In a real setup the model would be pulled from a model registry instead.
RUN python -m energy_forecast.data && python -m energy_forecast.train

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "energy_forecast.api:app", "--host", "0.0.0.0", "--port", "8000"]
