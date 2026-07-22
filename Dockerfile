# Reproducible environment for the forecasting + robustness + ST-HAE + faithfulness pipeline.
# Processed datasets are committed, so `docker run <img> make reproduce` needs no downloads.
FROM python:3.11-slim

WORKDIR /app

# CPU-only torch keeps the image small; the pipeline auto-detects CUDA when present.
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1

COPY requirements-core.txt .
RUN pip install -U pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu torch==2.1.0 && \
    pip install -r requirements-core.txt

COPY . .

# Make the repo's `make` targets use the system interpreter (no .venv inside the image).
ENV PY=python

# Default: run the test suite. Override with e.g. `docker run <img> make reproduce`.
CMD ["python", "-m", "pytest", "-q"]
