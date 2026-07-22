# Reproducible pipeline for "Robust and Explainable Grid-Level Urban Mobility Demand Forecasting".
# Data (processed CSVs) is committed, so every target below runs without external downloads.
# Heavy model training (ST-HAE) is GPU-friendly and separated out; see `train-*` targets.

PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
DATA_CHI = data/processed/chicago_taxi_sides.csv
DATA_NYC = data/processed/nyc_taxi_boroughs.csv

.PHONY: help setup test robustness faithfulness figures reproduce train-sthae train-adjacency clean

help:
	@echo "Targets:"
	@echo "  setup        create .venv (Python 3.11) and install requirements-core.txt"
	@echo "  test         run the pytest suite"
	@echo "  robustness   bootstrap CIs + conformal coverage on both cities -> results/"
	@echo "  faithfulness LLM faithfulness eval (mock; set GROQ_API_KEY + PROVIDERS=groq for real)"
	@echo "  figures      regenerate paper/figures/*.png from results/*.json"
	@echo "  reproduce    robustness + figures + test (the fast, no-GPU reproduction)"
	@echo "  train-sthae  ST-HAE ablation + ST-GNN baselines (GPU recommended; --device auto)"
	@echo "  train-adjacency  spatial-rescue adjacency variants, 5 seeds (coarse grids)"

setup:
	python3.11 -m venv .venv
	$(PIP) install -U pip
	$(PIP) install -r requirements-core.txt

test:
	$(PY) -m pytest -q

robustness:
	$(PY) src/robustness_ci.py --data $(DATA_CHI) --out results/chicago_sides_robustness_ci.json
	$(PY) src/robustness_ci.py --data $(DATA_NYC) --out results/nyc_robustness_ci.json

PROVIDERS ?= mock
faithfulness:
	$(PY) src/llm_faithfulness.py --data $(DATA_CHI) --providers $(PROVIDERS) --repeats 5 --out results/faithfulness_chicago.json
	$(PY) src/llm_faithfulness.py --data $(DATA_NYC) --providers $(PROVIDERS) --repeats 5 --out results/faithfulness_nyc.json

figures:
	$(PY) src/make_figures.py

reproduce: test robustness figures
	@echo "Reproduced robustness results + figures (no GPU needed)."

train-sthae:
	$(PY) src/st_hae.py --data $(DATA_CHI) --ablation --baselines --device auto --out results/st_hae_chicago.json
	$(PY) src/st_hae.py --data $(DATA_NYC) --ablation --baselines --device auto --out results/st_hae_nyc.json

train-adjacency:
	$(PY) src/st_hae.py --data $(DATA_CHI) --adjacency --seeds 5 --device auto --out results/st_hae_chicago_adj.json
	$(PY) src/st_hae.py --data $(DATA_NYC) --adjacency --seeds 5 --device auto --out results/st_hae_nyc_adj.json

clean:
	rm -rf .pytest_cache src/__pycache__ tests/__pycache__
