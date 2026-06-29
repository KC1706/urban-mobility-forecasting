# Robust & Explainable Urban Mobility Demand Forecasting

End-to-end pipeline for grid-level urban mobility (taxi demand) forecasting whose focus is
**operational robustness** rather than headline accuracy: models that look near-perfect on a
single global metric can fail badly exactly when reliable predictions matter most — peak
hours, high-demand spikes, and specific zones. This project surfaces those failures with a
robustness stress-test framework and explains them in natural language using an LLM layer.

## Key finding

A Random Forest achieves **R² = 0.994** globally — yet under stress testing:

| Stress dimension | Behavior |
|---|---|
| Per-zone (e.g. Downtown) | R² goes **strongly negative** (worse than predicting the mean) |
| Morning rush (7–9 AM) | RMSE ≈ **2×** the best off-peak hour |
| High-demand periods | Error **+106%** vs normal demand |

The lesson: aggregate R²/RMSE hide operationally critical failure modes. The robustness +
explainability layers are the contribution.

## Pipeline

```
Raw taxi trips (Chicago/NYC TLC open data)
  → grid-level spatial + hourly temporal aggregation   (src/data_processor.py)
  → baseline models: Random Forest, XGBoost, LSTM       (src/baseline_models.py)
  → robustness / tail-error stress testing              (src/robustness_eval.py, evaluator.py)
  → LLM explainability of failures (GPT-4 / Claude)      (src/llm_interpreter.py, llm_analyzer.py)
  → orchestration + CLI                                  (src/experiment_runner.py, run_pipeline.py)
```

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env          # add your own LLM key(s) for Phase 3

# Process data (see data/README.md to download the raw file first)
python src/data_processor.py

# Run the full pipeline
python run_pipeline.py --mode full

# Or run individual phases
python run_pipeline.py --mode baseline --models random_forest
python run_pipeline.py --mode robustness
python run_pipeline.py --mode interpretability --llm-provider openai
```

## Results (Chicago taxi, 32-day window, 10 zones)

| Model | RMSE | MAE | R² | MAPE |
|---|---|---|---|---|
| Random Forest | 8.54 | 4.37 | 0.9941 | 17.8% |
| XGBoost | 9.18 | 4.68 | 0.9936 | 18.2% |

Top features are economic (avg fare, trip distance) and temporal (rush-hour flag).
See `docs/PROJECT_COMPLETE_REPORT.md` for the full robustness analysis and `results/` for
spatial heatmaps and per-zone/per-hour breakdowns.

## ST-HAE (experimental prototype — not yet validated)

`src/st_hae_algorithm.py` is a **prototype** of a Spatial-Temporal Hierarchical Attention
Ensemble. In its current NumPy form it **underperforms** the baselines (R² ≈ 0.43) and is
included to document the architecture and as a negative-result baseline — **not** as a
state-of-the-art result. A proper PyTorch / `torch_geometric` re-implementation with a real
GCN, trained attention, and an honest ablation is tracked in [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Data
Public sources; see [`data/README.md`](data/README.md). The processed dataset is included;
the ~204 MB raw file is downloaded separately.

## License
MIT — see [LICENSE](LICENSE).
