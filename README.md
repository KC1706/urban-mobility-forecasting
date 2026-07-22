# Robust & Explainable Grid-Level Urban Mobility Demand Forecasting

Grid-level taxi-demand forecasting whose focus is **operational robustness and validated
explanation**, not headline accuracy. Models that look near-perfect on a single global metric
fail exactly where predictions matter most — peak hours, demand spikes, specific zones — and their
uncertainty is miscalibrated precisely there. This repo (1) surfaces those failures with a
statistically rigorous stress-test framework, (2) introduces **ST-HAE**, a trained spatio-temporal
model evaluated with an honest ablation against published ST-GNN baselines, and (3) measures whether
an LLM's failure explanations are actually **faithful** to the ground-truth error attribution.

> Working paper: [`docs/PAPER.md`](docs/PAPER.md) · engineering log: [`docs/ENGINEERING_LOG.md`](docs/ENGINEERING_LOG.md) · plan: [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md)

## Headline findings (two cities, leakage-free)

Aggregate accuracy is a false comfort. Across **Chicago** (9 real "sides", 32 days) and **NYC**
(6 boroughs, 6 months), the same Random Forest that scores R²=0.94 / 0.98 globally:

| Stress dimension | Chicago | NYC |
|---|---|---|
| Temporal error swing (worst/best hour RMSE) | **17.9×** [12.6, 32.2] | **14.3×** [11.3, 22.8] |
| High-demand (≥p95) error degradation | **+481%** [377, 607] | **+187%** [135, 248] |
| Split-conformal @90% → coverage on high-demand | **9.1%** | **31.0%** |

A 90%-calibrated prediction interval covers only 9–31% of high-demand events — **confidently wrong
exactly when demand is high** (Figure 1). All effects carry 95% bootstrap CIs and **replicate across
both cities**. (The earlier "per-zone R² strongly negative" claim did *not* survive CIs and was
retracted — small zones are simply *unmeasurable* at this window; see §7.)

## ST-HAE and the honest ablation

The recommended model — **ST-HAE−spatial** (a trained temporal-attention encoder + mixture-of-experts
head) — beats **RandomForest, XGBoost, STGCN, and Graph WaveNet** on all three grids and flattens the
temporal error swing to 3.8–6× (from 14–18×):

| Grid | RF | ST-HAE−spatial | STGCN | Graph WaveNet |
|---|---|---|---|---|
| Chicago (9 sides) | 0.939 | **0.968** | 0.857 | 0.958 |
| NYC (6 boroughs) | 0.981 | **0.990** | 0.968 | 0.981 |
| NYC (260 TLC zones) | 0.638 | **0.966** | 0.908 | 0.945 |

The ablation is the point: **temporal attention is essential**; the **spatial graph convolution
hurts** at every granularity (a learned sparse adjacency rescues it from harmful to roughly neutral,
but dropping it still wins/ties — decisively on NYC over 5 seeds, within noise on Chicago). We report
what the data says, not a fashionable architecture.

## LLM explainability with a *faithfulness* metric

Instead of unchecked free text, we score an LLM's explanation of the model's failures against the
ground-truth per-factor error attribution. Llama-3.3-70B scores **0.82 (Chicago) / 0.67 (NYC)**
faithful over 5 runs: reliable on intuitive scale drivers, but it misses counter-intuitive ones
(low-volume zones / nights have *lower* error) and hallucinates a `peak_hour` effect that isn't
significant — a gap invisible without the metric.

## Reproduce (no GPU, no downloads — processed data is committed)

```bash
make setup        # .venv (Python 3.11) + requirements-core.txt
make reproduce    # tests + bootstrap-CI robustness results + regenerate paper/figures/
```
Or with Docker:
```bash
docker build -t umf . && docker run --rm umf make reproduce
```
Model training (GPU-friendly; auto-detects CUDA, falls back to CPU):
```bash
make train-sthae        # ST-HAE ablation + STGCN/Graph WaveNet baselines, both cities
make train-adjacency    # spatial-rescue adjacency variants, 5 seeds
```
LLM faithfulness (free: get a key at console.groq.com):
```bash
export GROQ_API_KEY=gsk_...   # or OPENAI_API_KEY / ANTHROPIC_API_KEY
make faithfulness PROVIDERS=groq
```

## Layout

```
src/
  grid_processor.py      Chicago raw -> (zone×hour) grid (reproduces the historical CSV to ~1e-14)
  nyc_grid_processor.py  NYC TLC -> identical schema (second city / fine 260-zone grid)
  splits.py              leakage-free chronological train/val/test
  robustness_ci.py       bootstrap CIs + split-conformal coverage (§4)
  st_hae.py              trained ST-HAE + ablation + adjacency modes + multi-seed (§5)
  st_gnn_baselines.py    STGCN, Graph WaveNet
  llm_faithfulness.py    ground-truth attribution + faithfulness scoring (§6)
  make_figures.py        results/*.json -> paper/figures/*.png
tests/                   66+ tests (pytest); CI in .github/workflows/ci.yml
results/                 committed metrics JSONs (every number in the paper)
paper/figures/           generated figures
```

## Data
Public sources; processed CSVs are committed. See [`data/README.md`](data/README.md) to rebuild from
raw (Chicago Socrata + NYC TLC parquet).

## License
See repository.
