# Research Plan — Robust & Explainable Urban Mobility Forecasting

**Goal:** Raise this project from a functional course submission to research-grade work
supporting a **full-conference paper (target: KDD / IJCAI applied track)**.

Two interlocking tracks run in parallel:
- **Track A — Research & Engineering** (build the missing science)
- **Track B — Paper** (draft sections as each result lands)

They sync at the end of every phase: a validated Track A result → a locked Track B section.

---

## Current state (baseline)
- ~5,000 LOC across 10 modules; Phases 1–3 (baselines, robustness, LLM) functional.
- 32-day / 10-zone Chicago dataset; RF R²=0.994 globally but fails under stress.
- ST-HAE is an unvalidated NumPy prototype (R²≈0.43); LSTM broken (categorical encoding).
- No deterministic seeding, no tests, no CI; single git repo with GitHub remote.

## Conference-grade bar (what "done" means)
- Real trained ST-HAE (PyTorch GCN + attention) with honest ablation.
- Comparison vs published baselines: **STGCN** and/or **Graph WaveNet**.
- **Multi-city** (Chicago + NYC TLC) and multi-month data.
- Robustness claims with confidence intervals + calibrated coverage.
- Quantified LLM-explanation faithfulness.
- Fully reproducible (Docker + CI, one-command run).

---

## Phase 0 — Foundation & Reproducibility (Week 1)  ← IN PROGRESS
**Track A**
- Pin/verify env (CPU + one CUDA target).
  - DONE: `lightgbm` 4.1.0 → 4.6.0 (4.1.0 has no py3.11/arm64 wheel; sdist fails under CMake 4.x).
- Deterministic seeding across numpy/sklearn/torch.
- Fix the LSTM categorical-encoding bug so all baselines run.
  - DONE: `prepare_lstm_data` now label-encodes categoricals + builds per-zone sequences.
- Proper temporal train/val/test splits (val for early stopping).
  - DONE: canonical `src/splits.py` (`temporal_split` / `temporal_split_indices`), unit-tested.
  - TODO: wire into `experiment_runner` (replaces random split + fixes E-009 misindex) & re-run.
- pytest unit tests for the **regression metrics** (they live in `baseline_models.py` /
  `robustness_eval.py` / `st_hae_algorithm.py` — NOT `evaluator.py`, which scores LLM text).
  - DONE: `tests/` — 25 tests (metrics, splits, grid_processor) all green.
- ⚠️ Two paper-critical methodology bugs found (E-008 random-split leakage; E-009 robustness
  index misalignment) → current headline numbers must be re-derived after wiring the splitter.

### ⚠️ CRITICAL FINDING — missing dataset-generation code (blocks reproducibility)
The pipeline loads a pre-baked `data/processed/chicago_taxi_processed.csv`
(`run_pipeline.py` default, `experiment_runner.load_data`). **No code in the repo produces
it.** The README attributes grid aggregation to `src/data_processor.py`, but that file is an
unrelated NYC GTFS/taxi/OSM → JSON summarizer (identical copy in the llm submission folder);
it has no grid/hourly aggregation. `data_fetcher.py` has none either. So the documented
463K→7,147 grid-level spatial + hourly aggregation step **cannot be reproduced from source.**
For a KDD/IJCAI submission the dataset pipeline must be reproducible → **reconstructing
`data_processor.py` (raw trips → grid×hour matrix) is now a Phase 0/1 blocker**, folded into
the Phase 1 data scale-up.
  - ✅ RESOLVED (E-007): implemented `src/grid_processor.py`, verified to reproduce the
    historical CSV from raw (trip_count exact; features ~1e-14). Finding: zones are synthetic
    `(CA-1)//8` blocks, not real geography → Phase 1 should install a real spatial partition.

**Track B**
- Set up paper repo (Overleaf/LaTeX), pull KDD/IJCAI template.
- Draft problem statement + contributions.
- Begin related-work collection (STGCN, Graph WaveNet, DCRNN, robustness, LLM-explainability).

**Exit:** all baselines run reproducibly; tests green.

## Phase 1 — Data Scale-Up & Rigor (Weeks 2–3)
- ✅ **Second city + longer window (NYC TLC, Jan–Jun 2024, 6 mo, 19.66M trips):**
  `src/nyc_grid_processor.py`, identical schema to Chicago → pipeline runs unchanged. Longer-window
  goal met via NYC (Chicago Socrata too slow to extend the Chicago window). (E-014)
- ✅ **Re-run baselines at scale; robustness failures persist:** every core failure replicates on
  NYC (temporal 14.3×, high-demand +187%, conformal 90%→31% on high-demand); cross-city table is
  now §7.1. Staten Island recovers a CI-excludes-0 negative-R² micro-zone (read as metric
  degeneracy). (E-014)
- 🔲 Add weather / holidays-events / temporal-lag features.
- 🔲 Fix deferred LSTM predict-path bug (E-010b).
- **Track B:** ✅ Dataset section now two-city; §7.1 cross-city results drafted. First Methods draft 🔲.

## Phase 2 — Robustness Layer, Made Rigorous (Weeks 3–4)
- Bootstrap CIs on per-zone / peak-hour gaps.
- Quantile regression + conformal prediction for calibrated tail coverage.
- Formalize stress-test protocol (spatial/temporal/stability/extreme).
- **Track B:** Robustness Framework methods + headline result figures (with CIs).

## Phase 3 — ST-HAE: The Real Model (Weeks 5–8) ⭐ critical path
1. ✅ Zone-adjacency graph from train-set demand correlation (nan-safe, self-loops). (E-015)
2. ✅ Trained dense Kipf GCN in plain PyTorch (no `torch_geometric` — zones are few). (E-015)
3. ✅ Trained temporal multi-head self-attention encoder (L=24 lookback). (E-015)
4. ✅ End-to-end training + early stopping; masked leakage-free eval == RF's test cells. (E-015)
5. ✅ Honest ablation (full vs no_spatial/no_temporal/no_hierarchical, CIs) **done on Kaggle**
   (both cities). `results/st_hae_{chicago,nyc}.json`. (E-016)
6. ✅ Comparison vs **STGCN + Graph WaveNet** done (`src/st_gnn_baselines.py`, all 3 grids). (E-017)
7. ✅ Finer-grid GCN retry (260 TLC zones) done — spatial still hurts (E-017).

**Result (3 grids, GPU — E-016/E-017):** best variant is **ST-HAE−spatial** (temporal attention +
MoE): Chicago R²=0.9684, NYC boroughs 0.9902, NYC 260-zone 0.9657 — **beats RF, XGBoost, STGCN, and
Graph WaveNet on all three**, flattening the temporal swing to 3.8–6× (from 14–18×). Negative result
**confirmed at both granularities**: the spatial GCN over-smooths and hurts at every scale; the more
spatial-conv-heavy baselines underperform. RF collapses on the fine grid (R²=0.64) — temporal NN wins.

**Guardrail (resolved):** ST-HAE (temporal+MoE core) beats all baselines, so the paper leads with the
model + ablation. Framework (§4) remains the spine.

8. ✅ Learned sparse/distance adjacency + 5-seed variance done (E-018, §5.4). Learned sparse
   adjacency **beats** the correlation graph (Chicago 0.9554→0.9588, NYC 0.9826→0.9850) but still
   loses to `no_spatial` (NYC beyond seed noise; Chicago a tie within noise). `no_spatial > full`
   is decisive on NYC (σ≈0.001), within seed noise on Chicago (σ≈0.008, small data).

**Phase 3 CLOSED.** Recommended model = ST-HAE−spatial (temporal attention + MoE): beats
RF/XGBoost/STGCN/GraphWaveNet on all grids; spatial graph conv is at best neutral even with a learned
graph, so it is dropped. **Track B:** §5 fully written (ablation + ST-GNN baselines + 3-grid + spatial
rescue + multi-seed). Note: Kaggle weekly GPU quota (30 h) is spent — further heavy runs need the
weekly reset or CPU.

## Phase 4 — LLM Explainability, Evaluated (Weeks 8–9)
- Ground-truth failure attribution (per-zone/per-hour causes).
- Agreement metric: do LLM explanations match ground truth? Provider ablation (GPT-4/Claude/Mistral).
- **Track B:** Explainability section with faithfulness metric.

## Phase 5 — Synthesis & Release (Weeks 10–12)
- One-command reproducible run producing every paper number.
- Dockerfile + CI + final README; tagged release.
- **Track B:** Abstract/Intro/Discussion/Limitations/Conclusion; adversarial self-review; arXiv + venue submission.

---

## Section → phase mapping
| Paper section | Fed by | Ready |
|---|---|---|
| Dataset | Phase 1 | Wk 3 |
| Robustness framework | Phase 2 | Wk 4 |
| ST-HAE + ablation | Phase 3 | Wk 8 |
| Baseline comparison | Phase 3 | Wk 8 |
| LLM explainability | Phase 4 | Wk 9 |
| Intro/Related/Discussion | continuous | Wk 11 |

## Top risks
1. ST-HAE may not beat baselines → framework-as-contribution fallback.
2. Data scale-up (NYC) underestimated → can degrade to single-city + future work.
3. Robustness finding might weaken at scale → if it holds, it's the strongest result.
