# Robust and Explainable Grid-Level Urban Mobility Demand Forecasting

**Target venue:** KDD / IJCAI (applied track) — preprint on arXiv first.
**Status legend:** ✅ drafted · ✍️ in progress · 🔲 stub (awaiting phase result) · ⚠️ preliminary/unverified

> This is the working draft in Markdown. It converts to the venue LaTeX template in Phase 5.
> Each section is tagged with the plan phase (see `RESEARCH_PLAN.md`) that produces its content.

---

## Abstract  ✍️ (draft — will firm up as results land)
Short-term urban mobility demand forecasting underpins fleet allocation, congestion
management, and smart-city operations. Modern ML/DL models report near-perfect aggregate
accuracy, yet we show these headline numbers **systematically hide operationally critical
failures**. Across **two independent cities** — Chicago (aggregate R²=0.94) and New York City
(R²=0.98) — the same Random Forest that looks near-perfect globally swings its error **~14–18×
across hours of the day**, degrades **+187–481% on high-demand events**, and — most sharply — a
prediction interval calibrated to **90%** coverage overall covers only **9–31%** of high-demand
events. We contribute (1) a **robustness stress-test framework** that surfaces these
spatial/temporal/tail failures with calibrated uncertainty and shows they **replicate across
cities**, (2) **ST-HAE**, a spatial-temporal hierarchical attention ensemble, evaluated with an
honest ablation against STGCN/Graph WaveNet, and (3) an **LLM explainability layer** whose
failure explanations we quantitatively validate against ground-truth error attribution.
[ST-HAE / LLM numbers to be finalized after Phases 3–4.]

---

## 1. Introduction  ✍️ (Phase 0/continuous)
- **Problem.** Grid-level short-term demand forecasting for urban mobility.
- **Gap.** Evaluation is dominated by a single global metric (RMSE/R²); models that look
  excellent globally fail exactly where predictions matter most (peak hours, specific zones,
  demand spikes). Most models are also black boxes.
- **Contributions.**
  1. A robustness-oriented evaluation protocol (spatial / temporal / stability / extreme-event)
     with statistical significance + calibrated tail coverage.
  2. ST-HAE model + honest ablation vs published ST-GNN baselines.
  3. LLM explainability layer with a *quantified faithfulness* evaluation, not free text.
- **Framing note:** the robustness finding is the spine; ST-HAE is a contribution *and* a
  stress-test subject, so the paper stands even if ST-HAE does not beat baselines.

## 2. Related Work  🔲 (Phase 0/1 — collecting)
- Taxi/ride-hailing demand forecasting (classical + DL).
- Spatio-temporal GNNs: **STGCN**, **Graph WaveNet**, **DCRNN**, ST-ResNet.
- Robustness / tail-error / worst-case evaluation in forecasting; conformal prediction.
- LLMs for model explanation & post-hoc interpretability; faithfulness evaluation.
- *Positioning:* prior work optimizes aggregate accuracy; we center operational robustness
  and validated explanation.

## 3. Data and Preprocessing  ✍️ (Phase 1 — two cities, both reproducible from raw)

We evaluate on **two independent cities** built to an **identical (zone × hour) schema** (14
engineered features: temporal + economic + spatial-centroid), so the same forecasting and
robustness pipeline runs unchanged on both. This is the basis for the cross-city replication
in §7.

**City 1 — Chicago (`src/grid_processor.py`).**
- **Source:** City of Chicago "Taxi Trips" open data (`Taxi_Trips_2026.csv`, 463,001 trips,
  Jan 1 – Feb 1 2026 — a 32-day cut). Aggregation verified to reproduce the historical
  processed CSV (trip_count exact; averaged features to ~1e-14).
- **Spatial partition (real):** the **9 official Chicago "sides"** (community-area→side, all
  77 areas partitioned exactly once) + "Unknown". `--zone-scheme sides`. The legacy synthetic
  `(CA-1)//8` "blocks" scheme is retained only to reproduce the historical CSV.
- **Data-quality note:** 5 original rows (27 trips) had corrupt timestamps; the reproducible
  pipeline places them in valid hourly cells instead.

**City 2 — New York City (`src/nyc_grid_processor.py`, added Phase 1).**
- **Source:** NYC TLC "Yellow Taxi Trip Records", monthly parquet, **Jan–Jun 2024 (6 months)**.
  Raw: **20.3M trips → 19.66M kept** (≈3.2% dropped by cleaning) → **22,346 (borough × hour)
  cells**. A ~40× larger and 6× longer sample than the Chicago cut.
- **Spatial partition (real):** pickup TLC zone (`PULocationID`) → **NYC borough** via the
  official `taxi_zone_lookup.csv` (Manhattan / Brooklyn / Queens / Bronx / Staten Island / EWR
  + "Unknown") — the natural analogue of Chicago's "sides". A finer `--zone-scheme zone` keys on
  the ~260 TLC zones directly.
- **Cleaning (documented, reproducible):** keep pickups within the file's nominal month (drops a
  tail of corrupt out-of-range timestamps, e.g. year 2002); trip_distance ∈ (0,100] mi;
  fare ∈ [0,500] \$; duration ∈ (0,180] min; PULocationID present in the lookup.
- **Coordinates:** the parquet stores only zone IDs, so pickup lat/lon are filled with the
  pickup **borough centroid** (documented as approximate; occupies the same spatial-feature slot
  as Chicago's centroids). "Unknown"-zone centroids are NaN → mean-imputed in `prepare_features`,
  exactly as Chicago's Unknown zone.
- **Spatial imbalance (real, and useful):** yellow cabs are Manhattan-dominated — Manhattan
  averages **4,026 trips/hr** vs Staten Island **1.1 trips/hr**, a far steeper gradient than
  Chicago and a natural stress test for per-zone robustness.

- **Splits:** chronological train/val/test (70/15/15) on ordered unique timestamps, no same-hour
  straddle (`src/splits.py`). ✅
- Table: dataset statistics (trips, cells, features, demand distribution) per city. 🔲

## 4. Robustness Evaluation Framework  ✍️ (Phase 2 — core contribution; CIs + conformal done)
- Four stress dimensions: spatial (per zone), temporal (per hour), stability (over time),
  extreme events (demand strata) — `robustness_eval.py`.
- **Statistical rigor (`robustness_ci.py`):** percentile **bootstrap CIs** (B=2000) on per-zone
  R², the temporal worst/best-hour RMSE ratio, and high-demand degradation; **split-conformal**
  prediction intervals with per-stratum empirical coverage (overall / peak / high-demand / zone).
- Method value demonstrated: CIs *retract* the non-robust per-zone negative-R² claim and
  *confirm* the temporal (17.9×) and high-demand (+481%) effects; conformal exposes the
  calibration collapse (90%→9% coverage on high-demand). Rigor changes the conclusions.
- Headline figures: hourly RMSE curve w/ CIs, demand-stratified degradation w/ CIs, per-stratum
  conformal coverage bar. 🔲 (plots)

## 5. ST-HAE Model  ✍️ (Phase 3 — critical path; implemented + trained, ablation on Kaggle GPU)

**From negative result to real model.** The original `st_hae_algorithm.py` prototype left every
component *untrained* (fixed-identity "GCN", unlearned dot-product "attention", data-starving
per-quantile sub-models) and honestly underperformed the baselines (R²≈0.43). We re-implemented it
end-to-end in PyTorch (`src/st_hae.py`), keeping the four conceptual pillars but making each one
*learned*:
- **Spatial** — a trained Kipf-normalized graph convolution (2 layers) over zones. The adjacency
  is built from *training-set* per-zone demand correlation (edge where ρ>0.3) + self-loops. Zones
  are few (≤ dozens), so a dense GCN needs no `torch_geometric`.
- **Temporal** — a trained 2-layer multi-head self-attention encoder over an L=24 h lookback window
  (learned Q/K/V), replacing the prototype's raw-feature dot products.
- **Hierarchical** — a learned **mixture-of-experts** head (3 experts + softmax gate). Unlike the
  prototype's per-quantile data split, *all* data trains *all* experts end-to-end; the gate learns
  the demand-regime specialization.
- **Ensemble** — the MoE gate *is* the adaptive combination, trained jointly with everything else.

**Training/eval.** End-to-end (Adam, grad-clip, early stopping on val RMSE), leakage-free
chronological split (`splits.py`), per-zone standardized targets, masked to the cells observed in
the processed CSV so **y_true is identical to the RandomForest baseline** in §7. Same robustness
dimensions + bootstrap CIs as §4/§7. Deterministic (seed 42). Runs on GPU via the Kaggle notebook
`notebooks/st_hae_kaggle.ipynb` (`--device cuda`).

**Ablation:** leave-one-pillar-out — `full` vs `no_spatial` / `no_temporal` / `no_hierarchical`,
each with CIs, on both cities. *(Full ablation + NYC + CIs are produced by the Kaggle run;
`results/st_hae_{chicago,nyc}.json`.)* 🔲

**Preliminary result (Chicago, single leakage-free run — to be confirmed by the Kaggle ablation):**
the trained ST-HAE **beats RandomForest and narrows the robustness gaps it was built to expose**:

| Model | RMSE | R² | temporal worst/best | high-dmd degradation |
|---|---|---|---|---|
| RandomForest | 35.10 | 0.9388 | 17.9× | +481% |
| **ST-HAE (full)** ⚠️ | **30.04** | **0.9551** | **13.4×** | **+370%** |

This is the paper's arc closing: the robustness framework (§4) surfaced operational failures, and a
model designed around them *measurably reduces* the temporal swing and tail degradation while
improving aggregate accuracy — not just a headline-metric win. ⚠️ single run; the Kaggle ablation
supplies CIs, component attribution, and the NYC replication.

**Baselines:** RF / XGBoost / LightGBM / LSTM (Phase 0–1). Published ST-GNN baselines (**STGCN,
Graph WaveNet**) are not yet implemented — noted as remaining work, not claimed. 🔲

## 6. LLM Explainability  🔲 (Phase 4)
- Post-prediction natural-language explanation of failures.
- **Faithfulness eval:** agreement between generated explanations and ground-truth per-zone /
  per-hour error attribution; provider ablation (GPT-4 / Claude / Mistral).

## 7. Experiments and Results  ✍️ (leakage-free; Chicago 32 d + NYC 6 mo, CIs — Phase 1/2)

**Honest baselines (chronological 70/15/15 split, held-out test = latest 15%):**
| Model | RMSE | MAE | R² | MAPE |
|---|---|---|---|---|
| Random Forest | 32.55 | 15.37 | 0.9413 | 39.3% |
| XGBoost | 33.59 | 15.97 | 0.9375 | 51.5% |

vs. the earlier *leaky* numbers (random split + train-on-all): RF RMSE 8.54 / R² 0.9941 /
MAPE 17.8%. Leakage inflated RMSE ~3.8× and halved MAPE — the "near-perfect" model was largely
memorization. (ENGINEERING_LOG E-008/E-011.)

**Robustness with 95% bootstrap CIs (RandomForest R²=0.939, real Chicago sides, held-out test):**
- *Per-zone R² — claim retracted under rigor:* the smallest zone (Far Southwest) has
  R²=−0.02 **[−1.32, 0.62]** — the CI is enormous and straddles zero, so a negative-R² claim is
  **not statistically supported**. (The earlier −2674 and −0.795 point estimates were noise +
  the E-009 index bug.) Honest reframing: performance in low-volume zones is *unmeasurable* with
  this window — itself an operational caveat.
- *Temporal error dispersion (robust):* worst/best-hour RMSE ratio **17.9× [12.6, 32.2]**.
- *High-demand degradation (robust):* **+481% [377%, 607%]** vs normal demand.
- *⭐ Calibration collapse (headline result):* a split-conformal interval calibrated to **90%**
  coverage overall covers only **9.1%** of high-demand (≥p95) events and 80.7% at peak hours.
  The model is *confidently wrong exactly when demand is high* — a crisp, quantitative failure
  that a single global metric or interval hides.

### 7.1 Cross-city replication (NYC, 6 months) — the robustness failures generalize  ⭐

We rerun the *identical* pipeline on the second city (NYC yellow taxi, Jan–Jun 2024, borough
grid, RandomForest, leakage-free chronological split). Every core failure mode reproduces, and
the small-zone effect that Chicago's CIs retracted is *recovered* under NYC's steeper spatial
gradient.

| Metric (held-out test) | **Chicago** (32 d, 9 sides) | **NYC** (6 mo, boroughs) |
|---|---|---|
| Aggregate R² (the "false comfort") | 0.939 | **0.981** |
| Temporal worst/best-hour RMSE ratio | 17.9× [12.6, 32.2] | **14.3× [11.3, 22.8]** |
| High-demand (≥p95) RMSE degradation | +481% [377, 607] | **+187% [135, 248]** |
| ⭐ Conformal @90% nominal → high-demand coverage | 9.1% | **31.0%** |
| Worst per-zone R² (95% CI) | Far SW −0.02 [−1.32, 0.62] *(retracted)* | **Staten Is. −5.96e4 [−2.4e5, −826]** *(CI excludes 0)* |

**Reading of the table.**
- *Higher aggregate R² (0.98), worse hidden failures:* the larger, cleaner NYC sample pushes the
  headline metric even closer to "perfect," yet the same stress tests expose a 14× temporal swing
  and a 3× high-demand error blow-up. The false-comfort thesis is *not* an artifact of Chicago's
  small window.
- *Calibration collapse replicates:* a 90%-nominal conformal interval covers only **31%** of NYC
  high-demand events (vs 9% in Chicago). Two cities, same qualitative failure — the model is
  confidently wrong exactly where demand is high.
- *Negative-R² micro-zone, done honestly:* NYC's Staten Island (~1.1 trips/hr, near-constant
  demand) yields a strongly negative per-zone R² whose CI **excludes zero** — but we read this as
  **metric degeneracy**, not catastrophic error: with a near-constant target the R² denominator
  collapses, so R² is the wrong tool for micro-zones (absolute error there is tiny). This *is* the
  paper's point — global **and** naïve per-zone R² both mislead; only absolute-error + calibrated
  coverage tell the operational truth.

### 7.2 Thesis (rigorous)

Across **two cities**, an aggregate R² of 0.94–0.98 conceals a ~14–18× temporal error swing, a
+187–481% high-demand degradation, and a calibration collapse (90%→9–31% coverage) on high-demand
events. The dramatic per-zone R² numbers are either retracted (Chicago) or shown to be metric
degeneracy (NYC micro-zones); the robust, operationally meaningful failures are the temporal
dispersion, the tail degradation, and the conformal under-coverage — and all three replicate
across cities. (Reproduce: `robustness_ci.py --data data/processed/{chicago_taxi_sides,nyc_taxi_boroughs}.csv`;
JSON in `results/`.)

## 8. Discussion & Limitations  🔲
- Aggregate metrics as a false comfort; operational deployment implications.
- Limitations: **two cities** (Chicago 32 d + NYC 6 mo) but both US taxi systems and a single
  mode; NYC yellow-cab coverage is Manhattan-skewed (a property we exploit, but it limits
  outer-borough conclusions); reconstructed Chicago pipeline; NYC lat/lon approximated by borough
  centroid; ST-HAE outcome risk.

## 9. Conclusion  🔲

## References  🔲 (BibTeX collected alongside §2)
