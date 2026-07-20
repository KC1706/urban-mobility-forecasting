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
failures**: the same Random Forest that attains global R²≈0.99 on Chicago taxi demand
collapses to strongly negative per-zone R², doubles its error during the morning rush, and
degrades ~106% on high-demand events. We contribute (1) a **robustness stress-test
framework** that surfaces these spatial/temporal/tail failures with calibrated uncertainty,
(2) **ST-HAE**, a spatial-temporal hierarchical attention ensemble, evaluated with an honest
ablation against STGCN/Graph WaveNet, and (3) an **LLM explainability layer** whose
failure explanations we quantitatively validate against ground-truth error attribution.
[Numbers to be finalized after Phases 1–4.]

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

## 3. Data and Preprocessing  ✍️ (Phase 1; pipeline reproducible as of Phase 0)
- **Source:** City of Chicago "Taxi Trips" open data (`Taxi_Trips_2026.csv`, 463,001 trips,
  Jan 1 – Feb 1 2026 in the current cut). NYC TLC as second city (planned, Phase 1).
- **Grid×hour aggregation (now reproducible from raw):** trips → (zone, hour) cells via
  `src/grid_processor.py`, verified to reproduce the historical dataset (trip_count exact;
  averaged features to ~1e-14). 14 engineered features (temporal + economic + spatial-centroid).
- **⚠️ Spatial-semantics caveat (must state in the paper):** the 10 "zones" are **not real
  Chicago geography** — they are contiguous blocks of 8 Community-Area *numbers*
  (`zone = f((CA-1)//8)`; CA≥65→"Other"; missing→"Unknown"). The directional names are
  arbitrary labels. *Planned fix (Phase 1):* replace with a genuine spatial partition (Chicago
  community-area "sides", or a lat/lon grid) so the grid-level claim is geographically real.
- **Data-quality note:** 5 original rows (27 trips) had corrupt timestamps; the reproducible
  pipeline places them in valid hourly cells instead.
- **Splits:** chronological train/val/test, per-zone aware (Phase 0 utility). 🔲
- Table: dataset statistics (trips, cells, features, demand distribution). 🔲

## 4. Robustness Evaluation Framework  🔲 (Phase 2 — core contribution)
- Four stress dimensions: spatial (per zone), temporal (per hour), stability (over time),
  extreme events (demand strata).
- 90th/95th percentile tail-error isolation; bootstrap CIs on per-zone/peak gaps;
  conformal prediction for calibrated coverage.
- Headline figures: per-zone R² collapse, hourly RMSE curve, demand-stratified degradation. 🔲

## 5. ST-HAE Model  🔲 (Phase 3 — critical path)
- Architecture: zone-adjacency graph (proximity + demand correlation) → trained GCN
  (`torch_geometric`) → temporal `MultiheadAttention` → hierarchical ensemble head.
- Training: end-to-end PyTorch, early stopping, per-zone temporal splits.
- **Ablation:** spatial-only / temporal-only / hierarchical-only / full, with CIs.
- Baselines: RF, XGBoost, LightGBM, LSTM, **STGCN, Graph WaveNet**.

## 6. LLM Explainability  🔲 (Phase 4)
- Post-prediction natural-language explanation of failures.
- **Faithfulness eval:** agreement between generated explanations and ground-truth per-zone /
  per-hour error attribution; provider ablation (GPT-4 / Claude / Mistral).

## 7. Experiments and Results  ✍️ (leakage-free; 32-day Chicago, single run — CIs in Phase 2)

**Honest baselines (chronological 70/15/15 split, held-out test = latest 15%):**
| Model | RMSE | MAE | R² | MAPE |
|---|---|---|---|---|
| Random Forest | 32.55 | 15.37 | 0.9413 | 39.3% |
| XGBoost | 33.59 | 15.97 | 0.9375 | 51.5% |

vs. the earlier *leaky* numbers (random split + train-on-all): RF RMSE 8.54 / R² 0.9941 /
MAPE 17.8%. Leakage inflated RMSE ~3.8× and halved MAPE — the "near-perfect" model was largely
memorization. (ENGINEERING_LOG E-008/E-011.)

**Robustness (RandomForest, corrected row alignment):**
- *Per-zone:* every zone R² is **positive, 0.47–0.88** (Downtown 0.712). The earlier
  "Downtown R²≈−2674 / all regions negative" was an **index-misalignment artifact** (E-009),
  now retracted. What is real: absolute error varies ~30× across zones (Downtown RMSE 2.25 →
  North 68.75; CV=1.18).
- *Temporal:* worst/best hour RMSE ratio ≈ **15.8×** (hour 4: 3.36 → hour 15: 53.13).
- *Extreme events:* high-demand RMSE degrades **+340%** vs normal; low-demand −81%.

**Thesis stands (and is cleaner):** a strong aggregate R²=0.94 still hides that afternoon-peak
error is ~16× off-peak and high-demand error is +340% — but the claim is now the *dispersion of
error* (temporal + demand-strata + per-zone RMSE), not negative R². Phase 2 adds bootstrap CIs
and conformal coverage; Phase 1 re-checks all of this under real spatial zones + longer window.

## 8. Discussion & Limitations  🔲
- Aggregate metrics as a false comfort; operational deployment implications.
- Limitations: single-city/short-window (until Phase 1 fixes it), reconstructed pipeline,
  ST-HAE outcome risk.

## 9. Conclusion  🔲

## References  🔲 (BibTeX collected alongside §2)
