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
- **Spatial partition (real, as of Phase 1):** zones are the **9 official Chicago "sides"**
  (community-area→side, all 77 areas partitioned exactly once) + "Unknown" (missing area).
  `grid_processor.py --zone-scheme sides`. The legacy synthetic `(CA-1)//8` "blocks" scheme is
  retained only to reproduce the historical CSV. Headline results use real sides.
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

**Robustness (RandomForest, real Chicago sides, corrected row alignment):**
- *Per-zone:* aggregate R²=0.94 hides a genuinely **negative-R² zone — Far Southwest R²=−0.795**
  (low-volume residential far-SW; model can't beat the mean), while dense zones reach 0.90.
  Absolute error varies ~27× (Far Southeast RMSE 3.2 → Central/Loop 85.8; CV=1.2).
  *(The earlier "Downtown R²=−2674" was an index-misalignment artifact (E-009), retracted;
  the negative-R² finding now rests on a real, interpretable zone.)*
- *Temporal:* worst/best hour RMSE ratio ≈ **15.3×** (hour 4: 3.69 → hour 15: 56.37).
- *Extreme events:* high-demand RMSE degrades **+387%** vs normal.
- *Robust across zone schemes:* synthetic-blocks vs real-sides give near-identical aggregate R²
  (0.941 vs 0.939), temporal (15.8× vs 15.3×), and high-demand (+340% vs +387%) — the finding is
  not an artifact of the grouping.

**Thesis (on real geography):** a strong aggregate R²=0.94 hides a negative-R² zone, a ~15×
temporal error swing, and +387% high-demand degradation. Phase 2 adds bootstrap CIs + conformal
coverage; the window extension + second city (NYC) are the remaining Phase 1 items.

## 8. Discussion & Limitations  🔲
- Aggregate metrics as a false comfort; operational deployment implications.
- Limitations: single-city/short-window (until Phase 1 fixes it), reconstructed pipeline,
  ST-HAE outcome risk.

## 9. Conclusion  🔲

## References  🔲 (BibTeX collected alongside §2)
