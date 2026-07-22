# Engineering & Learning Log

A living record of problems hit, how they were resolved, findings, and per-phase learnings.
Runs in parallel with `PAPER.md` and `RESEARCH_PLAN.md`. Newest entries at the top of each phase.

**Entry format:** `ID · date · [phase] · title` → **Symptom / Root cause / Fix / Learning**.

---

## Phase 0 — Foundation & Reproducibility (in progress)

### E-011 · 2026-07-21 · [P0] ✅ Honest results after temporal-split rewire (resolves E-008/E-009)
- **Wired** `temporal_split_indices` into `experiment_runner`: split chronologically BEFORE
  training, train on the earliest 85%, evaluate on the latest 15% (1,084 rows), and pass real
  positional `test_indices` to the robustness join. Also fixed the nan summary (report held-out
  test metrics) and added `tabulate` (pandas `.to_markdown()` for the robustness report).
- **Leakage was severe.** RF: RMSE 8.54→**32.55** (~3.8×), R² 0.9941→**0.9413**,
  MAPE 17.8%→**39.3%**. The near-perfect model was largely memorization (trained on rows it was
  then tested on).
- **Downtown R²≈−2674 was an artifact (E-009 confirmed).** With correct alignment every zone
  R² is positive (0.47–0.88; Downtown 0.712). The "all regions negative" claim is retracted.
- **But the robustness thesis survives, stronger:** per-zone RMSE spread ~30× (Downtown 2.25 →
  North 68.75), temporal worst/best-hour RMSE ratio ≈**15.8×**, high-demand degradation **+340%**.
  The real finding is *error dispersion*, not negative R².
- **Learning:** two bugs pulled in opposite directions — leakage made the model look too good,
  the index bug made a zone look catastrophically bad. Fixing both gives a credible middle: a
  good-but-not-perfect model whose errors are highly concentrated in peaks/high-demand/hot zones.
  Deferred: LSTM predict-path shape bug (E-010b); LSTM excluded from this honest run.

### E-010 · 2026-07-21 · [P0] Baseline smoke run: plumbing OK, 3 minor bugs surfaced
- **Run:** `run_pipeline.py --mode baseline --models random_forest xgboost lstm --no-llm`
  → exit 0. Validates env + reproducible data + training + metrics end-to-end.
- **RF/XGB trained fine** (RF CV neg-MSE −870.4 → CV RMSE ≈29.5, consistent with history).
  Still the *leaky random split* (E-008), so not trustworthy — expected.
- **Bug a (cosmetic):** summary prints `random_forest: nan` — `compare_baseline_models`
  builds a mixed-schema DataFrame so RF/XGB rows get a NaN `rmse` column, and the printout does
  `row.get('rmse', cv_score)` which returns the NaN instead of falling back. Fix in the split rewire.
- **Bug b:** LSTM predict path in `experiment_runner` raises `input_shape[1] -> NoneType`
  (`model.predict(X_test_lstm)` with an undefined shape). Training works; the evaluate/predict
  path needs the scaled, correctly-shaped tensor. Fix alongside the rewire.
- **Learning:** "PIPELINE COMPLETED SUCCESSFULLY" + a printed number is not success — RF/XGB
  showed nan and LSTM errored mid-predict, yet exit code was 0. Need real assertions on outputs.

### E-009 · 2026-07-21 · [P0] ⚠️ Robustness analysis joins predictions to the WRONG rows
- **Symptom:** `experiment_runner._prepare_test_data` returns
  `test_indices = np.arange(len(X_test))`, and `run_robustness_analysis` then does
  `data.iloc[test_indices]`.
- **Root cause:** after a shuffled `train_test_split`, the test rows are NOT the first
  `len(X_test)` rows of `data`. So every prediction is matched to an unrelated row before the
  spatial/temporal/extreme-event breakdown.
- **Impact:** the headline robustness numbers (e.g. Downtown R²=−2674) are computed on
  mis-joined (row, prediction) pairs — they may be substantially an **artifact**, not a real
  operational failure. This must be re-derived after the fix before any paper claim.
- **Fix (in progress):** `src/splits.temporal_split_indices` returns the true positional
  indices of the test rows; wiring it into `experiment_runner` replaces both the random split
  and the bogus `arange`. Re-run required.
- **Learning:** a plausible, dramatic result is not evidence it's real; index bookkeeping bugs
  fabricate exactly this kind of "striking finding."

### E-008 · 2026-07-21 · [P0] ⚠️ Random train/test split leaks the future (time-series)
- **Symptom:** `experiment_runner` splits RF/XGB/MLP data with sklearn `train_test_split`
  (random), only LSTM sorts by time.
- **Root cause:** random splitting on temporally-ordered data puts future hours in the
  training set → optimistic, leakage-inflated accuracy (contributes to R²≈0.994).
- **Fix (in progress):** canonical chronological splitter `src/splits.py`
  (`temporal_split` / `temporal_split_indices`), unit-tested for no-leakage + correct indexing.
  Wire into `experiment_runner` and re-run baselines to get honest numbers.
- **Learning:** for forecasting, the split *is* part of the method; a random split silently
  invalidates the benchmark.

### E-007 · 2026-07-21 · [P0] ✅ Reconstructed the missing grid-aggregation pipeline
- **Resolves E-004.** Recovered the raw→processed recipe and implemented it as
  `src/grid_processor.py` (`ChicagoTaxiGridProcessor`). `--verify` reproduces the historical
  `chicago_taxi_processed.csv` from raw: **trip_count 0 mismatches / 7,142 shared rows**,
  averaged features to ~1e-14.
- **How the recovery worked:**
  1. Confirmed hourly bucketing = `Trip Start Timestamp` floored to the hour (total-per-hour
     matched 744/745; the 1 off-hour = 27 trips the original wrote as 5 corrupt-timestamp rows).
  2. `Unknown` zone = missing `Pickup Community Area` (per-hour exact match).
  3. Recovered the Community-Area→zone partition by matching per-(area, hour) counts against
     per-(zone, hour) target counts across 745 timestamps.
- **Key finding — zones are synthetic:** the partition is simply `(CommunityArea-1)//8` →
  8 directional labels, areas ≥65 → `Other`, missing → `Unknown`. The "geographic" zone names
  are **arbitrary labels on numeric blocks**, not real Chicago geography. Flagged in `PAPER.md`
  §3 so the spatial claims aren't overstated. *(Phase 1 should consider replacing this with a
  real spatial partition — community-area→side or lat/lon grid.)*
- **Gotchas hit:** (a) `pd.read_csv(parse_dates=...)` silently left `pickup_datetime` as
  strings → an index-type mismatch made the first recovery attempt match nothing; fixed with
  explicit `to_datetime`. (b) `Trip Miles`/`Trip Seconds`/`Fare` are object dtype (strings,
  `$`), needed numeric coercion before `mean`.
- **Learning:** the whole "grid-level spatial" framing rests on a throwaway `//8` block rule;
  reconstruction both restored reproducibility *and* surfaced a substantive validity caveat the
  paper must state.

### E-006 · 2026-07-21 · [P0] `evaluator.py` is not the metrics module
- **Symptom:** Planned unit tests for regression metrics (RMSE/MAE/R²/MAPE) assuming they
  lived in `evaluator.py`.
- **Root cause:** `evaluator.py` scores *LLM-generated text quality* (coverage,
  interpretability, actionability). The forecasting metrics actually live in
  `baseline_models.py`, `robustness_eval.py`, and `st_hae_algorithm.py`.
- **Fix:** Re-pointed the Phase 0 test target at those three modules.
- **Learning:** Don't infer a module's role from its name — verify against the README's
  claimed pipeline *and* the code. The repo's file names don't reliably match their function.

### E-005 · 2026-07-21 · [P0] Recovered the 10-zone structure
- **Finding:** Target `pickup_borough` values are 10 zones — `Downtown, North, Northwest,
  West, Southwest, South, Southeast, FarSouth` + `Other` (unmapped) + `Unknown` (missing
  community area) — geographic groupings of Chicago's 77 Community Areas. ~545–745 hourly
  rows each (of a 32-day × 24h = 768 max), which is internally consistent.
- **Open item:** the exact Community-Area→zone dictionary is not in any code copy; recover it
  empirically from the raw file (aggregate by `Pickup Community Area`, match per-zone counts
  and centroids to the target CSV). Byte-exact md5 is a stretch goal; realistic criterion is
  key/`trip_count` exact + avg features to ~1e-6.

### E-004 · 2026-07-21 · [P0] ⚠️ Missing dataset-generation code (major finding)
- **Symptom:** No code produces `data/processed/chicago_taxi_processed.csv`; the pipeline
  only *loads* it (`run_pipeline.py:43`, `experiment_runner.py:106`).
- **Root cause:** README attributes grid aggregation to `src/data_processor.py`, but that
  file (identical across repo, `submission_ready 3`, and the zip) is an unrelated NYC
  GTFS/taxi/OSM → JSON summarizer. The 463K→7,147 aggregation step was never committed.
- **Fix:** Raw source recovered — `submission_ready 2.zip` contained `Taxi_Trips_2026.csv`
  (204 MB, 463,001 trips) and a `chicago_taxi_processed.csv` that is **md5-identical**
  (`fb3ed84eb23e89e176b58ed1d6928b2f`) to the repo's. Extracted raw to `data/raw/`.
  Reconstruction of `data_processor.py` (raw→grid) is now a Phase 0/1 deliverable, verifiable
  against the known md5.
- **Learning:** For a conference submission, "the results reproduce" must mean *from raw*,
  not from a pre-baked artifact. A repo can pass a demo run while being unreproducible.

### E-003 · 2026-07-21 · [P0] LSTM categorical-encoding crash + zone-mixing bug
- **Symptom:** LSTM baseline failed ("categorical encoding issue"); never ran.
- **Root cause:** `prepare_lstm_data` kept all columns except target/datetime, so the string
  `pickup_borough` reached `StandardScaler` as object dtype → crash. Deeper: sequences were
  built over the frame sorted by time only, so each 24-step window silently mixed multiple
  zones.
- **Fix:** `prepare_lstm_data` now label-encodes categoricals + drops datetime cols (mirrors
  `prepare_features`), and builds sequences **per zone** before a within-zone temporal split.
- **Status:** Code fixed; needs an end-to-end run to confirm once verifying the full pipeline.
- **Learning:** A "preprocessing bug" was masking a correctness bug (cross-zone sequences)
  that would have quietly biased LSTM results in the paper.

### E-002 · 2026-07-21 · [P0] lightgbm 4.1.0 won't install on Python 3.11 / arm64
- **Symptom:** `pip install -r requirements.txt` failed building the lightgbm wheel:
  `CMake Error ... Compatibility with CMake < 3.5 has been removed`.
- **Root cause:** No prebuilt py3.11/arm64 wheel exists for lightgbm 4.1.0 (only 4.4.0+),
  so pip fell back to the source sdist, which declares an old `cmake_minimum_required` that
  Homebrew CMake 4.1.1 rejects.
- **Fix:** Bumped pin `lightgbm==4.1.0 → 4.6.0` (has a matching wheel). Install then completed
  with zero source builds.
- **Learning:** Aging pinned stacks fight modern build toolchains on Apple Silicon; prefer
  versions with prebuilt wheels, and consider conda-forge for compiled libs (Phase 5).

### E-001 · 2026-07-21 · [P0] Default interpreter unusable
- **Symptom:** `python3` is 3.14.6 with no project deps; `import pandas` fails.
- **Root cause:** Pinned stack (numpy 1.24.3, tf 2.15) predates Python 3.14 wheels; the
  project was run in a separate, unwired env.
- **Fix:** Created `.venv` on Python 3.11; installed after the E-002 fix. Verified imports:
  pandas 2.1.3, numpy 1.24.3, sklearn 1.3.2, xgboost 2.0.3, lightgbm 4.6.0, torch 2.1.0,
  tensorflow 2.15.0.
- **Learning:** Pin the *interpreter* version too (document Python 3.11) — an unpinned Python
  is an unpinned dependency.

### Phase 0 — running learnings
- The repo demonstrably diverges from its README; treat documented claims as hypotheses to verify.
- Reproducibility gaps (missing pipeline code) are the biggest near-term risk to the paper,
  above any modeling improvement.
- Keep the raw→processed path md5-checkable so regressions in preprocessing are caught early.

---

## Phase 1 — Data Scale-Up & Rigor (in progress)

### E-012 · 2026-07-21 · [P1] ✅ Real Chicago geography replaces synthetic zones
- Added `--zone-scheme {blocks,sides}` to `grid_processor.py`. `sides` = the 9 official Chicago
  sides (community-area→side, all 77 areas partitioned once, unit-tested). Legacy `blocks`
  retained + still reproduces the historical CSV. Built `data/processed/chicago_taxi_sides.csv`.
- **Robustness re-checked under real geography (RF, held-out test):**
  - Aggregate R² stable: 0.939 (sides) vs 0.941 (blocks).
  - A **genuine negative-R² zone re-emerges: Far Southwest R²=−0.795** (low-volume residential
    far-SW). Interpretable and real — unlike the retracted −2674 artifact.
  - Per-zone RMSE spread ~27× (Far Southeast 3.2 → Central/Loop 85.8).
  - Temporal worst/best-hour ratio 15.3× (vs 15.8× blocks); high-demand degradation +387%
    (vs +340%).
- **Learning:** the robustness thesis is *not* an artifact of the arbitrary grouping — it holds
  across schemes, and real geography makes the negative-R² claim defensible (a real neighborhood,
  not a numeric block). This is a stronger paper result than the original.
- **Remaining Phase 1:** extend window (6–12 mo), add NYC TLC as a second city, weather/holiday/
  lag features; fix deferred LSTM predict-path bug.


## Phase 2 — Robustness Layer, Made Rigorous (in progress)

### E-013 · 2026-07-21 · [P2] ✅ Bootstrap CIs + conformal coverage; two claims retracted
- Added `src/robustness_ci.py`: percentile bootstrap CIs (per-zone R², temporal ratio,
  high-demand degradation) + split-conformal coverage per stratum. Tests in
  `tests/test_robustness_ci.py` (5, green). Report: `results/phase2_robustness_ci.json`.
- **Gotcha found & fixed (why R² read 0.84 not 0.94):** reading the CSV left `pickup_datetime`
  as a *string*, so `prepare_features` label-encoded it into a feature with **inconsistent
  train/test codes** → silently degraded the model. `experiment_runner.load_data` avoids this by
  `pd.to_datetime(...)` (datetime64 gets dropped). Fixed the analysis script to parse it too;
  R² then matched the headline 0.9388 exactly. *(Latent sharp edge in `prepare_features`: it
  should drop/parse datetime-like string columns, not encode them.)*
- **Rigor changed the conclusions:**
  - *Retract* per-zone negative-R²: Far Southwest R²=−0.02 **[−1.32, 0.62]** — CI straddles 0,
    not supported. (Confirms −2674/−0.795 were noise + the E-009 bug.)
  - *Confirm* temporal dispersion **17.9× [12.6, 32.2]** and high-demand degradation
    **+481% [377, 607]**.
  - *New headline:* split-conformal @90% nominal covers only **9.1%** of high-demand events
    (peak hours 80.7%) — calibration collapses exactly where it matters.
- **Learning:** CIs are not decoration — they overturned the paper's most eye-catching claim
  (negative per-zone R²) and promoted a stronger, defensible one (coverage collapse). Point
  estimates on small strata were within noise.


### E-014 · 2026-07-21 · [P1] ✅ Second city (NYC, 6 mo) — every robustness failure replicates
- **What:** added `src/nyc_grid_processor.py` (+`tests/test_nyc_grid_processor.py`, 7 green)
  building an NYC yellow-taxi (zone × hour) dataset with the **identical schema** to Chicago, so
  the whole pipeline runs unchanged. Source: NYC TLC monthly parquet, Jan–Jun 2024.
  `data/processed/nyc_taxi_boroughs.csv` = **22,346 cells, 19.66M trips** (20.3M raw, 3.2% cleaned).
- **Symptom / gotchas resolved while building:**
  - Raw parquet carries a tail of **corrupt out-of-range timestamps** (pickup min = 2002) → filter
    each file to its nominal (year, month).
  - Parquet has **no lat/lon**, only `PULocationID` → map to borough via `taxi_zone_lookup.csv`,
    fill centroid from a borough-centroid table; "Unknown" (LocID 264/265) → NaN → mean-imputed by
    `prepare_features` (same path as Chicago's Unknown zone, so no code change needed).
  - `pyarrow` wasn't in the venv → `pip install pyarrow` (25.0.0).
  - Chicago Socrata endpoint is reachable but **too slow / no server-side date filter** to extend
    the Chicago window; NYC CloudFront parquet is fast (~10 s/file). So the "longer window" goal is
    met via NYC (6 mo) rather than more Chicago months — documented as a limitation.
- **New findings (cross-city replication — the big result):** rerunning `robustness_ci.py` on NYC,
  every core failure reproduces and the retracted small-zone effect is *recovered*:
  - Aggregate R² **0.981** (even higher than Chicago's 0.939 — false comfort is not a small-sample
    artifact).
  - Temporal worst/best-hour ratio **14.3× [11.3, 22.8]** (Chicago 17.9×).
  - High-demand degradation **+187% [135, 248]** (Chicago +481%).
  - ⭐ Conformal @90% nominal → **31.0%** coverage on high-demand (Chicago 9.1%) — *the headline
    calibration collapse replicates in a second city.*
  - Staten Island per-zone R² **−5.96e4 [−2.4e5, −826]**, CI **excludes 0** — but read as **metric
    degeneracy** (near-constant ~1.1 trips/hr → R² denominator collapses), reinforcing "R² is the
    wrong tool for micro-zones," not a catastrophic-error claim.
- **Learning:** building the second city to a *byte-identical output schema* (shared
  `OUTPUT_COLUMNS`, shared feature flags — asserted in a test) meant zero pipeline changes and a
  clean apples-to-apples comparison. The cross-city table is now the paper's strongest evidence:
  the robustness failures are a property of the *problem*, not one dataset.
- Reports: `results/nyc_robustness_ci.json`, `results/chicago_sides_robustness_ci.json`. Full
  suite 47 passed.


## Phase 4 — LLM Explainability, Evaluated  ✅ (framework + ground truth + real Llama-3.3-70B via Groq)

### E-020 · 2026-07-23 · [P4] Real faithfulness result via Groq (free): Llama-3.3-70B is only 0.67–0.82 faithful
- **Unblocked with Groq** (user supplied a free Groq key). Groq is OpenAI-API-compatible → added a
  `groq` provider (OpenAI SDK + `base_url=api.groq.com/openai/v1`, `llama-3.3-70b-versatile`).
- **Result (mean±std over 5 runs — added `--repeats` because LLM output is noisy even at temp 0):**
  Chicago **0.82±0.10**, NYC **0.67±0.00**. `results/faithfulness_{chicago,nyc}.json`.
- **What the metric caught (the whole point):**
  - LLM reliably gets the **scale-driven** drivers (high_demand↑, high_volume_zone↑ increase error) —
    the intuitive "more activity → more error" prior.
  - It **misses/flips the counter-intuitive** ones: low_volume_zone and night have *lower* abs error
    (small counts) — on Chicago it reversed night; on both it missed low_volume_zone.
  - It **hallucinates** a peak_hour effect on NYC (halluc 0.25) — peak hours *feel* error-prone but
    aren't significant once demand is controlled (the designed trap).
  - Explanations are **noisy**: Chicago faithfulness varies ±0.10 across identical prompts → single-shot
    LLM explanations are unreliable; must average.
- **Nondeterminism note:** first two single-run groq calls gave 0.72 then 0.92 on Chicago (temp 0),
  which is exactly why I added multi-run mean±std. NYC was stable (±0.0).
- **Security:** key was used from env only, never committed; verified no `gsk_` in any results JSON.
  (Flagged to user: rotate the key since it was shared in chat.)
- **Learning:** a free OpenAI-compatible provider (Groq) turned "pending credentials" into a real,
  defensible result in minutes — and multi-run averaging (same discipline as ST-HAE multi-seed) was
  essential because the LLM isn't deterministic. [[E-019]]

### E-019 · 2026-07-22 · [P4] Quantified faithfulness framework built + ground truth; real-provider run blocked (no funded keys)
- **What:** `src/llm_faithfulness.py` (+`tests/test_llm_faithfulness.py`, 5 green) — turns "LLM
  explanation quality" into a number. Ground-truth error attribution (Mann-Whitney signed effect of
  each interpretable factor on the RF's abs-error) → LLM elicited to infer drivers from a sample of
  test cells (strict JSON) → faithfulness = mean(directional accuracy on sig factors, top-3 driver
  recall, 1−hallucination rate) + Spearman rank agreement. Deterministic `mock` provider makes the
  whole thing testable/CI-able with no API.
- **Real ground truth (genuine result):** Chicago drivers `high_demand(9.6×) > high_volume_zone(9.2×)
  > low_volume_zone(0.11×) > night(0.52×)`; NYC `low_volume_zone(0.03×) > high_volume_zone(23.4×) >
  high_demand(9.3×) > night > weekend`. `peak_hour` is NOT a significant independent driver once
  demand is controlled — a nice trap for plausible-but-wrong explanations. `results/faithfulness_{chicago,nyc}.json`.
- **Blocker (documented honestly, not faked):** none of the three providers can call right now —
  OpenAI key valid but **out of billing quota (429)**; Anthropic key is a 20-char **placeholder**;
  HuggingFace token **lacks inference permission (403)**. The mock scores 0.82/0.76 (scorer sanity),
  but NO real-LLM numbers were produced. Paper §6 states this and gives the one-command path.
- **Env fixes:** upgraded `openai` 1.3.5→2.46 (old SDK crashed on httpx `proxies`), `huggingface_hub`
  →1.24 (old endpoint `api-inference.huggingface.co` is dead / unresolved; new `router.huggingface.co`
  works). Keys read from env only; verified **no secrets in the committed JSONs**.
- **Learning:** the faithfulness *framework* + the ground-truth attribution ARE the contribution and
  don't need an LLM to be valid; the provider ablation is a one-command add-on. Refused to invent
  LLM scores — a fabricated table would be worse than an honest "pending credentials".

## Phase 3 — ST-HAE: The Real Model  ✅ (trained; ablation + ST-GNN baselines across 3 grids, on Kaggle GPU)

### E-018 · 2026-07-22 · [P3] Spatial rescue (learned/geographic adjacency) + 5-seed variance: graph was suboptimal but spatial still doesn't win
- **What ran:** added 3 adjacency modes to the GCN — `full_distance` (geographic kNN Gaussian),
  `full_adaptive` (learned dense, Graph-WaveNet style), `full_adaptive_sparse` (learned top-k sparse)
  — and ran all variants over **5 seeds** for mean±std, coarse grids. `results/st_hae_{chicago,nyc}_adj.json`.
- **Findings:**
  1. *Part of the earlier failure was the graph, not the component:* a **learned sparse adjacency
     beats the correlation-thresholded one** on both cities (Chicago full 0.9554→0.9588; NYC
     0.9826→0.9850). The ρ>0.3 demand-correlation graph really was a poor choice.
  2. *But spatial still doesn't win:* even the best learned adjacency < `no_spatial` — NYC decisively
     (ΔR²=+0.0047, pooled σ≈0.0020, beyond noise), Chicago only a tie (ΔR²=+0.0032, pooled σ≈0.011,
     within noise).
  3. *Multi-seed tempers the headline:* `no_spatial > full` is decisive on NYC (σ≈0.0006) but **within
     seed noise on Chicago** (σ≈0.008, small 32-day data). Removed the "single-seed" caveat from §5.3
     and added §5.4 with mean±std.
- **Ops story (documented for reproducibility):**
  - **Kaggle weekly GPU quota (30 h) exhausted** — a `kernels push` with `enable_gpu:true` fails with
    "Maximum weekly GPU quota reached". The heavy 5-seed×3-grid GPU run (v4) had silently **fallen back
    to CPU** (P100 cu121 fix didn't land that allocation) and was heading for the 12 h wall on the
    260-node fine grid (~27 h of CPU work). Can't cancel via API.
  - **Fix:** re-pushed as a **CPU kernel** (`enable_gpu:false` — sidesteps the exhausted GPU quota),
    **coarse grids only** (fast on CPU), which supersedes the running version. Completed in ~3.5 h.
    The fine-grid spatial verdict stands on the E-017 GPU run. [[kaggle-p100-torch-fix]]
- **Learning:** the honest multi-seed result is *more* nuanced and more credible than the single-seed
  one — it both (a) vindicates the hypothesis that the correlation graph was the problem, and (b) shows
  spatial still doesn't beat dropping it, while (c) admitting Chicago can't resolve the difference at
  all. "Run it 5 times" changed a clean story into a true one. [[E-017]]

### E-017 · 2026-07-21 · [P3] ⭐ Fine-grid retry + ST-GNN baselines (Kaggle GPU): spatial GCN hurts at every scale; ST-HAE−spatial beats STGCN & Graph WaveNet
- **What ran (on GPU this time):** added STGCN + Graph WaveNet (`src/st_gnn_baselines.py`, sharing
  the ST-HAE forward signature + eval harness) and a fine ~260-zone NYC TLC grid
  (`nyc_taxi_zones.csv.gz`). Kaggle kernel v3 ran all variants+baselines on 3 grids.
  `results/st_hae_{chicago,nyc,nyc_zones}.json`.
- **GPU fix that finally worked:** the P100 (sm_60) is incompatible with Kaggle's default
  torch 2.10+cu128 (sm_70+). The kernel now detects "P100" via `nvidia-smi` *before* importing torch
  and installs `torch==2.4.1+cu121` (includes sm_60). Result: **Device: cuda** on all three grids
  (torchvision/torchaudio version warnings are harmless — unused). CPU fallback still guards the
  no-GPU case.
- **Memory refactor (required for 260 nodes):** replaced materialized `[T-L,N,L,C]` windows (~4 GB at
  260 zones) with vectorized on-the-fly `gather_batch` over a `[T,N,C]` tensor. Also lighter for the
  coarse grids.
- **⭐ Findings (GPU numbers, now canonical):**
  1. **Spatial GCN hurts at EVERY granularity.** `no_spatial` is the best variant on all three grids:
     Chicago R² 0.9603→0.9684, NYC boroughs 0.9805→0.9902, **fine 260-zone 0.9552→0.9657**. The
     finer grid did NOT rescue spatial — the correlation-graph GCN over-smooths at every scale.
  2. **ST-HAE−spatial beats both published ST-GNN baselines on all 3 grids** (Graph WaveNet
     0.9583/0.9811/0.9445; STGCN 0.8570/0.9675/0.9081). The more spatial-conv-heavy the baseline, the
     worse — STGCN weakest — independently corroborating (1).
  3. **RandomForest collapses on the fine grid (R²=0.64)** while every L=24 neural model stays ≥0.94 —
     shows the temporal/sequential modeling is what carries the fine grid.
- **Discrepancy vs E-016 (CPU):** GPU numbers are slightly better (Chicago no_spatial 0.9608→0.9684)
  because GPU training isn't bit-identical to CPU; the qualitative story is unchanged and stronger.
  GPU B=2000 numbers replace the CPU ones as canonical in §5.
- **Honest caveat recorded:** single seed per cell — aggregate R²/RMSE are point estimates (robustness
  columns keep bootstrap CIs); multi-seed variance is remaining work.
- **Learning:** the negative result got *stronger* under more scrutiny (finer grid + two ST-GNN
  baselines that themselves rely on spatial conv and underperform). Chasing a positive "spatial helps"
  result would have meant ignoring three converging pieces of evidence. Best next attempt: learned
  sparse / distance-aware adjacency instead of correlation-thresholded. [[E-016]]

### E-016 · 2026-07-21 · [P3] ⭐ Ablation (both cities, on Kaggle): the spatial GCN HURTS — honest negative result
- **What ran:** pushed a private Kaggle *script kernel* (`kunalchandola/st-hae-training`) via the
  Kaggle API that git-clones the repo and runs `st_hae.py --ablation` on both cities with B=2000
  bootstrap CIs; pulled `results/st_hae_{chicago,nyc}.json`. **Not run on the user's laptop.**
- **Kaggle GPU gotcha (E-015 device fix earned its keep):** the assigned GPU was a **Tesla P100
  (sm_60)**, incompatible with Kaggle's PyTorch build (sm_70+ only) → `cudaErrorNoKernelImage`. v1
  hard-crashed; after adding a device-probe fallback (`resolve_device` tries a tiny kernel, falls
  back to CPU), v2 completed on **Kaggle CPU**. Deterministic, so results are unaffected. (A T4
  would have run on GPU; there's no reliable Kaggle-API field to force the accelerator model.)
- **⭐ The result (replicates across BOTH cities):**
  - **Temporal attention is essential** — `no_temporal` is the *worst* variant (Chicago R²=0.9371,
    *below* RF's 0.9388; NYC 0.9784). The learned temporal encoder does the heavy lifting.
  - **The spatial GCN HURTS** — `no_spatial` is the *best* variant: Chicago R²=0.9608 (RMSE 28.08)
    vs full 0.9541; NYC R²=0.9902 (RMSE 192.3) vs full 0.9829 (RMSE 254.4). Root cause:
    demand-correlation graph over few coarse zones is near-complete (Chicago 88 edges/10 nodes), so
    the 2-layer GCN **over-smooths**, averaging Manhattan's ~4000/hr into tiny boroughs. Dropping it
    even raises tiny-zone R² (NYC Staten Island −0.39→−0.02, Manhattan 0.943→0.968).
  - **MoE head:** small consistent gain (NYC full 0.9829 vs no_hier 0.9801).
- **Recommended model:** ST-HAE−spatial (temporal + MoE) beats RF on both cities and **flattens the
  temporal error swing to 6–7× from RF's 14–18×**. The arc still closes; we just don't over-claim the
  spatial pillar.
- **Trade-off (reported, not hidden):** on NYC `full` has a better high-demand degradation *ratio*
  (+239% vs −spatial's +322%) — GCN smoothing spreads error more evenly at a big aggregate cost.
- **Discrepancy vs E-015 preliminary:** local smoke-test `full` was R²=0.9551/temporal 13.4×; Kaggle
  `full` R²=0.9541/temporal 7.0×. R² is stable; the worst/best-hour *ratio* is noisy and differs with
  torch build (local 2.1.0 vs Kaggle 2.10.0) + BLAS threading. Kaggle B=2000 numbers are canonical.
- **Learning:** the ablation is the point. A clean "our 4-part model wins" would have been *less*
  true and less interesting than "temporal+MoE wins, spatial over-smooths at this granularity" — and
  the negative result hands us a concrete, motivated future-work direction (finer ~260 TLC zones /
  learned sparse adjacency). Keeping eval identical to the baseline (same masked cells) is what makes
  both the win and the negative result defensible.

### E-015 · 2026-07-21 · [P3] ✅ Trained ST-HAE (R² 0.43 → 0.955) beats RF and narrows the gaps
- **What:** rebuilt the untrained prototype as a real end-to-end PyTorch model `src/st_hae.py`
  (+`tests/test_st_hae.py`, 7 green). Four *learned* pillars: trained Kipf GCN over a
  demand-correlation zone graph (no `torch_geometric` — zones are few, dense GCN suffices); trained
  2-layer multi-head temporal self-attention over an L=24 lookback; a learned 3-expert
  mixture-of-experts head (replaces the prototype's data-starving per-quantile split); MoE gate as
  the adaptive ensemble. Masked, leakage-free eval on the *same* observed test cells as the RF
  baseline; per-zone standardized targets; deterministic (seed 42).
- **Preliminary finding (Chicago, one leakage-free run):** ST-HAE **full** RMSE=30.04, R²=0.9551,
  temporal 13.4×, high-dmd +370% — vs RF RMSE=35.10, R²=0.9388, 17.9×, +481%. So the trained model
  not only beats the baseline on aggregate accuracy but **reduces the temporal swing and tail
  degradation the robustness framework surfaced** — the paper's arc closes. (Prototype was R²=0.43;
  the fix was making every component *trained*, exactly as the honest negative-result note predicted.)
- **Design decisions worth recording:**
  - *Per-zone target standardization:* Manhattan (~4000/hr) vs Staten Island (~1/hr) means a raw-MSE
    loss ignores small zones; z-scoring the target per zone gives every zone equal weight in the loss,
    then invert + clip≥0 for eval on raw counts.
  - *Masked loss/metrics:* the processed CSV has no explicit zero cells, so the grid is masked to
    observed cells → y_true matches the RF baseline exactly (fair head-to-head).
  - *Adjacency from train-only demand correlation* (ρ>0.3) to avoid leakage; nan-safe for
    near-constant zones.
- **Process note (IMPORTANT):** the user does **not** want training run on their laptop — training
  moves to **Kaggle GPU**. Added `--device {auto,cuda,mps,cpu}` (auto = cuda>mps>cpu) and a
  self-contained `notebooks/st_hae_kaggle.ipynb` that clones the repo (processed CSVs are committed,
  so nothing to upload) and runs the full ablation on both cities on GPU, writing
  `results/st_hae_{chicago,nyc}.json`. Only a single quick CPU smoke-test was run locally to verify
  correctness (the 0.9551 number); the full ablation + CIs + NYC come from Kaggle.
- **Learning:** the prototype's own honest self-critique was an exact spec for the fix — "trained GCN
  + trained attention + end-to-end" was all it took to flip a −0.55 R² deficit into a +0.016 gain.
  Keeping the eval *identical* to the baseline (same split, same masked cells) is what makes the
  improvement claim defensible rather than an apples-to-oranges artifact.


## Phase 4 — LLM Explainability, Evaluated  🔲 (not started)
## Phase 5 — Synthesis & Release  🔲 (not started)
