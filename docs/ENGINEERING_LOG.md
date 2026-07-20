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


## Phase 2 — Robustness Layer, Made Rigorous  🔲 (not started)
## Phase 3 — ST-HAE: The Real Model  🔲 (not started)
## Phase 4 — LLM Explainability, Evaluated  🔲 (not started)
## Phase 5 — Synthesis & Release  🔲 (not started)
