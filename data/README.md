# Data

Two cities, one identical `(zone × hour)` schema (14 engineered features), so the same
forecasting + robustness pipeline runs on both. See `docs/PAPER.md` §3.

## Processed data (included, model-ready)
- `processed/chicago_taxi_processed.csv` — Chicago, legacy synthetic "blocks" zones (reproduces
  the historical CSV).
- `processed/chicago_taxi_sides.csv` — Chicago, **real 9 official "sides"** (headline Chicago
  dataset). 7,182 hourly cells, Jan 1 – Feb 1 2026.
- `processed/nyc_taxi_boroughs.csv` — **NYC yellow taxi, Jan–Jun 2024**, 6 NYC boroughs.
  22,346 hourly cells, 19.66M trips (second city / longer window).
- `processed/nyc_taxi_zones.csv.gz` — **NYC fine grid**: the ~260 official TLC zones (gzipped,
  491,238 hourly cells). Used for the ST-HAE finer-grid spatial-GCN retry (§5). `pandas` reads the
  `.gz` directly; rebuild the uncompressed CSV with `--zone-scheme zone` (below).

## Raw data (not tracked — too large for git)
Excluded from version control (`data/raw/` + `*.parquet` are gitignored). Download and place under
`data/raw/`.

### Chicago (~204 MB CSV)
- Source: City of Chicago "Taxi Trips" — https://data.cityofchicago.org/Transportation/Taxi-Trips
- Place at `data/raw/Taxi_Trips_2026.csv`, then build:
  ```bash
  python src/grid_processor.py --build --zone-scheme sides --out data/processed/chicago_taxi_sides.csv
  python src/grid_processor.py --verify        # reproduces the historical CSV to ~1e-14
  ```

### NYC (~340 MB, 6 monthly parquet files)
- Source: NYC TLC "Yellow Taxi Trip Records" — https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- Download the zone lookup + monthly parquet into `data/raw/nyc/`:
  ```bash
  base=https://d37ci6vzurychx.cloudfront.net
  curl -o data/raw/nyc/taxi_zone_lookup.csv "$base/misc/taxi_zone_lookup.csv"
  for m in 01 02 03 04 05 06; do
    curl -o data/raw/nyc/yellow_tripdata_2024-$m.parquet "$base/trip-data/yellow_tripdata_2024-$m.parquet"
  done
  ```
- Build (requires `pyarrow`):
  ```bash
  python src/nyc_grid_processor.py --build      # -> data/processed/nyc_taxi_boroughs.csv
  # finer ~260-TLC-zone grid:
  python src/nyc_grid_processor.py --build --zone-scheme zone --out data/processed/nyc_taxi_zones.csv
  ```

## Reproduce the headline robustness results
```bash
python src/robustness_ci.py --data data/processed/chicago_taxi_sides.csv --out results/chicago_sides_robustness_ci.json
python src/robustness_ci.py --data data/processed/nyc_taxi_boroughs.csv  --out results/nyc_robustness_ci.json
```
