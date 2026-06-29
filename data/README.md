# Data

## Processed data (included)
- `processed/chicago_taxi_processed.csv` — 7,147 hourly-aggregated demand records
  across 10 zones with 14 engineered features. This is the model-ready dataset.

## Raw data (not tracked — too large for git, ~204 MB)
The raw Chicago taxi trip records are excluded from version control. Download them
yourself and place under `data/raw/`.

- **Chicago Taxi Trips**: https://data.cityofchicago.org/Transportation/Taxi-Trips-2013-2023-/wrvz-psew
- **NYC TLC Trip Records**: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

After downloading the raw CSV to `data/raw/Taxi_Trips_2026.csv`, regenerate the
processed dataset with:

```bash
python src/data_processor.py
```

This aggregates raw trips to hourly per-zone demand and writes
`data/processed/chicago_taxi_processed.csv`.
