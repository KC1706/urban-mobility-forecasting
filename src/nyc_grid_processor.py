#!/usr/bin/env python3
"""
Grid-level dataset builder for NEW YORK CITY taxi demand forecasting (second city).

Produces a (zone x hour) demand matrix with the SAME schema as the Chicago builder
(`src/grid_processor.py` / `OUTPUT_COLUMNS`), so the identical forecasting + robustness
pipeline runs unchanged on a second city. Source: NYC TLC "Yellow Taxi Trip Records"
monthly parquet files (`data/raw/nyc/yellow_tripdata_YYYY-MM.parquet`) +
`taxi_zone_lookup.csv` (LocationID -> Borough / Zone).

Zone definition (honest note):
    Default scheme 'borough' maps each trip's pickup TLC zone (PULocationID) to its NYC
    borough (Manhattan / Brooklyn / Queens / Bronx / Staten Island / EWR). This is REAL
    geography (unlike Chicago's legacy synthetic "blocks"), and is the natural analogue of
    Chicago's 9 official "sides": a small number of real, contiguous regions. The finer
    'zone' scheme keys on PULocationID directly (~260 TLC zones) for a fine spatial grid.

Coordinates: the NYC parquet stores only zone IDs (no lat/lon), so pickup_latitude/longitude
are filled with the pickup borough's approximate centroid. Documented as approximate; they
feed the same spatial feature slot the Chicago centroids occupy.

Data cleaning (each monthly file, documented for the paper):
    - keep only trips whose pickup timestamp falls in the file's nominal (year, month)
      [the raw files contain a tail of corrupt out-of-range timestamps, e.g. year 2002];
    - trip_distance in (0, 100] miles;
    - fare_amount in [0, 500] dollars;
    - trip duration (dropoff - pickup) in (0, 180] minutes;
    - PULocationID present in the zone lookup (drops 264/265 = "Unknown"/"N/A" -> "Unknown").

Usage:
    python src/nyc_grid_processor.py --build          # write data/processed/nyc_taxi_boroughs.csv
    python src/nyc_grid_processor.py --build --zone-scheme zone --out data/processed/nyc_taxi_zones.csv
"""

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_DIR_DEFAULT = Path(__file__).parent.parent / "data" / "raw" / "nyc"
PROCESSED_DEFAULT = Path(__file__).parent.parent / "data" / "processed" / "nyc_taxi_boroughs.csv"
ZONE_LOOKUP_DEFAULT = RAW_DIR_DEFAULT / "taxi_zone_lookup.csv"

# Approximate borough centroids (lat, lon). NYC parquet has no per-trip coordinates, so the
# spatial feature slot is filled with the pickup borough's centroid.
BOROUGH_CENTROIDS = {
    "Manhattan":     (40.7831, -73.9712),
    "Brooklyn":      (40.6782, -73.9442),
    "Queens":        (40.7282, -73.7949),
    "Bronx":         (40.8448, -73.8648),
    "Staten Island": (40.5795, -74.1502),
    "EWR":           (40.6895, -74.1745),
    "Unknown":       (np.nan,  np.nan),
}

RUSH_HOURS = {7, 8, 9, 17, 18, 19}
NIGHT_HOURS = {0, 1, 2, 3, 4, 5, 22, 23}

# Same column order as the Chicago processed CSV (grid_processor.OUTPUT_COLUMNS).
OUTPUT_COLUMNS = [
    "pickup_datetime", "pickup_borough", "trip_count",
    "avg_trip_distance", "avg_fare", "avg_duration",
    "pickup_latitude", "pickup_longitude",
    "hour", "day_of_week", "month", "is_weekend", "is_rush_hour", "is_night",
]

# Cleaning bounds (see module docstring).
MAX_DISTANCE_MI = 100.0
MAX_FARE = 500.0
MAX_DURATION_MIN = 180.0

_MONTH_RE = re.compile(r"yellow_tripdata_(\d{4})-(\d{2})\.parquet$")


def load_zone_lookup(path: Path = ZONE_LOOKUP_DEFAULT) -> pd.DataFrame:
    """Return the TLC zone lookup (LocationID -> Borough, Zone)."""
    lk = pd.read_csv(path)
    lk.columns = [c.strip() for c in lk.columns]
    return lk


def _file_year_month(path: Path):
    m = _MONTH_RE.search(str(path))
    if not m:
        raise ValueError(f"cannot parse year-month from {path!r}")
    return int(m.group(1)), int(m.group(2))


class NYCTaxiGridProcessor:
    """Build the (zone x hour) demand matrix from raw NYC yellow-taxi parquet files."""

    def __init__(self, raw_dir: Path = RAW_DIR_DEFAULT, processed_path: Path = PROCESSED_DEFAULT,
                 zone_lookup_path: Path = ZONE_LOOKUP_DEFAULT, zone_scheme: str = "borough"):
        if zone_scheme not in ("borough", "zone"):
            raise ValueError(f"zone_scheme must be 'borough' or 'zone', got {zone_scheme!r}")
        self.raw_dir = Path(raw_dir)
        self.processed_path = Path(processed_path)
        self.zone_lookup_path = Path(zone_lookup_path)
        self.zone_scheme = zone_scheme

    def _clean_month(self, path: Path, loc_to_borough: dict, loc_to_zone: dict) -> pd.DataFrame:
        """Read one monthly parquet, clean it, and return trip-level rows with a zone label."""
        year, month = _file_year_month(path)
        cols = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID",
                "trip_distance", "fare_amount"]
        df = pd.read_parquet(path, columns=cols)
        n0 = len(df)

        pu = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
        do = pd.to_datetime(df["tpep_dropoff_datetime"], errors="coerce")
        duration_min = (do - pu).dt.total_seconds() / 60.0

        keep = (
            pu.notna()
            & (pu.dt.year == year) & (pu.dt.month == month)   # drop corrupt out-of-range stamps
            & df["trip_distance"].between(0, MAX_DISTANCE_MI, inclusive="right")
            & df["fare_amount"].between(0, MAX_FARE, inclusive="both")
            & duration_min.between(0, MAX_DURATION_MIN, inclusive="right")
            & df["PULocationID"].isin(loc_to_borough)
        )
        out = pd.DataFrame({
            "pickup_datetime": pu[keep].dt.floor("h"),
            "PULocationID": df["PULocationID"][keep].astype(int),
            "trip_distance": df["trip_distance"][keep].astype(float),
            "fare_amount": df["fare_amount"][keep].astype(float),
            "duration_min": duration_min[keep].astype(float),
        })
        if self.zone_scheme == "borough":
            out["pickup_borough"] = out["PULocationID"].map(loc_to_borough).fillna("Unknown")
        else:
            out["pickup_borough"] = out["PULocationID"].map(loc_to_zone).fillna("Unknown")
        logger.info(f"  {path.name}: {n0} -> {len(out)} trips kept "
                    f"({100 * (n0 - len(out)) / max(n0, 1):.1f}% dropped)")
        return out

    def build(self) -> pd.DataFrame:
        """Aggregate all monthly parquet files into the grid dataset (with features)."""
        files = sorted(self.raw_dir.glob("yellow_tripdata_*.parquet"))
        if not files:
            raise FileNotFoundError(f"no yellow_tripdata_*.parquet under {self.raw_dir}")
        lk = load_zone_lookup(self.zone_lookup_path)
        loc_to_borough = dict(zip(lk["LocationID"], lk["Borough"]))
        loc_to_zone = dict(zip(lk["LocationID"], lk["Zone"]))
        # "Unknown"/"N/A" boroughs (LocationID 264/265) collapse to a single "Unknown" label.
        loc_to_borough = {k: ("Unknown" if v in ("Unknown", "N/A") else v)
                          for k, v in loc_to_borough.items()}

        logger.info(f"Reading {len(files)} monthly NYC parquet files from {self.raw_dir}")
        trips = pd.concat([self._clean_month(f, loc_to_borough, loc_to_zone) for f in files],
                          ignore_index=True)

        logger.info("Aggregating to (pickup_datetime, zone) cells...")
        g = trips.groupby(["pickup_datetime", "pickup_borough"])
        df = g.agg(
            trip_count=("PULocationID", "size"),
            avg_trip_distance=("trip_distance", "mean"),
            avg_fare=("fare_amount", "mean"),
            avg_duration=("duration_min", "mean"),
        ).reset_index()

        # Spatial centroids: borough centroid (borough scheme) or the zone's borough centroid.
        if self.zone_scheme == "borough":
            cent = df["pickup_borough"].map(BOROUGH_CENTROIDS)
        else:
            # map zone label -> its borough -> centroid (best-effort; NaN if unknown)
            zone_to_borough = dict(zip(lk["Zone"], lk["Borough"]))
            cent = df["pickup_borough"].map(
                lambda z: BOROUGH_CENTROIDS.get(zone_to_borough.get(z, "Unknown"),
                                                (np.nan, np.nan)))
        df["pickup_latitude"] = [c[0] for c in cent]
        df["pickup_longitude"] = [c[1] for c in cent]

        dt = df["pickup_datetime"]
        df["hour"] = dt.dt.hour
        df["day_of_week"] = dt.dt.dayofweek
        df["month"] = dt.dt.month
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_rush_hour"] = df["hour"].isin(RUSH_HOURS).astype(int)
        df["is_night"] = df["hour"].isin(NIGHT_HOURS).astype(int)

        df = df.sort_values(["pickup_datetime", "pickup_borough"]).reset_index(drop=True)
        df = df[OUTPUT_COLUMNS]
        logger.info(f"Built NYC grid dataset: {df.shape[0]} rows x {df.shape[1]} cols, "
                    f"{df['pickup_borough'].nunique()} zones, {int(df['trip_count'].sum())} trips, "
                    f"{dt.min()} -> {dt.max()}")
        return df

    def write(self, df: pd.DataFrame = None) -> Path:
        if df is None:
            df = self.build()
        self.processed_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.processed_path, index=False)
        logger.info(f"Wrote {self.processed_path}")
        return self.processed_path


def main():
    ap = argparse.ArgumentParser(description="Build the NYC taxi grid dataset (second city)")
    ap.add_argument("--raw-dir", default=str(RAW_DIR_DEFAULT))
    ap.add_argument("--out", default=str(PROCESSED_DEFAULT))
    ap.add_argument("--zone-lookup", default=str(ZONE_LOOKUP_DEFAULT))
    ap.add_argument("--zone-scheme", choices=["borough", "zone"], default="borough",
                    help="'borough' = 6 NYC boroughs (analogue of Chicago sides); "
                         "'zone' = ~260 TLC zones (fine grid)")
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()

    proc = NYCTaxiGridProcessor(args.raw_dir, args.out, args.zone_lookup, args.zone_scheme)
    if args.build:
        proc.write()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
