#!/usr/bin/env python3
"""
Grid-level dataset builder for Chicago taxi demand forecasting.

Reconstructs the (zone x hour) demand matrix that the forecasting pipeline consumes
(`data/processed/chicago_taxi_processed.csv`) directly from the raw City of Chicago
"Taxi Trips" export (`data/raw/Taxi_Trips_2026.csv`). The original aggregation script was
missing from the repository; this module was recovered by reverse-engineering the raw->
processed relationship and is verified to reproduce the historical CSV to floating-point
precision (trip_count exact; averaged features to ~1e-14). See docs/ENGINEERING_LOG.md (E-007).

Zone definition (IMPORTANT / honest note):
    The 10 "zones" are NOT real Chicago geography. They are contiguous blocks of 8
    Community-Area *numbers*: zone = ZONE_NAMES[(CommunityArea - 1) // 8], with community
    areas >= 65 bucketed to "Other" and missing community areas to "Unknown". The directional
    names (Downtown, North, ...) are arbitrary labels on numeric blocks. This is documented so
    the paper does not overstate the spatial semantics.

Usage:
    python src/grid_processor.py --build                 # write processed CSV from raw
    python src/grid_processor.py --verify                # rebuild and diff vs existing CSV
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_DEFAULT = Path(__file__).parent.parent / "data" / "raw" / "Taxi_Trips_2026.csv"
PROCESSED_DEFAULT = Path(__file__).parent.parent / "data" / "processed" / "chicago_taxi_processed.csv"

# Directional labels for community-area blocks 1-64 (8 areas each); >=65 -> "Other".
ZONE_NAMES = ["North", "Northwest", "West", "Southwest",
              "South", "Southeast", "FarSouth", "Downtown"]

RUSH_HOURS = {7, 8, 9, 17, 18, 19}
NIGHT_HOURS = {0, 1, 2, 3, 4, 5, 22, 23}

# Final column order (matches the historical processed CSV).
OUTPUT_COLUMNS = [
    "pickup_datetime", "pickup_borough", "trip_count",
    "avg_trip_distance", "avg_fare", "avg_duration",
    "pickup_latitude", "pickup_longitude",
    "hour", "day_of_week", "month", "is_weekend", "is_rush_hour", "is_night",
]

RAW_USECOLS = [
    "Trip Start Timestamp", "Pickup Community Area", "Trip Miles", "Fare",
    "Trip Seconds", "Pickup Centroid Latitude", "Pickup Centroid Longitude",
]


def community_area_to_zone(ca) -> str:
    """Map a Chicago Community Area number to its zone label (see module docstring)."""
    if pd.isna(ca):
        return "Unknown"
    idx = (int(ca) - 1) // 8
    return ZONE_NAMES[idx] if idx < len(ZONE_NAMES) else "Other"


def _to_numeric(series: pd.Series) -> pd.Series:
    """Coerce a raw column to numeric, stripping '$' and thousands separators."""
    return pd.to_numeric(series.astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")


class ChicagoTaxiGridProcessor:
    """Build the (zone x hour) demand matrix from raw Chicago taxi trips."""

    def __init__(self, raw_path: Path = RAW_DEFAULT, processed_path: Path = PROCESSED_DEFAULT):
        self.raw_path = Path(raw_path)
        self.processed_path = Path(processed_path)

    def build(self) -> pd.DataFrame:
        """Aggregate raw trips into the grid dataset and return it (also computes features)."""
        logger.info(f"Reading raw trips: {self.raw_path}")
        raw = pd.read_csv(self.raw_path, usecols=RAW_USECOLS)

        # Hourly bucket + zone label.
        ts = pd.to_datetime(raw["Trip Start Timestamp"], format="%m/%d/%Y %I:%M:%S %p",
                            errors="coerce")
        n_bad = int(ts.isna().sum())
        if n_bad:
            logger.warning(f"Dropping {n_bad} trips with unparseable Trip Start Timestamp")
        raw = raw.loc[ts.notna()].copy()
        raw["pickup_datetime"] = ts[ts.notna()].dt.floor("h")
        raw["pickup_borough"] = raw["Pickup Community Area"].map(community_area_to_zone)

        for col in ["Trip Miles", "Fare", "Trip Seconds",
                    "Pickup Centroid Latitude", "Pickup Centroid Longitude"]:
            raw[col] = _to_numeric(raw[col])

        logger.info("Aggregating to (pickup_datetime, zone) cells...")
        g = raw.groupby(["pickup_datetime", "pickup_borough"])
        df = g.agg(
            trip_count=("Trip Start Timestamp", "size"),
            avg_trip_distance=("Trip Miles", "mean"),
            avg_fare=("Fare", "mean"),
            avg_duration_sec=("Trip Seconds", "mean"),
            pickup_latitude=("Pickup Centroid Latitude", "mean"),
            pickup_longitude=("Pickup Centroid Longitude", "mean"),
        ).reset_index()
        df["avg_duration"] = df["avg_duration_sec"] / 60.0  # original stores minutes
        df = df.drop(columns=["avg_duration_sec"])

        # Temporal features derived from pickup_datetime.
        dt = df["pickup_datetime"]
        df["hour"] = dt.dt.hour
        df["day_of_week"] = dt.dt.dayofweek           # Monday = 0
        df["month"] = dt.dt.month
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_rush_hour"] = df["hour"].isin(RUSH_HOURS).astype(int)
        df["is_night"] = df["hour"].isin(NIGHT_HOURS).astype(int)

        df = df.sort_values(["pickup_datetime", "pickup_borough"]).reset_index(drop=True)
        df = df[OUTPUT_COLUMNS]
        logger.info(f"Built grid dataset: {df.shape[0]} rows x {df.shape[1]} cols, "
                    f"{df['pickup_borough'].nunique()} zones, {df['trip_count'].sum()} trips")
        return df

    def write(self, df: pd.DataFrame = None) -> Path:
        """Build (if needed) and write the processed CSV."""
        if df is None:
            df = self.build()
        self.processed_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.processed_path, index=False)
        logger.info(f"Wrote {self.processed_path}")
        return self.processed_path

    def verify(self, reference_path: Path = None) -> bool:
        """Rebuild and compare against an existing processed CSV via (datetime, zone) keys."""
        reference_path = Path(reference_path or self.processed_path)
        built = self.build()
        ref = pd.read_csv(reference_path)
        ref["pickup_datetime"] = pd.to_datetime(ref["pickup_datetime"], errors="coerce")
        ref_clean = ref.dropna(subset=["pickup_datetime"])
        n_nat = len(ref) - len(ref_clean)

        keys = ["pickup_datetime", "pickup_borough"]
        m = ref_clean.merge(built, on=keys, how="outer", suffixes=("_ref", "_new"), indicator=True)
        only_ref = int((m["_merge"] == "left_only").sum())
        only_new = int((m["_merge"] == "right_only").sum())
        both = m[m["_merge"] == "both"]

        logger.info(f"Reference rows: {len(ref)} ({n_nat} with corrupt timestamps, excluded)")
        logger.info(f"Keys only in reference: {only_ref} | only in rebuild: {only_new}")

        ok = (only_ref == 0)
        tc_mismatch = int((both["trip_count_ref"].astype(int) != both["trip_count_new"].astype(int)).sum())
        logger.info(f"trip_count mismatches on shared keys: {tc_mismatch}")
        ok = ok and tc_mismatch == 0
        for col in ["avg_trip_distance", "avg_fare", "avg_duration",
                    "pickup_latitude", "pickup_longitude"]:
            delta = (both[f"{col}_ref"] - both[f"{col}_new"]).abs().max()
            logger.info(f"  {col:18s} max|delta| = {delta:.3g}")
            ok = ok and (pd.isna(delta) or delta < 1e-6)
        logger.info(f"VERIFY {'PASSED' if ok else 'FAILED'} "
                    f"(rebuild reproduces reference to <1e-6; corrupt-timestamp rows excluded)")
        return ok


def main():
    ap = argparse.ArgumentParser(description="Build/verify the Chicago taxi grid dataset")
    ap.add_argument("--raw", default=str(RAW_DEFAULT))
    ap.add_argument("--out", default=str(PROCESSED_DEFAULT))
    ap.add_argument("--build", action="store_true", help="write processed CSV from raw")
    ap.add_argument("--verify", action="store_true", help="rebuild and diff vs existing CSV")
    args = ap.parse_args()

    proc = ChicagoTaxiGridProcessor(args.raw, args.out)
    if args.verify:
        raise SystemExit(0 if proc.verify() else 1)
    if args.build:
        proc.write()
    if not (args.build or args.verify):
        ap.print_help()


if __name__ == "__main__":
    main()
