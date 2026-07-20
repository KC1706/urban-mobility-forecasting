"""
Tests for the NYC second-city grid builder (src/nyc_grid_processor.py).

Covers the borough mapping / cleaning bounds / feature flags and a small end-to-end
aggregation on a synthetic parquet file, plus schema-parity with the Chicago builder.
"""
import numpy as np
import pandas as pd
import pytest

from nyc_grid_processor import (
    NYCTaxiGridProcessor, BOROUGH_CENTROIDS, OUTPUT_COLUMNS,
    RUSH_HOURS, NIGHT_HOURS, _file_year_month, MAX_DURATION_MIN,
)
import grid_processor as chi


def test_schema_matches_chicago_builder():
    """NYC output columns must be identical to Chicago's so the pipeline runs unchanged."""
    assert OUTPUT_COLUMNS == chi.OUTPUT_COLUMNS


def test_flag_hour_sets_match_chicago():
    assert RUSH_HOURS == chi.RUSH_HOURS == {7, 8, 9, 17, 18, 19}
    assert NIGHT_HOURS == chi.NIGHT_HOURS == {0, 1, 2, 3, 4, 5, 22, 23}


def test_borough_centroids_cover_the_five_boroughs_plus_ewr():
    for b in ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island", "EWR"]:
        lat, lon = BOROUGH_CENTROIDS[b]
        assert 40.4 < lat < 41.0 and -74.3 < lon < -73.7


@pytest.mark.parametrize("path,ym", [
    ("data/raw/nyc/yellow_tripdata_2024-01.parquet", (2024, 1)),
    ("x/yellow_tripdata_2023-12.parquet", (2023, 12)),
])
def test_file_year_month_parsing(path, ym):
    assert _file_year_month(path) == ym


def test_file_year_month_rejects_bad_name():
    with pytest.raises(ValueError):
        _file_year_month("data/raw/nyc/taxi_zone_lookup.csv")


def _write_synthetic(tmp_path):
    """Zone lookup (2 zones) + one monthly parquet with 3 valid + 3 dirty trips."""
    lk = pd.DataFrame({
        "LocationID": [140, 7, 265],
        "Borough": ["Manhattan", "Queens", "Unknown"],
        "Zone": ["Lenox Hill", "Astoria", "N/A"],
        "service_zone": ["Yellow Zone", "Boro Zone", "N/A"],
    })
    lk_path = tmp_path / "taxi_zone_lookup.csv"
    lk.to_csv(lk_path, index=False)

    ts = pd.Timestamp
    rows = [
        # two valid Manhattan trips in the same 08:00 hour
        (ts("2024-01-05 08:15"), ts("2024-01-05 08:35"), 140, 2.0, 12.0),   # 20 min
        (ts("2024-01-05 08:45"), ts("2024-01-05 09:05"), 140, 4.0, 18.0),   # 20 min
        # one valid Queens trip at 23:00
        (ts("2024-01-05 23:10"), ts("2024-01-05 23:25"), 7,   3.0, 15.0),   # 15 min
        # dirty: out-of-month timestamp (dropped)
        (ts("2002-12-31 23:00"), ts("2002-12-31 23:10"), 140, 1.0, 5.0),
        # dirty: duration > MAX_DURATION_MIN (dropped)
        (ts("2024-01-06 10:00"), ts("2024-01-06 10:00") + pd.Timedelta(minutes=MAX_DURATION_MIN + 1),
         140, 1.0, 5.0),
        # dirty: zero distance (dropped)
        (ts("2024-01-06 11:00"), ts("2024-01-06 11:20"), 140, 0.0, 5.0),
    ]
    df = pd.DataFrame(rows, columns=["tpep_pickup_datetime", "tpep_dropoff_datetime",
                                     "PULocationID", "trip_distance", "fare_amount"])
    pq = tmp_path / "yellow_tripdata_2024-01.parquet"
    df.to_parquet(pq)
    return lk_path


def test_build_aggregates_cleans_and_features(tmp_path):
    lk_path = _write_synthetic(tmp_path)
    proc = NYCTaxiGridProcessor(raw_dir=tmp_path, processed_path=tmp_path / "out.csv",
                                zone_lookup_path=lk_path, zone_scheme="borough")
    df = proc.build()

    assert list(df.columns) == OUTPUT_COLUMNS
    # 3 dirty rows dropped -> only Manhattan@08:00 (2 trips) and Queens@23:00 (1 trip) remain.
    assert set(df["pickup_borough"]) == {"Manhattan", "Queens"}

    man = df[df["pickup_borough"] == "Manhattan"].iloc[0]
    assert man["trip_count"] == 2 and man["hour"] == 8
    assert man["is_rush_hour"] == 1 and man["is_night"] == 0
    assert man["avg_trip_distance"] == pytest.approx(3.0)     # mean(2,4)
    assert man["avg_fare"] == pytest.approx(15.0)             # mean(12,18)
    assert man["avg_duration"] == pytest.approx(20.0)         # both 20 min
    assert man["pickup_latitude"] == pytest.approx(BOROUGH_CENTROIDS["Manhattan"][0])

    q = df[df["pickup_borough"] == "Queens"].iloc[0]
    assert q["trip_count"] == 1 and q["hour"] == 23 and q["is_night"] == 1
