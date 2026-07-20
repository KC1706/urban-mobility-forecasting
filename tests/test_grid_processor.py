"""
Tests for the reconstructed grid dataset builder (src/grid_processor.py).

Covers the recovered zone mapping, numeric coercion, the temporal feature flags, and a small
end-to-end aggregation (trip_count + zone assignment) on a synthetic raw file.
"""
import numpy as np
import pandas as pd
import pytest

from grid_processor import (
    ChicagoTaxiGridProcessor, community_area_to_zone, community_area_to_side,
    CHICAGO_SIDES, _to_numeric, RUSH_HOURS, NIGHT_HOURS, OUTPUT_COLUMNS,
)


@pytest.mark.parametrize("ca,zone", [
    (1, "North"), (8, "North"), (9, "Northwest"), (16, "Northwest"),
    (17, "West"), (32, "Southwest"), (57, "Downtown"), (64, "Downtown"),
    (65, "Other"), (77, "Other"),
])
def test_zone_mapping_blocks(ca, zone):
    assert community_area_to_zone(ca) == zone


def test_missing_community_area_is_unknown():
    assert community_area_to_zone(np.nan) == "Unknown"
    assert community_area_to_zone(float("nan")) == "Unknown"


def test_chicago_sides_partition_covers_all_77_areas_once():
    """Real-geography scheme must partition community areas 1..77 exactly once."""
    all_cas = [ca for cas in CHICAGO_SIDES.values() for ca in cas]
    assert sorted(all_cas) == list(range(1, 78))      # every area, exactly once
    assert len(CHICAGO_SIDES) == 9                     # the 9 official sides


@pytest.mark.parametrize("ca,side", [
    (8, "Central"), (32, "Central"), (33, "Central"),   # the Loop / near-downtown
    (76, "Far North"), (1, "Far North"),                # O'Hare + Rogers Park
    (6, "North"), (28, "West"), (49, "Far Southeast"),
])
def test_community_area_to_side_spot_checks(ca, side):
    assert community_area_to_side(ca) == side


def test_side_missing_is_unknown():
    assert community_area_to_side(np.nan) == "Unknown"


def test_flag_hour_sets_match_recovered_definition():
    # Recovered from the historical CSV (ENGINEERING_LOG E-007).
    assert RUSH_HOURS == {7, 8, 9, 17, 18, 19}
    assert NIGHT_HOURS == {0, 1, 2, 3, 4, 5, 22, 23}


def test_to_numeric_strips_currency():
    s = pd.Series(["$6.92", "1,200.50", "", "3.0"])
    out = _to_numeric(s)
    assert out.iloc[0] == pytest.approx(6.92)
    assert out.iloc[1] == pytest.approx(1200.50)
    assert np.isnan(out.iloc[2])
    assert out.iloc[3] == pytest.approx(3.0)


def _write_synthetic_raw(path):
    """Two trips at 08:15 in CA 8 (North) + one at 23:40 in CA 57 (Downtown)."""
    rows = [
        # ts,                    CA, miles, fare,   seconds, lat,     lon
        ("01/05/2026 08:15:00 AM", 8,  2.0, "$10.00", 600, 41.90, -87.63),
        ("01/05/2026 08:45:00 AM", 8,  4.0, "$20.00", 1200, 41.92, -87.65),
        ("01/05/2026 11:40:00 PM", 57, 1.0, "$5.00",  300, 41.88, -87.63),
    ]
    df = pd.DataFrame(rows, columns=[
        "Trip Start Timestamp", "Pickup Community Area", "Trip Miles", "Fare",
        "Trip Seconds", "Pickup Centroid Latitude", "Pickup Centroid Longitude"])
    df.to_csv(path, index=False)


def test_build_aggregates_and_features(tmp_path):
    raw = tmp_path / "raw.csv"
    _write_synthetic_raw(raw)
    df = ChicagoTaxiGridProcessor(raw_path=raw, processed_path=tmp_path / "out.csv").build()

    assert list(df.columns) == OUTPUT_COLUMNS
    # Two North trips collapse into one 08:00 cell; Downtown is its own 23:00 cell.
    north = df[df["pickup_borough"] == "North"].iloc[0]
    assert north["trip_count"] == 2
    assert north["hour"] == 8
    assert north["is_rush_hour"] == 1 and north["is_night"] == 0
    assert north["avg_trip_distance"] == pytest.approx(3.0)   # mean(2,4)
    assert north["avg_fare"] == pytest.approx(15.0)           # mean(10,20)
    assert north["avg_duration"] == pytest.approx(15.0)       # mean(600,1200)/60

    downtown = df[df["pickup_borough"] == "Downtown"].iloc[0]
    assert downtown["trip_count"] == 1
    assert downtown["hour"] == 23
    assert downtown["is_night"] == 1 and downtown["is_rush_hour"] == 0
