"""
Tests for the shared temporal splitter (src/splits.py).

Guards the two properties the old random split violated: (1) no temporal leakage
(all train times precede all test times), and (2) returned indices actually address the
right rows of the original DataFrame.
"""
import numpy as np
import pandas as pd
import pytest

from splits import temporal_split, temporal_split_indices


@pytest.fixture
def grid_df():
    # 100 hourly timestamps x 3 zones = 300 rows, intentionally shuffled.
    times = pd.date_range("2026-01-01", periods=100, freq="h")
    rows = [{"pickup_datetime": t, "pickup_borough": z, "trip_count": i}
            for i, t in enumerate(times) for z in ["A", "B", "C"]]
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=0).reset_index(drop=True)
    return df


def test_no_temporal_leakage(grid_df):
    tr, va, te = temporal_split(grid_df, ratios=(0.7, 0.15, 0.15))
    tmax_tr = pd.to_datetime(tr["pickup_datetime"]).max()
    tmin_va = pd.to_datetime(va["pickup_datetime"]).min()
    tmax_va = pd.to_datetime(va["pickup_datetime"]).max()
    tmin_te = pd.to_datetime(te["pickup_datetime"]).min()
    assert tmax_tr < tmin_va      # train strictly before val
    assert tmax_va < tmin_te      # val strictly before test


def test_partition_is_complete_and_disjoint(grid_df):
    idx = temporal_split_indices(grid_df)
    allidx = np.concatenate([idx["train"], idx["val"], idx["test"]])
    assert len(allidx) == len(grid_df)              # covers every row
    assert len(np.unique(allidx)) == len(grid_df)   # no row in two splits


def test_indices_address_correct_rows(grid_df):
    """The positional indices must map back to rows with the expected timestamps."""
    idx = temporal_split_indices(grid_df)
    test_times = pd.to_datetime(grid_df.iloc[idx["test"]]["pickup_datetime"])
    train_times = pd.to_datetime(grid_df.iloc[idx["train"]]["pickup_datetime"])
    assert train_times.max() < test_times.min()


def test_same_timestamp_never_straddles_splits(grid_df):
    tr, va, te = temporal_split(grid_df)
    s_tr = set(pd.to_datetime(tr["pickup_datetime"]))
    s_te = set(pd.to_datetime(te["pickup_datetime"]))
    assert s_tr.isdisjoint(s_te)  # a given hour's zones stay together


def test_ratios_respected_approximately(grid_df):
    idx = temporal_split_indices(grid_df, ratios=(0.7, 0.15, 0.15))
    frac_train = len(idx["train"]) / len(grid_df)
    assert frac_train == pytest.approx(0.70, abs=0.02)


def test_bad_ratios_raise(grid_df):
    with pytest.raises(ValueError):
        temporal_split_indices(grid_df, ratios=(0.6, 0.2, 0.1))  # sums to 0.9


def test_unparseable_datetime_raises():
    df = pd.DataFrame({"pickup_datetime": ["2026-01-01", "not-a-date"], "trip_count": [1, 2]})
    with pytest.raises(ValueError):
        temporal_split_indices(df)
