#!/usr/bin/env python3
"""
Shared chronological train/val/test splitting for the forecasting pipeline.

Why this exists (see docs/ENGINEERING_LOG.md E-008/E-009):
  The original pipeline used sklearn's random `train_test_split` for the tree/MLP baselines,
  which (a) leaks future information into training for a time-series task, inflating reported
  accuracy, and (b) then set `test_indices = np.arange(len(X_test))`, mis-joining predictions
  to rows for the robustness analysis. Both are fixed by splitting on time and returning the
  true positional indices of each split.

All splits are cut on the ordered *unique* timestamps, so a given hour never appears in two
splits (no same-timestamp leakage across zones). By construction:
    max(train time) <= min(val time) <= min(test time).
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def temporal_split_indices(
    df: pd.DataFrame,
    datetime_col: str = "pickup_datetime",
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> Dict[str, np.ndarray]:
    """
    Chronological split returning positional indices into `df` (as-is order).

    Args:
        df: dataset with a datetime column (need not be pre-sorted).
        datetime_col: name of the timestamp column used for ordering.
        ratios: (train, val, test) fractions of the *unique timestamps*; must sum to ~1.

    Returns:
        {"train": idx, "val": idx, "test": idx} where each idx is an int array of positions
        into `df` such that all train timestamps precede all val, which precede all test.
    """
    if datetime_col not in df.columns:
        raise KeyError(f"datetime_col '{datetime_col}' not in DataFrame")
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"ratios must sum to 1.0, got {ratios} (sum={sum(ratios)})")

    times = pd.to_datetime(df[datetime_col], errors="coerce")
    if times.isna().any():
        raise ValueError(f"{int(times.isna().sum())} rows have unparseable {datetime_col}; "
                         "clean them before splitting")

    unique_times = np.sort(times.unique())
    n = len(unique_times)
    r_train, r_val, _ = ratios
    i_train = int(n * r_train)
    i_val = int(n * (r_train + r_val))

    train_end = unique_times[max(i_train - 1, 0)]
    val_end = unique_times[max(i_val - 1, 0)]

    t = times.values
    train_mask = t <= train_end
    val_mask = (t > train_end) & (t <= val_end)
    test_mask = t > val_end

    return {
        "train": np.where(train_mask)[0],
        "val": np.where(val_mask)[0],
        "test": np.where(test_mask)[0],
    }


def temporal_split(
    df: pd.DataFrame,
    datetime_col: str = "pickup_datetime",
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split returning (train_df, val_df, test_df) DataFrame views."""
    idx = temporal_split_indices(df, datetime_col, ratios)
    return df.iloc[idx["train"]].copy(), df.iloc[idx["val"]].copy(), df.iloc[idx["test"]].copy()


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else \
        str((__import__("pathlib").Path(__file__).parent.parent /
             "data" / "processed" / "chicago_taxi_processed.csv"))
    d = pd.read_csv(path)
    d = d.dropna(subset=["pickup_datetime"])
    tr, va, te = temporal_split(d)
    for name, part in [("train", tr), ("val", va), ("test", te)]:
        t = pd.to_datetime(part["pickup_datetime"])
        print(f"{name:5s} rows={len(part):5d}  {t.min()} -> {t.max()}  zones={part['pickup_borough'].nunique()}")
