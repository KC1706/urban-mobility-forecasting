"""
Fast unit tests for the trained ST-HAE (src/st_hae.py) — grid construction, adjacency
normalization, and a single forward pass with the ablation flags. No full training
(kept fast); the end-to-end training path is exercised by the CLI on real data.
"""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from st_hae import (
    build_grid, build_adjacency, STHAE, DYN_CHANNELS, _make_windows, _standardizers,
)


def _toy_df():
    """3 zones over 60 hours; zone C only appears intermittently (missing cells)."""
    times = pd.date_range("2024-01-01", periods=60, freq="h")
    rows = []
    for i, t in enumerate(times):
        for z in ["A", "B"]:
            rows.append((t, z, 10 + i % 5, 12.0, 2.0, 15.0))
        if i % 3 == 0:                       # zone C sparse
            rows.append((t, "C", 1, 5.0, 1.0, 8.0))
    return pd.DataFrame(rows, columns=["pickup_datetime", "pickup_borough", "trip_count",
                                       "avg_fare", "avg_trip_distance", "avg_duration"])


def test_build_grid_shapes_and_mask():
    g = build_grid(_toy_df())
    assert g["T"] == 60 and g["N"] == 3
    assert g["demand"].shape == (60, 3) and g["dyn"].shape == (60, 3, len(DYN_CHANNELS))
    assert g["cal"].shape == (60, 7)
    # zone C observed only every 3rd hour -> ~20 observed cells, rest masked False
    zc = g["zones"].index("C")
    assert g["mask"][:, zc].sum() == 20
    # demand channel of dyn equals the demand grid
    assert np.allclose(g["dyn"][:, :, 0], g["demand"])


def test_build_adjacency_is_symmetric_normalized_with_self_loops():
    g = build_grid(_toy_df())
    train = np.ones(g["T"], bool)
    A = build_adjacency(g["demand"][train], g["mask"][train])
    assert A.shape == (3, 3)
    assert np.allclose(A, A.T, atol=1e-6)          # symmetric
    assert np.all(np.isfinite(A))                  # no nan from constant/near-constant zones
    assert np.all(np.diag(A) > 0)                  # self loops present


@pytest.mark.parametrize("flags", [(1, 1, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0)])
def test_forward_pass_shapes_all_ablations(flags):
    g = build_grid(_toy_df())
    mu, sd, fmu, fsd = _standardizers(g, np.ones(g["T"], bool))
    hist, cal, y, m, y_raw = _make_windows(g, L=6, mu=mu, sd=sd, fmu=fmu, fsd=fsd)
    A = torch.tensor(build_adjacency(g["demand"], g["mask"]))
    model = STHAE(g["N"], len(DYN_CHANNELS), g["cal"].shape[1],
                  use_spatial=bool(flags[0]), use_temporal=bool(flags[1]),
                  use_hierarchical=bool(flags[2]))
    b = torch.arange(4)
    out = model(hist[b], cal[b], A, torch.arange(g["N"]))
    assert out.shape == (4, g["N"])
    assert torch.isfinite(out).all()


def test_no_hierarchical_uses_single_expert():
    g = build_grid(_toy_df())
    m = STHAE(g["N"], len(DYN_CHANNELS), 7, use_hierarchical=False)
    assert m.n_experts == 1
    m2 = STHAE(g["N"], len(DYN_CHANNELS), 7, use_hierarchical=True)
    assert m2.n_experts == 3
