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
    build_grid, build_adjacency, STHAE, DYN_CHANNELS, _standardizers,
    prepare_arrays, gather_batch, build_model, ALL_VARIANTS,
)
from st_gnn_baselines import STGCN, GraphWaveNet


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


def test_gather_batch_shapes_and_window_alignment():
    g = build_grid(_toy_df())
    mu, sd, fmu, fsd = _standardizers(g, np.ones(g["T"], bool))
    arr = prepare_arrays(g, 6, mu, sd, fmu, fsd, "cpu")
    hist, cal, y, m = gather_batch(arr, np.array([6, 7, 8, 9]), L=6)
    assert hist.shape == (4, g["N"], 6, len(DYN_CHANNELS))
    assert cal.shape == (4, 7) and y.shape == (4, g["N"]) and m.shape == (4, g["N"])
    # window for target t=6 is rows 0..5 of the normalized dyn tensor
    assert torch.allclose(hist[0].permute(1, 0, 2), arr["dyn"][0:6])


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_forward_pass_shapes_all_variants(variant):
    """Every ST-HAE ablation variant AND both ST-GNN baselines share the forward signature."""
    g = build_grid(_toy_df())
    mu, sd, fmu, fsd = _standardizers(g, np.ones(g["T"], bool))
    L = 12                                              # >= sum of GWN dilations for a valid conv
    arr = prepare_arrays(g, L, mu, sd, fmu, fsd, "cpu")
    hist, cal, y, m = gather_batch(arr, np.array([12, 13, 14, 15]), L=L)
    A = torch.tensor(build_adjacency(g["demand"], g["mask"]))
    model = build_model(variant, g, "cpu")
    out = model(hist, cal, A, torch.arange(g["N"]))
    assert out.shape == (4, g["N"])
    assert torch.isfinite(out).all()


def test_baselines_backward_step_runs():
    """A single forward+backward step on each ST-GNN baseline (grads flow, no crash)."""
    g = build_grid(_toy_df())
    mu, sd, fmu, fsd = _standardizers(g, np.ones(g["T"], bool))
    L = 12
    arr = prepare_arrays(g, L, mu, sd, fmu, fsd, "cpu")
    hist, cal, y, m = gather_batch(arr, np.array([12, 13, 14, 15]), L=L)
    A = torch.tensor(build_adjacency(g["demand"], g["mask"]))
    for Model in (STGCN, GraphWaveNet):
        model = Model(g["N"], len(DYN_CHANNELS), g["cal"].shape[1])
        out = model(hist, cal, A, torch.arange(g["N"]))
        loss = ((out - y)[m] ** 2).mean()
        loss.backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_no_hierarchical_uses_single_expert():
    g = build_grid(_toy_df())
    m = STHAE(g["N"], len(DYN_CHANNELS), 7, use_hierarchical=False)
    assert m.n_experts == 1
    m2 = STHAE(g["N"], len(DYN_CHANNELS), 7, use_hierarchical=True)
    assert m2.n_experts == 3


@pytest.mark.parametrize("adj_mode", ["corr", "adaptive", "adaptive_sparse"])
def test_learned_adjacency_modes_forward_and_shape(adj_mode):
    g = build_grid(_toy_df())
    mu, sd, fmu, fsd = _standardizers(g, np.ones(g["T"], bool))
    arr = prepare_arrays(g, 12, mu, sd, fmu, fsd, "cpu")
    hist, cal, y, m = gather_batch(arr, np.array([12, 13, 14, 15]), L=12)
    A = torch.tensor(build_adjacency(g["demand"], g["mask"]))
    model = STHAE(g["N"], len(DYN_CHANNELS), g["cal"].shape[1], adj_mode=adj_mode, adj_topk=2)
    out = model(hist, cal, A, torch.arange(g["N"]))
    assert out.shape == (4, g["N"]) and torch.isfinite(out).all()
    # learned adjacency should be row-stochastic; sparse keeps <= topk nonzeros per row
    A_used = model._adjacency(A)
    if adj_mode == "adaptive_sparse":
        assert int((A_used[0] > 0).sum()) <= 2


def test_adaptive_modes_have_learned_embeddings():
    g = build_grid(_toy_df())
    plain = dict(STHAE(g["N"], len(DYN_CHANNELS), 7, adj_mode="corr").named_parameters())
    adap = dict(STHAE(g["N"], len(DYN_CHANNELS), 7, adj_mode="adaptive").named_parameters())
    assert "e1" not in plain and "e1" in adap and "e2" in adap


def test_distance_adjacency_degenerate_returns_none():
    from st_hae import build_distance_adjacency
    import numpy as np
    # all-identical coordinates (like the fine grid's borough-level centroids) -> unusable
    assert build_distance_adjacency(np.tile([40.7, -73.9], (6, 1)), k=4) is None
    assert build_distance_adjacency(np.array([[np.nan, 1.0], [2.0, 3.0], [4.0, 5.0]])) is None
