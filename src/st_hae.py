#!/usr/bin/env python3
"""
ST-HAE: Spatial-Temporal Hierarchical Attention Ensemble  (Phase 3 — the REAL model).

The prototype in `st_hae_algorithm.py` documented the intended architecture but left every
component UNTRAINED (fixed-identity "GCN", unlearned "attention", data-starving per-quantile
sub-models) and honestly underperformed the baselines (R²≈0.43). This module is the trained,
end-to-end PyTorch re-implementation that keeps the four conceptual pillars but makes each one
*learned*:

  1. Spatial  — a trained Kipf-normalized Graph Convolution over zones (adjacency built from
                training-set demand correlation). Zones are few (≤ a few dozen), so a dense GCN
                needs no torch_geometric.
  2. Temporal — a trained multi-head self-attention encoder over the L-hour lookback window
                (learned Q/K/V projections), not raw-feature dot products.
  3. Hierarchical — a learned mixture-of-experts head (K experts + a softmax gate). Unlike the
                prototype's per-quantile split, ALL data trains ALL experts end-to-end; the gate
                learns the demand-regime specialization.
  4. Ensemble — the MoE gate IS the adaptive combination; trained jointly with everything else.

Evaluation is leakage-free (chronological split via `splits.py`), masked to the cells actually
observed in the processed CSV (so y_true is identical to the RandomForest baseline in §7), and
reported with the same robustness dimensions + bootstrap CIs as `robustness_ci.py`. An `--ablation`
mode drops one pillar at a time. NOTHING here claims to beat the baselines unless the numbers say
so — the model is a contribution *and* a stress-test subject.

Usage:
    python src/st_hae.py --data data/processed/chicago_taxi_sides.csv --out results/st_hae_chicago.json
    python src/st_hae.py --data data/processed/nyc_taxi_boroughs.csv  --out results/st_hae_nyc.json --ablation
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).parent))
from splits import temporal_split_indices  # noqa: E402
import robustness_ci as rci  # noqa: E402
from st_gnn_baselines import STGCN, GraphWaveNet  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SEED = 42
DYN_CHANNELS = ["trip_count", "avg_fare", "avg_trip_distance", "avg_duration"]


def _seed_everything(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _device_executes(dev):
    """True iff `dev` can actually launch a kernel (guards against e.g. Kaggle's P100 sm_60
    being incompatible with a PyTorch build compiled only for sm_70+ -> cudaErrorNoKernelImage)."""
    try:
        _ = (torch.zeros(2, device=dev) + 1).sum().item()
        return True
    except Exception as e:                       # noqa: BLE001 — any accelerator failure -> fall back
        logger.info(f"  device {dev!r} unusable ({type(e).__name__}: {str(e)[:80]}); falling back")
        return False


def resolve_device(name="auto"):
    """Resolve to a device that actually works. 'auto' tries cuda>mps>cpu; an explicit accelerator
    is honored only if it can launch a kernel, else falls back to cpu (never hard-fails)."""
    candidates = []
    if name == "auto":
        if torch.cuda.is_available():
            candidates.append("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            candidates.append("mps")
    elif name != "cpu":
        candidates.append(name)
    for dev in candidates:
        if _device_executes(dev):
            return dev
    return "cpu"


# --------------------------------------------------------------------------------------
# Data: long CSV -> dense (time x zone) grid tensors, masked to observed cells.
# --------------------------------------------------------------------------------------
def build_grid(df, dt_col="pickup_datetime", zone_col="pickup_borough"):
    """Return the regular (T x N) grid: times, zones, demand, dyn features, mask, calendar."""
    df = df.copy()
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.dropna(subset=[dt_col])
    times = pd.DatetimeIndex(sorted(df[dt_col].unique()))
    zones = sorted(df[zone_col].unique())
    t_idx = {t: i for i, t in enumerate(times)}
    z_idx = {z: j for j, z in enumerate(zones)}
    T, N = len(times), len(zones)

    demand = np.zeros((T, N), np.float32)
    dyn = np.zeros((T, N, len(DYN_CHANNELS)), np.float32)
    mask = np.zeros((T, N), bool)
    ti = df[dt_col].map(t_idx).to_numpy()
    zj = df[zone_col].map(z_idx).to_numpy()
    mask[ti, zj] = True
    for c, col in enumerate(DYN_CHANNELS):
        dyn[ti, zj, c] = np.nan_to_num(df[col].to_numpy(dtype=np.float32))
    demand[:] = dyn[:, :, 0]

    # Calendar features known at prediction time (cyclical hour/day + flags).
    hour = times.hour.to_numpy(); dow = times.dayofweek.to_numpy()
    cal = np.stack([
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
        (dow >= 5).astype(np.float32),
        np.isin(hour, [7, 8, 9, 17, 18, 19]).astype(np.float32),
        np.isin(hour, [0, 1, 2, 3, 4, 5, 22, 23]).astype(np.float32),
    ], axis=1).astype(np.float32)
    return {"times": times, "zones": zones, "demand": demand, "dyn": dyn,
            "mask": mask, "cal": cal, "T": T, "N": N}


def build_adjacency(demand_train, mask_train):
    """Kipf-normalized adjacency from train-set per-zone demand correlation (+ self loops)."""
    N = demand_train.shape[1]
    # Pearson correlation across time between zone demand series (rows w/ any obs).
    d = demand_train.copy()
    d[~mask_train] = np.nan
    with np.errstate(invalid="ignore"):
        corr = pd.DataFrame(d).corr().to_numpy()
    corr = np.nan_to_num(corr, nan=0.0)
    A = (corr > 0.3).astype(np.float32)          # edge where demand correlation > 0.3
    return _kipf_normalize(A)


def _kipf_normalize(A):
    """D^-1/2 (A+I) D^-1/2 with self-loops added."""
    N = A.shape[0]
    A = A.astype(np.float32).copy()
    np.fill_diagonal(A, 0.0)
    A = A + np.eye(N, dtype=np.float32)
    deg = A.sum(1)
    dinv = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-8)))
    return (dinv @ A @ dinv).astype(np.float32)


def zone_centroids(df, grid, zone_col="pickup_borough"):
    """Per-zone mean (lat, lon) in grid-zone order; NaN row if a zone has no coordinates."""
    m = (df.groupby(zone_col)[["pickup_latitude", "pickup_longitude"]].mean()
         .reindex(grid["zones"]))
    return m.to_numpy(dtype=np.float64)


def build_distance_adjacency(centroids, k=4):
    """Sparse geographic adjacency: k-nearest-neighbour graph on zone centroids (great-circle-ish
    Euclidean on lat/lon), Gaussian-weighted, symmetrized, Kipf-normalized. Returns None if
    centroids are unusable (missing, or degenerate — e.g. the fine grid where all zones in a borough
    share a borough centroid, giving < k distinct points)."""
    C = np.asarray(centroids, float)
    if np.isnan(C).any():
        return None
    N = len(C)
    n_distinct = len({tuple(np.round(c, 6)) for c in C})
    if N < 3 or n_distinct <= k:                 # too few / degenerate coordinates
        return None
    d2 = ((C[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    sigma2 = np.median(d2[np.isfinite(d2)]) + 1e-9
    A = np.zeros((N, N), np.float32)
    for i in range(N):                           # keep k nearest neighbours per node
        nn_idx = np.argsort(d2[i])[:k]
        A[i, nn_idx] = np.exp(-d2[i, nn_idx] / sigma2)
    A = np.maximum(A, A.T)                        # symmetrize
    return _kipf_normalize(A)


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
class GCNLayer(nn.Module):
    """Dense Kipf graph conv: H' = A_hat @ (H W)."""
    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)

    def forward(self, H, A):                      # H:[B,N,d]  A:[N,N]
        return torch.einsum("nm,bmd->bnd", A, self.lin(H))


def _renorm(A, eps=1e-8):
    """Row-normalize a non-negative adjacency (so GCN mixing is a convex combination per node)."""
    return A / (A.sum(-1, keepdim=True) + eps)


class STHAE(nn.Module):
    """adj_mode controls the spatial GCN's graph (only used when use_spatial=True):
        'corr'/'distance' -> use the fixed A passed in (built in run(): demand-correlation or
                             centroid-distance kNN);
        'adaptive'        -> a LEARNED dense adjacency from node embeddings (Graph-WaveNet style);
        'adaptive_sparse' -> the learned adjacency, top-k per row then renormalized (learned SPARSE).
    """
    def __init__(self, n_zones, c_dyn, c_cal, d_model=64, n_heads=4, n_experts=3,
                 use_spatial=True, use_temporal=True, use_hierarchical=True,
                 adj_mode="corr", adj_emb=10, adj_topk=8):
        super().__init__()
        self.use_spatial, self.use_temporal, self.use_hierarchical = \
            use_spatial, use_temporal, use_hierarchical
        self.adj_mode, self.adj_topk = adj_mode, adj_topk
        self.in_proj = nn.Linear(c_dyn, d_model)
        # (1) temporal self-attention encoder over the lookback window
        enc = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=2 * d_model,
                                         batch_first=True, dropout=0.1)
        self.temporal = nn.TransformerEncoder(enc, num_layers=2)
        # (2) spatial GCN over zones (+ learned adjacency embeddings for adaptive modes)
        self.gcn1 = GCNLayer(d_model, d_model)
        self.gcn2 = GCNLayer(d_model, d_model)
        if adj_mode in ("adaptive", "adaptive_sparse"):
            self.e1 = nn.Parameter(torch.randn(n_zones, adj_emb) * 0.05)
            self.e2 = nn.Parameter(torch.randn(n_zones, adj_emb) * 0.05)
        self.zone_emb = nn.Embedding(n_zones, 16)
        self.cal_proj = nn.Linear(c_cal, 16)
        fuse_dim = d_model + 16 + 16
        # (3) hierarchical mixture-of-experts head
        self.n_experts = n_experts if use_hierarchical else 1
        self.gate = nn.Linear(fuse_dim, self.n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(fuse_dim, d_model), nn.ReLU(), nn.Linear(d_model, 1))
            for _ in range(self.n_experts)])

    def _adjacency(self, A_fixed):
        """Resolve the adjacency actually used by the GCN for this forward pass."""
        if self.adj_mode not in ("adaptive", "adaptive_sparse"):
            return A_fixed                                    # fixed: corr / distance
        A = torch.softmax(torch.relu(self.e1 @ self.e2.t()), dim=1)   # learned dense [N,N]
        if self.adj_mode == "adaptive_sparse":
            k = min(self.adj_topk, A.shape[-1])
            val, idx = A.topk(k, dim=-1)
            A = _renorm(torch.zeros_like(A).scatter(-1, idx, val))    # learned top-k SPARSE
        return A

    def forward(self, hist, cal, A, zone_ids):
        # hist:[B,N,L,C]  cal:[B,Ccal]  A:[N,N]  zone_ids:[N]
        B, N, L, C = hist.shape
        h = self.in_proj(hist).reshape(B * N, L, -1)          # [B*N, L, d]
        if self.use_temporal:
            h = self.temporal(h)[:, -1, :]                    # trained attention -> last step
        else:
            h = h.mean(1)                                     # ablation: mean-pool over lags
        H = h.reshape(B, N, -1)                               # [B,N,d]
        if self.use_spatial:
            A_use = self._adjacency(A)
            H = torch.relu(self.gcn1(H, A_use))
            H = torch.relu(self.gcn2(H, A_use))               # trained GCN
        z = self.zone_emb(zone_ids).unsqueeze(0).expand(B, -1, -1)   # [B,N,16]
        c = self.cal_proj(cal).unsqueeze(1).expand(-1, N, -1)        # [B,N,16]
        fuse = torch.cat([H, z, c], dim=-1)                          # [B,N,fuse]
        ex = torch.cat([e(fuse) for e in self.experts], dim=-1)      # [B,N,K]
        if self.n_experts == 1:
            return ex.squeeze(-1)
        g = torch.softmax(self.gate(fuse), dim=-1)                   # [B,N,K] learned gate
        return (g * ex).sum(-1)                                      # [B,N]


# --------------------------------------------------------------------------------------
# Sample construction, training, evaluation
# --------------------------------------------------------------------------------------
def _split_time_masks(df, grid, dt_col="pickup_datetime"):
    """Map each grid timestamp to train/val/test using the same chronological split."""
    idx = temporal_split_indices(df.assign(**{dt_col: pd.to_datetime(df[dt_col], errors="coerce")}),
                                 dt_col, ratios=(0.7, 0.15, 0.15))
    times = grid["times"]
    split_of_time = {}
    for name in ("train", "val", "test"):
        for t in pd.to_datetime(df.iloc[idx[name]][dt_col].unique()):
            split_of_time[t] = name
    arr = np.array([split_of_time.get(t, "train") for t in times])
    return arr


def _standardizers(grid, train_t):
    """Per-zone demand mean/std (from train observed cells) + global feature stats."""
    d, m = grid["demand"], grid["mask"]
    N = grid["N"]
    mu = np.zeros(N, np.float32); sd = np.ones(N, np.float32)
    dtrain, mtrain = d[train_t], m[train_t]
    for j in range(N):
        obs = dtrain[:, j][mtrain[:, j]]
        if obs.size:
            mu[j] = obs.mean(); sd[j] = max(obs.std(), 1e-3)
    dyn = grid["dyn"][train_t][mtrain]                          # [n_obs, C]
    fmu = dyn.mean(0); fsd = np.maximum(dyn.std(0), 1e-6)
    return mu, sd, fmu.astype(np.float32), fsd.astype(np.float32)


# ST-HAE variant -> (use_spatial, use_temporal, use_hierarchical, adj_mode).
STHAE_SPEC = {
    "full":                 (1, 1, 1, "corr"),       # spatial GCN on demand-correlation graph
    "no_spatial":           (0, 1, 1, "corr"),
    "no_temporal":          (1, 0, 1, "corr"),
    "no_hierarchical":      (1, 1, 0, "corr"),
    "full_distance":        (1, 1, 1, "distance"),        # geographic kNN adjacency
    "full_adaptive":        (1, 1, 1, "adaptive"),        # learned dense adjacency
    "full_adaptive_sparse": (1, 1, 1, "adaptive_sparse"), # learned top-k SPARSE adjacency
}
ALL_VARIANTS = list(STHAE_SPEC) + ["stgcn", "gwn"]


def variant_adj_mode(variant):
    """Which fixed adjacency (if any) a variant consumes: 'corr', 'distance', or None (learned)."""
    if variant in STHAE_SPEC:
        mode = STHAE_SPEC[variant][3]
        return mode if mode in ("corr", "distance") else None
    return "corr"                                    # baselines use the fixed corr adjacency


def build_model(variant, grid, device):
    """Model factory. ST-HAE ablation/adjacency variants + published ST-GNN baselines share the
    forward signature (hist[B,N,L,C], cal[B,Ccal], A[N,N], zone_ids[N]) -> [B,N], so the identical
    training/eval harness gives a fair, apples-to-apples comparison on the same masked test cells."""
    N, c_dyn, c_cal = grid["N"], len(DYN_CHANNELS), grid["cal"].shape[1]
    if variant in STHAE_SPEC:
        s, t, h, adj = STHAE_SPEC[variant]
        return STHAE(N, c_dyn, c_cal, use_spatial=bool(s), use_temporal=bool(t),
                     use_hierarchical=bool(h), adj_mode=adj).to(device)
    if variant == "stgcn":
        return STGCN(N, c_dyn, c_cal).to(device)
    if variant == "gwn":
        return GraphWaveNet(N, c_dyn, c_cal).to(device)
    raise ValueError(f"unknown variant {variant!r}")


def prepare_arrays(grid, L, mu, sd, fmu, fsd, device):
    """Normalized dynamic tensor [T,N,C] + targets/mask/calendar on `device`. History windows are
    gathered on the fly per batch (see gather_batch) so we never materialize [T-L,N,L,C] — that
    would be multiple GB at the ~260-zone TLC grid. mu/sd are per-zone; fmu/fsd are per-channel."""
    dyn = grid["dyn"].astype(np.float32).copy()
    dyn = (dyn - fmu) / fsd
    dyn[:, :, 0] = (grid["demand"] - mu) / sd                    # demand channel per-zone z-score
    dyn[~grid["mask"]] = 0.0                                     # missing -> mean (0 after z-score)
    y = ((grid["demand"] - mu) / sd).astype(np.float32)
    return {"dyn": torch.tensor(dyn, device=device),            # [T,N,C]
            "y": torch.tensor(y, device=device),               # [T,N] normalized target
            "mask": torch.tensor(grid["mask"], device=device), # [T,N] bool
            "cal": torch.tensor(grid["cal"], device=device),   # [T,Ccal]
            "y_raw": grid["demand"].astype(np.float32),        # [T,N] numpy raw counts
            "mu": mu, "sd": sd}


def gather_batch(arr, idx, L):
    """Vectorized window gather. idx: absolute target time indices (each >= L).
    Returns hist[B,N,L,C], cal[B,Ccal], y[B,N], mask[B,N]."""
    dev = arr["dyn"].device
    t = torch.as_tensor(np.asarray(idx), device=dev, dtype=torch.long)
    win = t.unsqueeze(1) - L + torch.arange(L, device=dev).unsqueeze(0)   # [B,L] = t-L .. t-1
    hist = arr["dyn"][win].permute(0, 2, 1, 3).contiguous()              # [B,L,N,C]->[B,N,L,C]
    return hist, arr["cal"][t], arr["y"][t], arr["mask"][t]


def _batches(idx, bs=64, shuffle=True):
    idx = np.asarray(idx).copy()
    if shuffle:
        np.random.shuffle(idx)
    for k in range(0, len(idx), bs):
        yield idx[k:k + bs]


def _predict(model, arr, A_t, zone_ids, idx, L, bs=128):
    """Predict normalized demand for target indices `idx` (in order). Returns [len(idx), N]."""
    model.eval()
    outs = []
    with torch.no_grad():
        for b in _batches(idx, bs, shuffle=False):
            hist, cal, _, _ = gather_batch(arr, b, L)
            outs.append(model(hist, cal, A_t, zone_ids).cpu().numpy())
    return np.concatenate(outs, axis=0)


def train_variant(grid, A, splits, L, variant, device="cpu", epochs=120, patience=18, lr=1e-3,
                  bs=64, seed=SEED):
    _seed_everything(seed)
    mu, sd, fmu, fsd = _standardizers(grid, splits == "train")
    arr = prepare_arrays(grid, L, mu, sd, fmu, fsd, device)
    targets = np.arange(L, grid["T"])
    tsplit = splits[targets]
    tr, va, te = (targets[tsplit == s] for s in ("train", "val", "test"))

    A_t = torch.tensor(A, device=device)
    zone_ids = torch.arange(grid["N"], device=device)
    model = build_model(variant, grid, device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    def val_rmse():
        vp = np.clip(_predict(model, arr, A_t, zone_ids, va, L) * sd + mu, 0, None)
        mvE = grid["mask"][va]
        e = (vp - arr["y_raw"][va])[mvE]
        return float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan")

    best_val, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        for b in _batches(tr, bs):
            opt.zero_grad()
            hist, cal, yb, mb = gather_batch(arr, b, L)
            pred = model(hist, cal, A_t, zone_ids)
            loss = ((pred - yb)[mb] ** 2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        vrmse = val_rmse()
        if vrmse < best_val - 1e-4:
            best_val = vrmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    pred_te = np.clip(_predict(model, arr, A_t, zone_ids, te, L) * sd + mu, 0, None)  # [len_te,N]

    # Flatten observed test cells -> paired arrays for metrics/robustness (matches RF's cells).
    mte = grid["mask"][te]
    yt = arr["y_raw"][te][mte]
    yp = pred_te[mte]
    hours = np.repeat(grid["times"][te].hour.to_numpy()[:, None], grid["N"], axis=1)[mte]
    zj = np.repeat(np.arange(grid["N"])[None, :], len(te), axis=0)[mte]
    zones = np.array(grid["zones"])[zj]
    return {"y_true": yt, "y_pred": yp, "hours": hours, "zones": zones,
            "n_params": int(sum(p.numel() for p in model.parameters())),
            "epochs_trained": ep + 1, "best_val_rmse": best_val}


def robustness_summary(y_true, y_pred, hours, zones, B=2000):
    ci = lambda d: [round(d["point"], 3), round(d["lo"], 3), round(d["hi"], 3)]
    overall = {"rmse": rci._rmse(y_true, y_pred), "r2": rci._r2(y_true, y_pred),
               "mae": float(np.mean(np.abs(y_true - y_pred))), "n_test": int(len(y_true))}
    temporal = rci.bootstrap_temporal_ratio(y_true, y_pred, hours, B=B)
    high = rci.bootstrap_high_demand_degradation(y_true, y_pred, y_true, B=B)
    per_zone = {}
    for z in np.unique(zones):
        mm = zones == z
        if mm.sum() >= 10:
            per_zone[str(z)] = ci(rci.bootstrap_ci(y_true[mm], y_pred[mm], rci._r2, B=B))
    return {"overall": overall, "temporal_ratio_ci": ci(temporal),
            "high_demand_degradation_ci": ci(high), "per_zone_r2_ci": per_zone}


def _overall(y_true, y_pred):
    return {"rmse": rci._rmse(y_true, y_pred), "r2": rci._r2(y_true, y_pred),
            "mae": float(np.mean(np.abs(y_true - y_pred))), "n_test": int(len(y_true))}


def run(data_path, out=None, variants=None, L=24, device="auto", B=2000, seeds=(SEED,)):
    if variants is None:
        variants = ["full"]
    seeds = list(seeds)
    _seed_everything()
    device = resolve_device(device)
    logger.info(f"Device: {device} | seeds={seeds}")
    df = pd.read_csv(data_path)
    grid = build_grid(df)
    splits = _split_time_masks(df, grid)
    A_corr = build_adjacency(grid["demand"][splits == "train"], grid["mask"][splits == "train"])
    A_dist = build_distance_adjacency(zone_centroids(df, grid))
    adjacencies = {"corr": A_corr, "distance": A_dist, None: A_corr}  # None (learned) ignores it
    logger.info(f"Grid: T={grid['T']} x N={grid['N']} zones; lookback L={L}; "
                f"corr-edges={int((A_corr > 0).sum() - grid['N'])}; "
                f"distance-adj={'built' if A_dist is not None else 'unavailable (degenerate centroids)'}")

    results = {}
    for v in variants:
        adj = variant_adj_mode(v)
        if adj == "distance" and adjacencies["distance"] is None:
            logger.info(f"\n=== Skipping [{v}] — no usable geographic centroids on this grid ===")
            continue
        A = adjacencies[adj]
        logger.info(f"\n=== Training [{v}] over {len(seeds)} seed(s) ===")
        per_seed, ref = [], None
        for si, sd_ in enumerate(seeds):
            r = train_variant(grid, A, splits, L, v, device=device, seed=sd_)
            ov = _overall(r["y_true"], r["y_pred"])
            per_seed.append(ov)
            logger.info(f"    seed {sd_}: RMSE={ov['rmse']:.2f} R²={ov['r2']:.4f} "
                        f"MAE={ov['mae']:.2f} ({r['epochs_trained']} ep)")
            if si == 0:
                ref = r                                   # robustness CIs computed on first seed
        summ = robustness_summary(ref["y_true"], ref["y_pred"], ref["hours"], ref["zones"], B=B)
        summ.update({"n_params": ref["n_params"], "n_seeds": len(seeds), "per_seed_overall": per_seed})
        if len(seeds) > 1:                                # mean ± std of the headline metrics
            for k in ("rmse", "r2", "mae"):
                vals = np.array([s[k] for s in per_seed], float)
                summ[f"{k}_mean"], summ[f"{k}_std"] = float(vals.mean()), float(vals.std(ddof=1))
        results[v] = summ
        o = summ["overall"]
        extra = (f" | R²={summ['r2_mean']:.4f}±{summ['r2_std']:.4f} over {len(seeds)} seeds"
                 if len(seeds) > 1 else "")
        logger.info(f"  {v:20s} RMSE={o['rmse']:.2f} R²={o['r2']:.4f}{extra}")

    # RandomForest baseline on the identical split/test cells (from robustness_ci).
    logger.info("\n=== RandomForest baseline (same split) ===")
    rf = rci.run_analysis(data_path, B=B)
    out_obj = {"data": str(data_path), "lookback_L": L, "device": device,
               "random_forest": {"overall": rf["overall"],
                                 "temporal_ratio_ci": [rf["temporal_ratio_ci"]["point"],
                                                       rf["temporal_ratio_ci"]["lo"],
                                                       rf["temporal_ratio_ci"]["hi"]],
                                 "high_demand_degradation_ci": [rf["high_demand_degradation_ci"]["point"],
                                                                rf["high_demand_degradation_ci"]["lo"],
                                                                rf["high_demand_degradation_ci"]["hi"]]},
               "models": results}
    _print_table(out_obj)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out_obj, open(out, "w"), indent=2, default=float)
        logger.info(f"\nWrote {out}")
    return out_obj


def _print_table(o):
    print("\n" + "=" * 78)
    print(f"MODELS vs BASELINE  ({Path(o['data']).name}, leakage-free test)")
    print("=" * 78)
    print(f"{'Model':22s} {'RMSE':>8s} {'R²':>8s} {'R²(mean±std)':>18s} {'temporal':>9s} {'high-dmd':>9s}")
    rf = o["random_forest"]
    print(f"{'RandomForest':22s} {rf['overall']['rmse']:8.2f} {rf['overall']['r2']:8.4f} "
          f"{'—':>18s} {rf['temporal_ratio_ci'][0]:8.1f}x {rf['high_demand_degradation_ci'][0]:8.0f}%")
    for v, s in o["models"].items():
        ov = s["overall"]
        ms = (f"{s['r2_mean']:.4f}±{s['r2_std']:.4f}" if "r2_mean" in s else "—")
        print(f"{v:22s} {ov['rmse']:8.2f} {ov['r2']:8.4f} {ms:>18s} "
              f"{s['temporal_ratio_ci'][0]:8.1f}x {s['high_demand_degradation_ci'][0]:8.0f}%")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ablation", action="store_true",
                    help="ST-HAE leave-one-out: full + no_spatial/no_temporal/no_hierarchical")
    ap.add_argument("--baselines", action="store_true", help="also train STGCN + Graph WaveNet")
    ap.add_argument("--variants", default=None,
                    help="comma list overriding --ablation/--baselines "
                         f"(any of: {','.join(ALL_VARIANTS)})")
    ap.add_argument("--adjacency", action="store_true",
                    help="spatial-rescue attempt: full + no_spatial + full_distance/adaptive/adaptive_sparse")
    ap.add_argument("--lookback", type=int, default=24)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                    help="'auto' picks cuda>mps>cpu (use 'cuda' on Kaggle GPU)")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=1,
                    help="train each variant over this many seeds (42..42+n-1) for mean±std")
    args = ap.parse_args()
    if args.variants:
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    else:
        variants = ["full"]
        if args.ablation:
            variants += ["no_spatial", "no_temporal", "no_hierarchical"]
        if args.adjacency:
            variants += ["no_spatial", "full_distance", "full_adaptive", "full_adaptive_sparse"]
        if args.baselines:
            variants += ["stgcn", "gwn"]
    variants = list(dict.fromkeys(variants))                 # de-dup, keep order
    run(args.data, out=args.out, variants=variants, L=args.lookback, device=args.device,
        B=args.B, seeds=list(range(SEED, SEED + args.seeds)))


if __name__ == "__main__":
    main()
