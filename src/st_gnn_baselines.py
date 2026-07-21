#!/usr/bin/env python3
"""
Published spatio-temporal GNN baselines for the ST-HAE comparison (Phase 3).

Compact but faithful re-implementations of the two canonical grid-demand baselines, sharing the
ST-HAE forward signature (hist[B,N,L,C], cal[B,Ccal], A[N,N], zone_ids[N]) -> [B,N] so the SAME
leakage-free, masked training/eval harness (`st_hae.py`) scores them on the identical test cells.

  - STGCN (Yu, Yin, Zhu, IJCAI 2018): "sandwich" ST-conv blocks — a temporal gated (GLU) 1-D conv,
    a spatial graph convolution over the (fixed, Kipf-normalized) zone adjacency, then another
    temporal gated conv.
  - Graph WaveNet (Wu et al., IJCAI 2019): dilated causal (WaveNet) gated temporal convolutions
    with graph diffusion over BOTH the fixed adjacency and a *learned adaptive* adjacency
    (self-adaptive node embeddings), plus residual connections.

Calendar features are broadcast across nodes/time and concatenated as extra input channels, so the
baselines receive exactly the same information as ST-HAE. Both models mean-pool over the remaining
temporal axis before a small output head (robust to the exact post-conv sequence length).
"""
import torch
import torch.nn as nn


def _with_calendar(hist, cal):
    """[B,N,L,C] + [B,Ccal] -> [B, C+Ccal, N, L] (channels-first for Conv2d over (N,L))."""
    B, N, L, C = hist.shape
    cal_bc = cal[:, None, None, :].expand(B, N, L, cal.shape[-1])
    x = torch.cat([hist, cal_bc], dim=-1)            # [B,N,L,Cin]
    return x.permute(0, 3, 1, 2).contiguous()        # [B,Cin,N,L]


# --------------------------------------------------------------------------------------
# STGCN
# --------------------------------------------------------------------------------------
class TemporalGatedConv(nn.Module):
    """Gated 1-D conv along time (GLU), applied per node. x:[B,C,N,L] -> [B,Cout,N,L-kt+1]."""
    def __init__(self, c_in, c_out, kt=3):
        super().__init__()
        self.conv = nn.Conv2d(c_in, 2 * c_out, kernel_size=(1, kt))

    def forward(self, x):
        a, b = self.conv(x).chunk(2, dim=1)
        return a * torch.sigmoid(b)


class SpatialGraphConv(nn.Module):
    """Graph conv over nodes with a fixed adjacency. x:[B,C,N,L] -> [B,Cout,N,L]."""
    def __init__(self, c_in, c_out):
        super().__init__()
        self.lin = nn.Linear(c_in, c_out)

    def forward(self, x, A):
        h = torch.einsum("nm,bcml->bcnl", A, x)                     # mix over nodes
        h = self.lin(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)    # channel mixing
        return torch.relu(h)


class STGCN(nn.Module):
    def __init__(self, n_zones, c_dyn, c_cal, c_hidden=32, kt=3):
        super().__init__()
        c_in = c_dyn + c_cal
        self.t1 = TemporalGatedConv(c_in, c_hidden, kt)
        self.s1 = SpatialGraphConv(c_hidden, c_hidden)
        self.t2 = TemporalGatedConv(c_hidden, c_hidden, kt)
        self.bn = nn.BatchNorm2d(c_hidden)
        self.head = nn.Linear(c_hidden, 1)

    def forward(self, hist, cal, A, zone_ids):
        x = _with_calendar(hist, cal)                # [B,Cin,N,L]
        x = self.t1(x)
        x = self.s1(x, A)
        x = self.bn(self.t2(x))                       # [B,Ch,N,L'']
        x = x.mean(dim=-1).permute(0, 2, 1)          # avg over time -> [B,N,Ch]
        return self.head(x).squeeze(-1)              # [B,N]


# --------------------------------------------------------------------------------------
# Graph WaveNet
# --------------------------------------------------------------------------------------
class GraphWaveNet(nn.Module):
    def __init__(self, n_zones, c_dyn, c_cal, d=32, dilations=(1, 2, 4), n_emb=10):
        super().__init__()
        c_in = c_dyn + c_cal
        self.start = nn.Conv2d(c_in, d, kernel_size=(1, 1))
        # self-adaptive adjacency via learned node embeddings
        self.e1 = nn.Parameter(torch.randn(n_zones, n_emb) * 0.05)
        self.e2 = nn.Parameter(torch.randn(n_zones, n_emb) * 0.05)
        self.filters, self.gates, self.gconv = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for dil in dilations:
            self.filters.append(nn.Conv2d(d, d, (1, 2), dilation=(1, dil)))
            self.gates.append(nn.Conv2d(d, d, (1, 2), dilation=(1, dil)))
            self.gconv.append(nn.Linear(2 * d, d))   # diffusion over [fixed A | adaptive A]
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, hist, cal, A, zone_ids):
        x = self.start(_with_calendar(hist, cal))                     # [B,d,N,L]
        A_adp = torch.softmax(torch.relu(self.e1 @ self.e2.t()), dim=1)  # [N,N] learned
        for f, g, gc in zip(self.filters, self.gates, self.gconv):
            res = x
            h = torch.tanh(f(x)) * torch.sigmoid(g(x))                # gated dilated conv [B,d,N,L']
            h1 = torch.einsum("nm,bdml->bdnl", A, h)                  # diffusion over fixed adj
            h2 = torch.einsum("nm,bdml->bdnl", A_adp, h)             # diffusion over adaptive adj
            hcat = torch.cat([h1, h2], dim=1)                        # [B,2d,N,L']
            hmix = gc(hcat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # [B,d,N,L']
            x = hmix + res[..., -hmix.shape[-1]:]                     # residual (align time)
        x = x.mean(dim=-1).permute(0, 2, 1)                          # [B,N,d]
        return self.head(x).squeeze(-1)                              # [B,N]
