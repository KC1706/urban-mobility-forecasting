#!/usr/bin/env python3
"""
Generate the paper's figures from the committed results JSONs (Phase 5).

Pure post-processing — reads results/*.json and writes paper/figures/*.png. No training, no API.

    python src/make_figures.py
"""
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
RES = ROOT / "results"
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

CITY = {"chicago": "Chicago (9 sides)", "nyc": "NYC (6 boroughs)"}
BLUE, RED, GREEN, GRAY = "#2b6cb0", "#c53030", "#2f855a", "#718096"


def _load(name):
    p = RES / name
    return json.load(open(p)) if p.exists() else None


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote paper/figures/{name}")


# 1. Conformal calibration collapse (headline robustness figure)
def fig_conformal():
    data = {c: _load(f"{'chicago_sides' if c == 'chicago' else 'nyc'}_robustness_ci.json")
            for c in CITY}
    if any(v is None for v in data.values()):
        return
    order = ["overall", "off_peak", "peak_hours", "high_demand(>=p95)"]
    labels = ["Overall", "Off-peak", "Peak hours", "High demand\n(≥p95)"]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(order)); w = 0.38
    for i, (c, o) in enumerate(data.items()):
        strata = o["conformal_coverage"]["strata"]
        cov = [100 * strata.get(k, {}).get("coverage", np.nan) for k in order]
        ax.bar(x + (i - 0.5) * w, cov, w, label=CITY[c], color=[BLUE, RED][i], alpha=0.85)
    ax.axhline(90, ls="--", color="black", lw=1)
    ax.text(len(order) - 0.5, 91.5, "nominal 90%", ha="right", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Empirical coverage (%)"); ax.set_ylim(0, 100)
    ax.set_title("Split-conformal calibration collapse on high-demand events")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig1_conformal_collapse.png")


# 2. Per-zone R² forest plot with 95% CIs
def fig_per_zone():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (c, fn) in zip(axes, [("chicago", "chicago_sides"), ("nyc", "nyc")]):
        o = _load(f"{fn}_robustness_ci.json")
        if o is None:
            continue
        pz = o["per_zone_r2_ci"]
        items = sorted(pz.items(), key=lambda kv: kv[1]["point"])
        names = [k for k, _ in items]
        pts = [v["point"] for _, v in items]
        lo = [v["point"] - v["lo"] for _, v in items]
        hi = [v["hi"] - v["point"] for _, v in items]
        y = np.arange(len(names))
        cols = [RED if p < 0 else BLUE for p in pts]
        ax.errorbar(pts, y, xerr=[lo, hi], fmt="none", ecolor=GRAY, capsize=3, lw=1)
        ax.scatter(pts, y, color=cols, zorder=3)
        ax.axvline(0, ls="--", color="black", lw=1)
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Per-zone R² (95% bootstrap CI)")
        ax.set_title(CITY[c]); ax.grid(axis="x", alpha=0.3)
        ax.set_xlim(min(-1.5, min(v["lo"] for v in pz.values()) * 1.05), 1.05)
    fig.suptitle("Aggregate R² hides per-zone heterogeneity (small zones unmeasurable)")
    _save(fig, "fig2_per_zone_r2.png")


# 3. Model comparison across the three grids (R²)
def fig_models():
    grids = [("chicago", "st_hae_chicago.json", "Chicago sides"),
             ("nyc", "st_hae_nyc.json", "NYC boroughs"),
             ("nyc_zones", "st_hae_nyc_zones.json", "NYC 260 zones")]
    variants = ["no_spatial", "full", "no_temporal", "no_hierarchical", "stgcn", "gwn"]
    vlabel = {"no_spatial": "ST-HAE−spatial", "full": "ST-HAE full", "no_temporal": "−temporal",
              "no_hierarchical": "−hier", "stgcn": "STGCN", "gwn": "GraphWaveNet"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for ax, (key, fn, title) in zip(axes, grids):
        o = _load(fn)
        if o is None:
            continue
        names, r2s, cols = [], [], []
        rf = o["random_forest"]["overall"]["r2"]
        names.append("RandomForest"); r2s.append(rf); cols.append(GRAY)
        for v in variants:
            if v in o["models"]:
                names.append(vlabel[v]); r2s.append(o["models"][v]["overall"]["r2"])
                cols.append(GREEN if v == "no_spatial" else BLUE)
        y = np.arange(len(names))
        ax.barh(y, r2s, color=cols, alpha=0.85)
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        lo = min(r2s); ax.set_xlim(max(0, lo - 0.05), 1.0)
        for yi, r in zip(y, r2s):
            ax.text(r, yi, f" {r:.3f}", va="center", fontsize=7)
        ax.set_title(title); ax.set_xlabel("R²"); ax.grid(axis="x", alpha=0.3)
    fig.suptitle("ST-HAE−spatial (green) beats RF, STGCN & Graph WaveNet on all three grids")
    _save(fig, "fig3_model_comparison.png")


# 4. Adjacency rescue + 5-seed variance (R² mean±std)
def fig_adjacency():
    order = ["no_spatial", "full", "full_distance", "full_adaptive", "full_adaptive_sparse"]
    lab = {"no_spatial": "−spatial", "full": "corr graph", "full_distance": "distance",
           "full_adaptive": "learned dense", "full_adaptive_sparse": "learned sparse"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, c in zip(axes, ["chicago", "nyc"]):
        o = _load(f"st_hae_{c}_adj.json")
        if o is None:
            continue
        names, mean, std, cols = [], [], [], []
        for v in order:
            if v in o["models"] and "r2_mean" in o["models"][v]:
                names.append(lab[v]); mean.append(o["models"][v]["r2_mean"])
                std.append(o["models"][v]["r2_std"])
                cols.append(GREEN if v == "no_spatial" else BLUE)
        x = np.arange(len(names))
        ax.bar(x, mean, yerr=std, capsize=4, color=cols, alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("R² (mean ± std, 5 seeds)")
        lo = min(m - s for m, s in zip(mean, std))
        ax.set_ylim(lo - 0.01, max(m + s for m, s in zip(mean, std)) + 0.005)
        ax.set_title(CITY[c]); ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Learned sparse adjacency > correlation graph, but −spatial (green) still wins/ties")
    _save(fig, "fig4_adjacency_multiseed.png")


# 5. LLM faithfulness (Groq / Llama-3.3-70B)
def fig_faithfulness():
    fig, ax = plt.subplots(figsize=(7.5, 4))
    metrics = [("faithfulness_mean", "Faithfulness"), ("directional_accuracy_mean", "Directional acc."),
               ("top3_driver_recall_mean", "Top-3 recall"), ("hallucination_rate_mean", "Hallucination")]
    x = np.arange(len(metrics)); w = 0.38
    got = False
    for i, c in enumerate(["chicago", "nyc"]):
        o = _load(f"faithfulness_{c}.json")
        if o is None or "groq" not in o["providers"] or "faithfulness_mean" not in o["providers"]["groq"]:
            continue
        g = o["providers"]["groq"]; got = True
        vals = [g.get(k, np.nan) for k, _ in metrics]
        err = [g["faithfulness_std"] if k == "faithfulness_mean" else 0 for k, _ in metrics]
        ax.bar(x + (i - 0.5) * w, vals, w, yerr=err, capsize=4, label=CITY[c],
               color=[BLUE, RED][i], alpha=0.85)
    if not got:
        plt.close(fig); return
    ax.set_xticks(x); ax.set_xticklabels([m[1] for m in metrics])
    ax.set_ylabel("Score (0–1)"); ax.set_ylim(0, 1)
    ax.set_title("LLM explanation faithfulness — Llama-3.3-70B (5 runs)\n"
                 "high on scale drivers, misses counter-intuitive ones, hallucinates peak_hour (NYC)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig5_faithfulness.png")


def main():
    for f in (fig_conformal, fig_per_zone, fig_models, fig_adjacency, fig_faithfulness):
        try:
            f()
        except Exception as e:                        # noqa: BLE001
            logger.info(f"[skip] {f.__name__}: {type(e).__name__}: {e}")
    logger.info(f"\nFigures in {FIG}")


if __name__ == "__main__":
    main()
