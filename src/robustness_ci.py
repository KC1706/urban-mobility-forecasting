#!/usr/bin/env python3
"""
Statistical rigor for the robustness findings (Phase 2).

Turns the point-estimate robustness story into defensible claims:
  1. Bootstrap confidence intervals on the headline metrics — per-zone R² (incl. the
     negative-R² zone), the temporal worst/best-hour RMSE ratio, and the high-demand
     RMSE degradation.
  2. Split-conformal prediction intervals with per-stratum empirical coverage — showing that
     a single global interval, calibrated to nominal coverage overall, systematically
     UNDER-covers in the operationally hard strata (peak hours, high demand, hot zones).

Everything runs on the leakage-free chronological split (src/splits.py) and the real-geography
"sides" dataset. Deterministic via a fixed RNG seed.

Usage:
    python src/robustness_ci.py --data data/processed/chicago_taxi_sides.csv
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import sys
sys.path.insert(0, str(Path(__file__).parent))
from splits import temporal_split_indices  # noqa: E402
from baseline_models import BaselineModelTrainer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SEED = 42


def _rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def _r2(y, p):
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def bootstrap_ci(y_true, y_pred, stat_fn, B=2000, alpha=0.05, seed=SEED):
    """Percentile bootstrap CI for a statistic of (y_true, y_pred)."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    stats = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        stats[b] = stat_fn(y_true[idx], y_pred[idx])
    lo, hi = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": stat_fn(y_true, y_pred), "lo": float(lo), "hi": float(hi),
            "B": B, "alpha": alpha}


def bootstrap_temporal_ratio(y_true, y_pred, hours, B=2000, alpha=0.05, seed=SEED):
    """Bootstrap CI for the max/min hourly-RMSE ratio (a whole-test-set statistic)."""
    y_true, y_pred, hours = map(np.asarray, (y_true, y_pred, hours))
    n = len(y_true)
    rng = np.random.default_rng(seed)

    def ratio(idx):
        df = pd.DataFrame({"h": hours[idx], "e2": (y_true[idx] - y_pred[idx]) ** 2})
        rmse_by_h = np.sqrt(df.groupby("h")["e2"].mean())
        return float(rmse_by_h.max() / rmse_by_h.min())

    stats = np.array([ratio(rng.integers(0, n, n)) for _ in range(B)])
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": ratio(np.arange(n)), "lo": float(lo), "hi": float(hi)}


def bootstrap_high_demand_degradation(y_true, y_pred, demand, high_pct=95, B=2000,
                                      alpha=0.05, seed=SEED):
    """Bootstrap CI for %RMSE degradation on high-demand rows vs the rest."""
    y_true, y_pred, demand = map(np.asarray, (y_true, y_pred, demand))
    thr = np.percentile(demand, high_pct)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    def degr(idx):
        hi_mask = demand[idx] >= thr
        if hi_mask.sum() < 5 or (~hi_mask).sum() < 5:
            return np.nan
        r_hi = _rmse(y_true[idx][hi_mask], y_pred[idx][hi_mask])
        r_lo = _rmse(y_true[idx][~hi_mask], y_pred[idx][~hi_mask])
        return (r_hi / r_lo - 1) * 100

    stats = np.array([degr(rng.integers(0, n, n)) for _ in range(B)])
    lo, hi = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": float(degr(np.arange(n))), "lo": float(lo), "hi": float(hi),
            "threshold": float(thr)}


def split_conformal_coverage(model, X_cal, y_cal, X_test, y_test, strata: dict, alpha=0.1):
    """
    Split-conformal absolute-residual intervals calibrated to (1-alpha) coverage OVERALL,
    then report empirical coverage within each stratum in `strata` (name -> boolean mask on test).
    """
    resid = np.abs(np.asarray(y_cal) - model.predict(X_cal))
    n = len(resid)
    # finite-sample conformal quantile level
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    q = float(np.quantile(resid, level, method="higher"))
    pred = model.predict(X_test)
    lo, hi = pred - q, pred + q
    covered = (y_test >= lo) & (y_test <= hi)

    out = {"alpha": alpha, "nominal_coverage": 1 - alpha, "interval_halfwidth": q,
           "overall_coverage": float(np.mean(covered)),
           "overall_avg_width": float(2 * q), "strata": {}}
    for name, mask in strata.items():
        mask = np.asarray(mask)
        if mask.sum() == 0:
            continue
        out["strata"][name] = {"coverage": float(np.mean(covered[mask])),
                               "n": int(mask.sum())}
    return out


def run_analysis(data_path: str, target="trip_count", zone_col="pickup_borough",
                 dt_col="pickup_datetime", B=2000, alpha=0.05):
    df = pd.read_csv(data_path)
    # Parse to real datetime so prepare_features DROPS it (matches experiment_runner.load_data).
    # Left as a string it gets label-encoded into an inconsistent train/test feature that
    # silently degrades the model (R² 0.94 -> 0.84). See ENGINEERING_LOG E-013.
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.dropna(subset=[dt_col]).reset_index(drop=True)

    idx = temporal_split_indices(df, dt_col, ratios=(0.7, 0.15, 0.15))
    train_df, cal_df, test_df = df.iloc[idx["train"]], df.iloc[idx["val"]], df.iloc[idx["test"]]
    trainval_df = df.iloc[np.concatenate([idx["train"], idx["val"]])]

    trainer = BaselineModelTrainer(task_type="regression", random_state=SEED)
    X_trainval, y_trainval = trainer.prepare_features(trainval_df, target)
    X_train, y_train = trainer.prepare_features(train_df, target)
    X_cal, y_cal = trainer.prepare_features(cal_df, target)
    X_test, y_test = trainer.prepare_features(test_df, target)

    # Headline model: trained on train+val (85%), matches the paper's §7 baseline.
    # Used for the bootstrap robustness CIs so they are consistent with the headline number.
    # Hyperparameters match the pipeline's GridSearch winner (regularized forest generalizes
    # better on the future test set than an unregularized one), so R²≈0.94 matches §7.
    rf_kwargs = dict(n_estimators=100, min_samples_split=5, max_depth=None,
                     random_state=SEED, n_jobs=-1)
    model = RandomForestRegressor(**rf_kwargs)
    model.fit(X_trainval, y_trainval)
    pred_test = model.predict(X_test)

    # Conformal model: trained on train (70%) only, so val (15%) is an untouched calibration
    # set (split-conformal requires calibration data the model never saw).
    conf_model = RandomForestRegressor(**rf_kwargs)
    conf_model.fit(X_train, y_train)

    zones = test_df[zone_col].to_numpy()
    hours = test_df["hour"].to_numpy()
    demand = y_test  # trip_count is the target/demand

    logger.info(f"Test rows: {len(y_test)} | overall RMSE={_rmse(y_test, pred_test):.2f} "
                f"R2={_r2(y_test, pred_test):.4f}")

    # --- per-zone R2 with bootstrap CIs ---
    per_zone = {}
    for z in np.unique(zones):
        m = zones == z
        if m.sum() < 10:
            continue
        per_zone[z] = bootstrap_ci(y_test[m], pred_test[m], _r2, B=B, alpha=alpha)
    per_zone = dict(sorted(per_zone.items(), key=lambda kv: kv[1]["point"]))

    # --- temporal ratio + high-demand degradation with CIs ---
    temporal = bootstrap_temporal_ratio(y_test, pred_test, hours, B=B, alpha=alpha)
    high_dmd = bootstrap_high_demand_degradation(y_test, pred_test, demand, B=B, alpha=alpha)

    # --- conformal coverage per stratum ---
    peak = np.isin(hours, [7, 8, 9, 17, 18, 19])
    thr_hi = np.percentile(demand, 95)
    worst_zone = next(iter(per_zone))  # lowest-R2 zone
    strata = {"overall": np.ones(len(y_test), bool),
              "peak_hours": peak, "off_peak": ~peak,
              "high_demand(>=p95)": demand >= thr_hi,
              f"zone:{worst_zone}": zones == worst_zone}
    conformal = split_conformal_coverage(conf_model, X_cal, y_cal, X_test, y_test, strata, alpha=0.1)

    return {"overall": {"rmse": _rmse(y_test, pred_test), "r2": _r2(y_test, pred_test),
                        "n_test": int(len(y_test))},
            "per_zone_r2_ci": per_zone, "temporal_ratio_ci": temporal,
            "high_demand_degradation_ci": high_dmd, "conformal_coverage": conformal}


def _print_report(res):
    ci = lambda d: f"{d['point']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"
    print("\n" + "=" * 68)
    print("ROBUSTNESS WITH 95% BOOTSTRAP CIs  (RandomForest, held-out test)")
    print("=" * 68)
    o = res["overall"]
    print(f"Overall: RMSE={o['rmse']:.2f}  R²={o['r2']:.4f}  (n={o['n_test']})")
    print("\nPer-zone R² (sorted worst→best):")
    for z, d in res["per_zone_r2_ci"].items():
        flag = "  <-- negative, CI excludes 0" if d["hi"] < 0 else ""
        print(f"  {z:16s} R²={ci(d)}{flag}")
    t = res["temporal_ratio_ci"]
    print(f"\nTemporal worst/best-hour RMSE ratio: {t['point']:.1f}x "
          f"[{t['lo']:.1f}, {t['hi']:.1f}]")
    h = res["high_demand_degradation_ci"]
    print(f"High-demand (>=p95) RMSE degradation: {h['point']:.0f}% "
          f"[{h['lo']:.0f}%, {h['hi']:.0f}%]")
    c = res["conformal_coverage"]
    print(f"\nSplit-conformal intervals @ nominal {c['nominal_coverage']:.0%} "
          f"(half-width {c['interval_halfwidth']:.1f} trips):")
    for name, s in c["strata"].items():
        gap = s["coverage"] - c["nominal_coverage"]
        flag = "  <-- UNDER-covers" if gap < -0.02 else ""
        print(f"  {name:22s} coverage={s['coverage']:.1%} (n={s['n']}){flag}")
    print("=" * 68)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/chicago_taxi_sides.csv")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()
    res = run_analysis(args.data, B=args.B)
    _print_report(res)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=2)
        logger.info(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
