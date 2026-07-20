"""
Tests for the Phase 2 statistical-rigor helpers (src/robustness_ci.py):
bootstrap CIs and split-conformal coverage.
"""
import numpy as np
import pytest

from robustness_ci import bootstrap_ci, split_conformal_coverage, _r2, _rmse


def test_bootstrap_ci_perfect_prediction_is_degenerate():
    y = np.array([1.0, 5.0, 2.0, 8.0, 3.0, 7.0])
    out = bootstrap_ci(y, y.copy(), _r2, B=200, seed=0)
    assert out["point"] == pytest.approx(1.0)
    # every resample is still perfect -> CI collapses to 1.0
    assert out["lo"] == pytest.approx(1.0) and out["hi"] == pytest.approx(1.0)


def test_bootstrap_ci_brackets_point_and_is_ordered():
    rng = np.random.default_rng(1)
    y = rng.normal(50, 20, 400)
    pred = y + rng.normal(0, 8, 400)
    out = bootstrap_ci(y, pred, _rmse, B=500, alpha=0.05, seed=7)
    assert out["lo"] <= out["point"] <= out["hi"]
    assert out["hi"] - out["lo"] > 0            # non-degenerate interval
    assert out["point"] == pytest.approx(_rmse(y, pred))


def test_bootstrap_ci_is_deterministic_with_seed():
    rng = np.random.default_rng(2)
    y = rng.normal(0, 1, 100); p = y + rng.normal(0, 1, 100)
    a = bootstrap_ci(y, p, _rmse, B=300, seed=123)
    b = bootstrap_ci(y, p, _rmse, B=300, seed=123)
    assert (a["lo"], a["hi"]) == (b["lo"], b["hi"])


class _ConstModel:
    """Predicts a constant; residuals are then |y - c|, easy to reason about."""
    def __init__(self, c): self.c = c
    def predict(self, X): return np.full(len(X), self.c)


def test_split_conformal_coverage_math():
    # Calibration residuals |y_cal - 0| = [1,2,3,...,10]; nominal 90% -> quantile picks ~10.
    y_cal = np.arange(1, 11, dtype=float)
    X_cal = np.zeros((10, 1))
    # Test points: some inside [-q, q], some outside.
    y_test = np.array([0.0, 5.0, 9.0, 50.0])     # last one is far outside
    X_test = np.zeros((4, 1))
    strata = {"overall": np.ones(4, bool), "far": np.array([False, False, False, True])}
    out = split_conformal_coverage(_ConstModel(0.0), X_cal, y_cal, X_test, y_test,
                                   strata, alpha=0.1)
    q = out["interval_halfwidth"]
    assert q == pytest.approx(10.0)                       # conformal quantile at ~max residual
    # points 0,5,9 within +/-10 -> covered; 50 not -> 3/4 overall
    assert out["overall_coverage"] == pytest.approx(0.75)
    assert out["strata"]["far"]["coverage"] == pytest.approx(0.0)


def test_split_conformal_marginal_coverage_is_near_nominal():
    """On exchangeable data, split-conformal should give ~>= (1-alpha) marginal coverage."""
    rng = np.random.default_rng(3)
    y_cal = rng.normal(0, 1, 500)
    y_test = rng.normal(0, 1, 2000)
    out = split_conformal_coverage(_ConstModel(0.0), np.zeros((500, 1)), y_cal,
                                   np.zeros((2000, 1)), y_test,
                                   {"overall": np.ones(2000, bool)}, alpha=0.1)
    assert out["overall_coverage"] >= 0.86   # ~0.90 nominal, allow sampling slack
