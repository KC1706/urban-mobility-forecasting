"""
Tests for the regression metrics that feed the paper's numbers.

These live in `robustness_eval.RobustnessEvaluator.calculate_metrics` and
`baseline_models.BaselineModelTrainer.evaluate_model`. We pin their exact behaviour against
hand-computed values, and document the (real) discrepancy between the two MAPE definitions.
"""
import numpy as np
import pytest

from robustness_eval import RobustnessEvaluator
from baseline_models import BaselineModelTrainer


@pytest.fixture
def y():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 5.0])  # last off by 1
    return y_true, y_pred


def test_robustness_metrics_known_values(y):
    y_true, y_pred = y
    m = RobustnessEvaluator(task_type="regression").calculate_metrics(y_true, y_pred)
    # errors: [0,0,0,1] -> MAE=0.25, MSE=0.25 -> RMSE=0.5
    assert m["mae"] == pytest.approx(0.25)
    assert m["rmse"] == pytest.approx(0.5)
    # R^2 = 1 - SSE/SST ; SSE=1, SST=var*n = 5 -> 1 - 1/5 = 0.8
    assert m["r2"] == pytest.approx(0.8)


def test_robustness_metrics_perfect_prediction():
    y_true = np.array([3.0, 7.0, 11.0])
    m = RobustnessEvaluator(task_type="regression").calculate_metrics(y_true, y_true)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["mae"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)
    assert m["mape"] == pytest.approx(0.0, abs=1e-6)


def test_baseline_evaluate_model_matches_formulas(y):
    y_true, y_pred = y

    class _Dummy:
        """A stand-in model whose predict() returns preset predictions."""
        def __init__(self, preds): self._p = preds
        def predict(self, X): return self._p

    trainer = BaselineModelTrainer(task_type="regression")
    trainer.models["dummy"] = _Dummy(y_pred)
    metrics = trainer.evaluate_model("dummy", X_test=np.zeros((4, 1)), y_test=y_true)
    assert metrics["rmse"] == pytest.approx(0.5)
    assert metrics["mae"] == pytest.approx(0.25)
    assert metrics["r2"] == pytest.approx(0.8)


def test_mape_definitions_differ_by_epsilon(y):
    """baseline_models divides by y_true; robustness_eval divides by (y_true + 1e-8).

    They agree to ~1e-6 when no y_true is zero. This test documents the inconsistency so a
    future cleanup unifies them intentionally rather than by accident.
    """
    y_true, y_pred = y
    rob = RobustnessEvaluator(task_type="regression").calculate_metrics(y_true, y_pred)["mape"]
    baseline_mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)  # baseline formula
    assert rob == pytest.approx(baseline_mape, rel=1e-6)
