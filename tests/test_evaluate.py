"""Focused checks for the evaluation job's algorithm handling and metric shape.

`evaluate.py` is uploaded and executed standalone inside a SageMaker processing
container, so it cannot rely on the `credit_fraud` package being installed.
These tests import the script by path and verify the algorithm alias
normalization and the metric structure that drives the deployment gate.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "credit_fraud"
    / "pipeline"
    / "jobs"
    / "evaluate.py"
)


@pytest.fixture(scope="module")
def evaluate_module():
    spec = importlib.util.spec_from_file_location("evaluate", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluate"] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("xgboost", "xgboost"),
        ("xgb", "xgboost"),
        ("XGBoost", "xgboost"),
        ("lgbm", "lgbm"),
        ("lightgbm", "lgbm"),
        ("LightGBM", "lgbm"),
    ],
)
def test_normalize_algorithm(evaluate_module, raw, expected):
    assert evaluate_module._normalize_algorithm(raw) == expected


def test_normalize_algorithm_rejects_unknown(evaluate_module):
    with pytest.raises(ValueError):
        evaluate_module._normalize_algorithm("catboost")


def test_compute_metrics_shape_matches_gate_path(evaluate_module):
    y_true = np.array([0, 1, 0, 1])
    predictions = np.array([0.1, 0.9, 0.2, 0.8])
    metrics = evaluate_module.compute_metrics(y_true, predictions)

    classification = metrics["classification_metrics"]
    assert float(classification["ROC-AUC"]["value"]) == pytest.approx(1.0)
    assert float(classification["Balanced-Accuracy"]["value"]) == pytest.approx(1.0)
    assert classification["True Negative"]["value"] == 2
    assert classification["True Positive"]["value"] == 2

    # The deployment gate reads validation.classification_metrics.ROC-AUC.value;
    # the written report nests the same structure under "validation" and "test".
    report = {"validation": metrics, "test": metrics}
    assert report["validation"]["classification_metrics"]["ROC-AUC"]["value"] == pytest.approx(1.0)
