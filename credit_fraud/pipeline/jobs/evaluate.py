"""Evaluation job responsible for automatically evaluating the model metrics."""

import json
import logging
import pathlib
import pickle
import tarfile
import sys
import subprocess
import argparse

import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def _normalize_algorithm(model_algorithm):
    algorithm = model_algorithm.lower()
    if algorithm in ("xgboost", "xgb"):
        return "xgboost"
    if algorithm in ("lgbm", "lightgbm"):
        return "lgbm"
    raise ValueError(f"Unsupported model algorithm: {model_algorithm}")


def install_dependencies(model_algorithm):
    logger.info("Attempting to install dependencies")
    model_algorithm = _normalize_algorithm(model_algorithm)
    if model_algorithm == "xgboost":
        global xgb
        import xgboost as xgb
    elif model_algorithm == "lgbm":
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "lightgbm==3.3.3"]
        )
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "mlflow>=2.13", "sagemaker-mlflow"]
    )
    global mlflow
    import mlflow


def compute_metrics(y_true, predictions):
    pred_class = np.where(predictions > 0.5, 1, 0)
    roc_auc = roc_auc_score(y_true, predictions)
    bacc = balanced_accuracy_score(y_true, pred_class)
    cm = confusion_matrix(y_true, pred_class)
    return {
        "classification_metrics": {
            "ROC-AUC": {"value": roc_auc},
            "Balanced-Accuracy": {"value": bacc},
            "True Negative": {"value": int(cm[0][0])},
            "False Positive": {"value": int(cm[0][1])},
            "False Negative": {"value": int(cm[1][0])},
            "True Positive": {"value": int(cm[1][1])},
        }
    }


def evaluate_split(model, df, model_algorithm):
    model_algorithm = _normalize_algorithm(model_algorithm)
    y = df["Class"]
    X = df.drop("Class", axis=1)
    if model_algorithm == "xgboost":
        X = xgb.DMatrix(X)
    predictions = model.predict(X)
    return compute_metrics(y, predictions)


def log_metrics(prefix, metrics):
    for key, value in metrics["classification_metrics"].items():
        metric_name = key.replace(" ", "-")
        mlflow.log_metric(f"{prefix}/{metric_name}", value["value"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-algorithm", type=str, default="xgboost")
    parser.add_argument("--mlflow-arn", type=str)
    parser.add_argument("--mlflow-run-id", type=str)
    args = parser.parse_args()

    install_dependencies(args.model_algorithm)

    # Start MLFlow run to continue metrics logging
    mlflow.set_tracking_uri(args.mlflow_arn)
    mlflow.start_run(run_id=args.mlflow_run_id)

    logger.info("Loading model pickle file.")
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
    model = pickle.load(open("model.pkl", "rb"))

    logger.info("Reading validation and test data.")
    df_validation = pd.read_parquet("/opt/ml/processing/validation.parquet")
    df_test = pd.read_parquet("/opt/ml/processing/test.parquet")

    logger.info("Evaluating the model on the validation set.")
    validation_metrics = evaluate_split(model, df_validation, args.model_algorithm)

    logger.info("Evaluating the model on the test set.")
    test_metrics = evaluate_split(model, df_test, args.model_algorithm)

    # The validation metrics drive the deployment gate; the test metrics are
    # kept as a final, report-only assessment.
    metric_dict = {
        "validation": validation_metrics,
        "test": test_metrics,
    }

    # Save model evaluation metrics
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(
        "Writing evaluation report with validation ROC-AUC: %f",
        validation_metrics["classification_metrics"]["ROC-AUC"]["value"],
    )
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(metric_dict))

    log_metrics("Validation", validation_metrics)
    log_metrics("Test", test_metrics)
    mlflow.end_run()
