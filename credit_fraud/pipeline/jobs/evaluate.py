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


def install_dependencies(model_algorithm):
    logger.info("Attempting to install dependencies")
    if model_algorithm == "xgboost":
        global xgb
        import xgboost as xgb
    elif model_algorithm == "lgbm":
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "lightgbm==4.1.0"]
        )
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "mlflow>=2.13", "sagemaker-mlflow"]
    )
    global mlflow
    import mlflow


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

    logger.info("Reading test data.")
    df_test = pd.read_parquet("/opt/ml/processing/test.parquet")
    y_test = df_test["Class"]
    X_test = df_test.drop("Class", axis=1)
    if args.model_algorithm == "xgboost":
        X_test = xgb.DMatrix(X_test)

    logger.info("Generating predictions for test data.")
    pred = model.predict(X_test)
    pred_class = np.where(pred > 0.5, 1, 0)

    # Calculate model evaluation score
    logger.debug("Calculating ROC-AUC score.")
    roc_auc_test = roc_auc_score(y_test, pred)
    bacc_test = balanced_accuracy_score(y_test, pred_class)
    cm = confusion_matrix(y_test, pred_class)
    metric_dict = {
        "classification_metrics": {
            "ROC-AUC": {"value": roc_auc_test},
            "Balanced-Accuracy": {"value": bacc_test},
            "True Negative": {"value": str(cm[0][0])},
            "False Positive": {"value": str(cm[0][1])},
            "False Negative": {"value": str(cm[1][0])},
            "True Positive": {"value": str(cm[1][1])},
        }
    }

    # Save model evaluation metrics
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Writing evaluation report with ROC-AUC: %f", roc_auc_test)
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(metric_dict))

    mlflow.log_metric("Test/Balanced-Accuracy", bacc_test)
    mlflow.log_metric("Test/ROC-AUC", roc_auc_test)
    t_n, f_p, f_n, t_p = cm.ravel()
    mlflow.log_metric("Test/True-Negative", t_n)
    mlflow.log_metric("Test/False-Positive", f_p)
    mlflow.log_metric("Test/False-Negative", f_n)
    mlflow.log_metric("Test/True-Positive", t_p)
    mlflow.end_run()
