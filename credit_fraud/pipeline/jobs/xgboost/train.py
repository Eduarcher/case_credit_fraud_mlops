"""Training job for XGBoost framework."""

import argparse
import os
import logging
import subprocess
import sys

import xgboost as xgb
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def install_extra_dependencies():
    logger.info("Attempting to extra dependencies")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            "./requirements.txt",
        ]
    )
    global mlflow
    import mlflow


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", type=str, default="binary:logistic")
    parser.add_argument("--num_round", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--nfold", type=int, default=3)
    parser.add_argument("--early_stopping_rounds", type=int, default=3)
    parser.add_argument(
        "--train_data_path",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN_DATA_PATH"),
    )
    parser.add_argument(
        "--validation_data_path",
        type=str,
        default=os.environ.get("SM_CHANNEL_VALIDATION_DATA_PATH"),
    )
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--mlflow-arn", type=str, default=os.environ.get("MLFLOW_ARN"))
    parser.add_argument(
        "--mlflow-run-id", type=str, default=os.environ.get("MLFLOW_RUN_ID")
    )
    args = parser.parse_args()

    # Start an MLflow run
    install_extra_dependencies()
    mlflow.set_tracking_uri(args.mlflow_arn)
    run_context = mlflow.start_run(run_id=args.mlflow_run_id)
    mlflow.xgboost.autolog()

    # Load data for training
    logger.info("Loading data.")
    df_train = pd.read_parquet(args.train_data_path)
    X_train = df_train.drop("Class", axis=1)
    y_train = pd.DataFrame(df_train["Class"])
    dm_train = xgb.DMatrix(X_train, label=y_train)

    df_validation = pd.read_parquet(args.validation_data_path)
    X_validation = df_validation.drop("Class", axis=1)
    y_validation = pd.DataFrame(df_validation["Class"])
    dm_validation = xgb.DMatrix(X_validation, label=y_validation)

    logger.info("Starting model training...")
    classifier = xgb.train(
        params={
            "max_depth": args.max_depth,
            "eta": args.eta,
            "objective": args.objective,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
        },
        dtrain=dm_train,
        num_boost_round=args.num_round,
    )
    logger.info("Model training complete!")

    # Predict validation data and generate training metrics
    y_validation_predict_proba = classifier.predict(dm_validation)
    y_validation_predict_class = np.where(y_validation_predict_proba > 0.5, 1, 0)

    bacc_validation = balanced_accuracy_score(y_validation, y_validation_predict_class)
    logger.info(f"Balanced Accuracy Score Validation: {bacc_validation}")

    roc_auc = roc_auc_score(y_validation, y_validation_predict_proba)
    logger.info(f"ROC-AUC Validation: {roc_auc}")

    logger.info("Confusion Matrix: (Absolute and Normalized)")
    cm = confusion_matrix(y_validation, y_validation_predict_class)
    t_n, f_p, f_n, t_p = cm.ravel()
    logger.info(cm)
    logger.info(
        np.round(
            confusion_matrix(
                y_validation, y_validation_predict_class, normalize="true"
            ),
            decimals=4,
        )
    )

    # Save the trained model to the location specified by model_dir
    model_location = args.model_dir + "/model.pkl"
    logger.info("Saving model")
    with open(model_location, "wb") as f:
        pickle.dump(classifier, f)

    # Log the evaluation metrics
    mlflow.log_metric("Validation/Balanced-Accuracy", bacc_validation)
    mlflow.log_metric("Validation/ROC-AUC", roc_auc)
    mlflow.log_metric("Validation/True-Negative", t_n)
    mlflow.log_metric("Validation/False-Positive", f_p)
    mlflow.log_metric("Validation/False-Negative", f_n)
    mlflow.log_metric("Validation/True-Positive", t_p)
    mlflow.log_dict(np.array(cm).tolist(), "confusion_matrix_validation.json")
    mlflow.end_run()
