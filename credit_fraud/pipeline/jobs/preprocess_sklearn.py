"""Preprocessing job for Scikit-Learn framework."""

import logging
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-key", type=str)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    args, _ = parser.parse_known_args()

    assert args.train_ratio + args.validation_ratio + args.test_ratio == 1.0

    local_dir = "/opt/ml/processing"
    logger.info(f"Reading raw data from: {args.raw_data_key}")
    df = pd.read_csv(args.raw_data_key, engine="python")

    # Split variables and train/validation/test sets
    logger.info("Split into train, validation test sets.")
    df_time = df["Time"]
    df_y = df["Class"]
    df_x = df.drop(["Class", "Time"], axis=1)
    time_train, time_test, X_train, X_test, y_train, y_test = train_test_split(
        df_time, df_x, df_y, test_size=args.test_ratio, shuffle=False
    )
    time_train, time_validation, X_train, X_validation, y_train, y_validation = (
        train_test_split(
            time_train, X_train, y_train, test_size=args.validation_ratio, shuffle=False
        )
    )

    # Define standardization
    # Fit with train and transform train and test
    logger.info("Applying standardization.")
    ct = ColumnTransformer(
        [
            ("norm_others", MinMaxScaler(), slice(0, 28)),
            ("norm_amount", RobustScaler(), [28]),
        ]
    )
    ct = ct.fit(X_train)
    X_train_scaled = pd.DataFrame(ct.transform(X_train), columns=X_train.columns)
    X_validation_scaled = pd.DataFrame(
        ct.transform(X_validation), columns=X_validation.columns
    )
    X_test_scaled = pd.DataFrame(ct.transform(X_test), columns=X_test.columns)

    # Save to csv
    logger.info(f"Saving output to S3. Location: {local_dir}")
    df_train = pd.concat([y_train, X_train_scaled], axis=1)
    df_validation = pd.concat(
        [y_validation.reset_index(drop=True), X_validation_scaled], axis=1
    )
    df_test = pd.concat([y_test.reset_index(drop=True), X_test_scaled], axis=1)
    df_train.to_parquet(f"{local_dir}/train.parquet/data.parquet", index=False)
    df_validation.to_parquet(
        f"{local_dir}/validation.parquet/data.parquet", index=False
    )
    df_test.to_parquet(f"{local_dir}/test.parquet/data.parquet", index=False)
