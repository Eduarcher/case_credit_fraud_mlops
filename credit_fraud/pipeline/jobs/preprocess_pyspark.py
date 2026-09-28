"""Preprocessing job for PySpark framework."""

import logging
import argparse
import os

from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as f
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructField,
    StructType,
    DoubleType,
    IntegerType,
    FloatType,
)
from pyspark.ml.feature import MinMaxScaler, RobustScaler
from pyspark.ml.functions import vector_to_array
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def move_column_to_first(df: DataFrame, col: str):
    df_columns = df.columns
    df_columns.remove(col)
    return df.select(col, *df_columns)


def transform_dataframe(df):
    # Apply Scaling to training dataset
    df_scaled = scalerModel.transform(df)
    for i in range(1, 29):
        df_scaled = df_scaled.withColumn(
            f"V{i}", vector_to_array("min_max_features_scaled").getItem(i)
        )
    df_scaled = df_scaled.withColumn(
        "Amount", vector_to_array("Amount_scaled").getItem(0)
    ).drop("min_max_features", "min_max_features_scaled", "Amount_vec", "Amount_scaled")
    # Reorder columns. `Class` need to come first.
    df_reordered = move_column_to_first(df_scaled, "Class")
    return df_reordered


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-method", type=str, default="s3")
    parser.add_argument("--raw-data-key", type=str)
    parser.add_argument("--train-data-folder", type=str)
    parser.add_argument("--validation-data-folder", type=str)
    parser.add_argument("--test-data-folder", type=str)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    args, _ = parser.parse_known_args()

    assert args.train_ratio + args.validation_ratio + args.test_ratio == 1.0

    local_dir = "/opt/ml/processing"

    spark = (
        SparkSession.builder.appName("PreprocessingJob")
        .config("spark.jars", "mysql-connector-j-9.0.0.jar")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    # Define raw data schema
    schema = StructType(
        [
            StructField("Time", IntegerType(), True),
            StructField("V1", DoubleType(), True),
            StructField("V2", DoubleType(), True),
            StructField("V3", DoubleType(), True),
            StructField("V4", DoubleType(), True),
            StructField("V5", DoubleType(), True),
            StructField("V6", DoubleType(), True),
            StructField("V7", DoubleType(), True),
            StructField("V8", DoubleType(), True),
            StructField("V9", DoubleType(), True),
            StructField("V10", DoubleType(), True),
            StructField("V11", DoubleType(), True),
            StructField("V12", DoubleType(), True),
            StructField("V13", DoubleType(), True),
            StructField("V14", DoubleType(), True),
            StructField("V15", DoubleType(), True),
            StructField("V16", DoubleType(), True),
            StructField("V17", DoubleType(), True),
            StructField("V18", DoubleType(), True),
            StructField("V19", DoubleType(), True),
            StructField("V20", DoubleType(), True),
            StructField("V21", DoubleType(), True),
            StructField("V22", DoubleType(), True),
            StructField("V23", DoubleType(), True),
            StructField("V24", DoubleType(), True),
            StructField("V25", DoubleType(), True),
            StructField("V26", DoubleType(), True),
            StructField("V27", DoubleType(), True),
            StructField("V28", DoubleType(), True),
            StructField("Amount", FloatType(), True),
            StructField("Class", IntegerType(), True),
        ]
    )

    # Load dataset from s3
    if args.source_method.lower() == "s3":
        df = spark.read.csv(args.raw_data_key, header=True, schema=schema)
    elif args.source_method.lower() == "rds":
        df = spark.read.jdbc(
            url=f"jdbc:mysql://{os.environ.get('RDS_HOST_URL')}",
            table="credit_fraud.transactions",
            properties={
                "user": os.environ.get("RDS_SECRET_USERNAME"),
                "password": os.environ.get("RDS_SECRET_PASSWORD"),
                "driver": "com.mysql.cj.jdbc.Driver",
            },
        )

    # insert id column
    df = df.withColumn("id", f.row_number().over(Window.orderBy("Time")))

    # Split Train, Validation and Test Sets
    df_row_count = df.count()
    train_row_count = int(df_row_count * args.train_ratio)
    validation_row_count = int(df_row_count * args.validation_ratio)
    df_train = df.filter(f"id <= {train_row_count}").drop("id", "Time")
    df_validation = df.filter(
        f"id > {train_row_count} AND id <= {train_row_count + validation_row_count}"
    ).drop("id", "Time")
    df_test = df.filter(f"id > {train_row_count + validation_row_count}").drop(
        "id", "Time"
    )

    # Define Assemblers and Scalers
    columns_to_scale = [f"V{col_id}" for col_id in range(1, 29)]
    assemblers = [
        VectorAssembler(inputCols=columns_to_scale, outputCol="min_max_features"),
        VectorAssembler(inputCols=["Amount"], outputCol="Amount_vec"),
    ]
    scalers = [
        MinMaxScaler(inputCol="min_max_features", outputCol="min_max_features_scaled"),
        RobustScaler(inputCol="Amount_vec", outputCol="Amount_scaled"),
    ]
    pipeline = Pipeline(stages=assemblers + scalers)
    scalerModel = pipeline.fit(df_train)

    df_train = transform_dataframe(df_train)
    df_validation = transform_dataframe(df_validation)
    df_test = transform_dataframe(df_test)

    df_train.write.mode("overwrite").parquet(args.train_data_folder)
    df_validation.write.mode("overwrite").parquet(args.validation_data_folder)
    df_test.write.mode("overwrite").parquet(args.test_data_folder)
