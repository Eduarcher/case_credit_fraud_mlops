"""File used for setting up the pipeline context with configurations and parameters"""

import os
import logging
from datetime import datetime
import pytz
import boto3
import json

import yaml
from dotenv import load_dotenv
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.parameters import (
    ParameterString,
    ParameterFloat,
)
import mlflow

from credit_fraud.utils import S3ScriptManager, PyProjectHelper
from credit_fraud.pipeline.exceptions import InvalidAlgorithmFramework


class MLFlowContext:
    """
    A class representing the MLFlow server context for tracking experiments and runs.

    Parameters:
    - mlflow_server_arn (str): The ARN (Amazon Resource Name) of the MLFlow server.
    - run_name (str, optional): The name of the MLFlow run. Defaults to None.
    - local_run (bool, optional): Flag indicating whether the run is local or remote.
        Defaults to False.
    """

    def __init__(
        self, mlflow_server_arn: str, run_name: str = None, local_run: bool = False
    ):
        self.server_arn = mlflow_server_arn
        if not local_run:
            mlflow.set_tracking_uri(self.server_arn)
        run_context = mlflow.start_run(run_name=run_name)
        self.experiment_run_id = run_context.info.run_id
        mlflow.end_run()


class LambdaFunctionsContext:
    """
    A class representing the context for Lambda functions.

    Attributes:
        client: The Boto3 client for Lambda.
        register_model_func_name: The name of the Lambda function used 
            for registering models.
        deploy_func_name: The name of the Lambda function used for deploying models.

    Args:
        cfg: A dictionary containing the configuration settings.
        region (str): The AWS region to use for the Lambda client.
    """

    def __init__(self, cfg, region: str = None):
        self.client = boto3.client("lambda", region_name=region)
        self.register_model_func_name = cfg["Registry"][
            "RegisterModelLambdaFunctionName"
        ]
        self.deploy_func_name = cfg["Deployment"]["DeployLambdaFunctionName"]


class CreditFraudPipelineContext(PipelineSession):
    """
    Represents the context for the Credit Fraud pipeline.

    Args:
        args: Command-line arguments passed to the pipeline.
        logger: Logger object for logging pipeline events.

    Attributes:
        logger: Logger object for logging pipeline events.
        cfg: Configuration loaded from the config.yml file.
        region: AWS region to be used for the pipeline.
        sagemaker_role: IAM role for Amazon SageMaker.
        bucket_name: Name of the S3 bucket for storing pipeline artifacts.
        bucket_folder_prefix: Prefix for the S3 bucket folder.
        bucket_folder: Full path of the S3 bucket folder.
        rds_secret_name: Name of the RDS secret.
        s3_raw_data_key: Key for the raw credit card data in S3.
        processed_train_data_folder: Folder for storing processed training data in S3.
        processed_validation_data_folder: Folder for storing processed 
            validation data in S3.
        processed_test_data_folder: Folder for storing processed test data in S3.
        training_algorithm: Training algorithm of the ML model.
        s3_script_manager: S3ScriptManager object for managing scripts in S3.
        mlflow: MLFlowContext object for managing MLflow runs.
        lambda_functions: LambdaFunctionsContext object for managing 
            AWS Lambda functions.

    Methods:
        __set_execution_name: Sets the execution name for the pipeline.
        __init_pipeline_params: Initializes the pipeline parameters.
    """

    def __init__(self, args, logger=logging):
        self.logger = logger
        self.__set_execution_name()

        self.logger.info("Loading configurations from config.yml file.")
        self.cfg = yaml.safe_load(open("config.yml"))
        self.logger.debug(f"Configurations loaded: {self.cfg}")

        self.logger.info("Loading environment variables from .env file.")
        is_env_loaded = load_dotenv(f"{os.getcwd()}/.env", override=True)
        self.logger.info(
            f"Environment variables from .env files are loaded: {is_env_loaded}"
        )
        self.logger.debug(f"Current Environment: {os.environ}")

        super().__init__(
            default_bucket=os.environ.get("AWS_SAGEMAKER_S3_BUCKET_NAME", None),
            default_bucket_prefix=os.environ.get(
                "AWS_SAGEMAKER_S3_BUCKET_FOLDER_PREFIX", "case-credit-fraud"
            ),
        )

        self.account_id = boto3.client("sts").get_caller_identity().get("Account")
        self.region = os.environ.get("AWS_REGION", self.boto_region_name)
        self.sagemaker_role = os.environ.get(
            "AWS_SAGEMAKER_ROLE_IAM",
            f"arn:aws:iam::{self.account_id}:role/service-role/SageMakerExecutionRole",
        )
        self.bucket_name = os.environ.get(
            "AWS_SAGEMAKER_S3_BUCKET_NAME", self.default_bucket()
        )
        self.bucket_folder_prefix = os.environ.get(
            "AWS_SAGEMAKER_S3_BUCKET_FOLDER_PREFIX", self.default_bucket_prefix
        )
        self.bucket_folder = f"s3://{self.bucket_name}/{self.bucket_folder_prefix}"

        self.rds_host_url = os.environ.get("RDS_HOST_URL")
        self.rds_secret_name = os.environ.get("RDS_SECRET_NAME")
        self.s3_raw_data_key = os.environ.get(
            "S3_RAW_DATA_KEY", f"{self.bucket_folder}/raw/creditcard.csv"
        )
        self.processed_train_data_folder = (
            f"{self.bucket_folder}/runs/{self.execution_name}/processed/train.parquet"
        )
        self.processed_validation_data_folder = (
            f"{self.bucket_folder}/runs/{self.execution_name}/processed/validation.parquet"
        )
        self.processed_test_data_folder = (
            f"{self.bucket_folder}/runs/{self.execution_name}/processed/test.parquet"
        )

        self.training_algorithm = os.environ.get(
            "TRAINING_ALGORITHM", str(self.cfg["Training"]["DefaultTrainingAlgorithm"])
        )
        self.__init_model_params()

        self.s3_script_manager = S3ScriptManager(
            region=self.region,
            destination_bucket_name=self.bucket_name,
            destination_folder=f"{self.bucket_folder_prefix}/runs/{self.execution_name}/scripts",
            logger=logger,
        )

        self.__init_pipeline_params(args)

        self.mlflow = MLFlowContext(
            mlflow_server_arn=os.environ.get("MLFLOW_ARN"),
            run_name=self.execution_name,
            local_run=args.local_run,
        )

        self.lambda_functions = LambdaFunctionsContext(cfg=self.cfg, region=self.region)

    def __set_execution_name(self):
        """
        Sets the execution name for the pipeline.

        The execution name must satisfy the regex pattern:
        ^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,81}
        """
        tz = pytz.timezone("America/Sao_Paulo")
        self.execution_name = (
            f"v{PyProjectHelper.get_version()}--{datetime.now(tz).strftime('%Y%m%d-%H%M%S')}"
            .replace(".", "-")
        )

    def __init_pipeline_params(self, args):
        """
        Initializes the pipeline parameters.

        Args:
            args: Command-line arguments passed to the pipeline.
        """
        self.pipeline_params = {
            "preprocess_train_ratio": ParameterString(
                name="PreprocessTrainRatio",
                default_value=str(self.cfg["Preprocess"]["TrainRatio"]),
            ),
            "preprocess_validation_ratio": ParameterString(
                name="PreprocessValidationRatio",
                default_value=str(self.cfg["Preprocess"]["ValidationRatio"]),
            ),
            "preprocess_test_ratio": ParameterString(
                name="PreprocessTestRatio",
                default_value=str(self.cfg["Preprocess"]["TestRatio"]),
            ),
            "roc_auc_min_threshold": ParameterFloat(
                name="ROCAUCMinThreshold",
                default_value=float(self.cfg["Evaluation"]["ROCAUCMinThreshold"]),
            ),
        }

    def __init_model_params(self):
        """
        Initializes the model parameters based on the training algorithm.

        If the training algorithm is 'xgboost' or 'xgb',
            it loads the model parameters from the 'xgboost_default.json' file.
        If the training algorithm is 'lightgbm' or 'lgbm',
            it loads the model parameters from the 'lgbm_default.json' file.
        Otherwise, it raises an InvalidAlgorithmFramework exception.

        After loading the default model parameters, it checks for any
            environment variables that match the model parameter prefix
        (either 'XGBOOST_' or 'LIGHTGBM_') and updates the corresponding
            model parameter with the environment variable value.

        Finally, it logs the model parameters.

        Raises:
            InvalidAlgorithmFramework: If the training algorithm is not supported.
        """
        if self.training_algorithm.lower() in ["xgboost", "xgb"]:
            model_param_prefix = "XGBOOST_"
            with open("./models/xgboost_default.json", "r") as file:
                self.model_params = json.load(file)
        elif self.training_algorithm.lower() in ["lightgbm", "lgbm"]:
            model_param_prefix = "LIGHTGBM_"
            with open("./models/lgbm_default.json", "r") as file:
                self.model_params = json.load(file)
        else:
            raise InvalidAlgorithmFramework(self.training_algorithm)
        for key, value in os.environ.items():
            if model_param_prefix in key:
                key_clean = key.split(model_param_prefix)[-1].lower()
                self.model_params[key_clean] = value
        self.logger.info(f"Model parameters: {self.model_params}")
