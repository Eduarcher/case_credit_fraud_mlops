from abc import ABC, abstractmethod
from typing import Literal

from sagemaker import model_uris
from sagemaker.estimator import Estimator
from sagemaker.xgboost.estimator import XGBoost
from sagemaker.workflow.steps import TrainingStep
from sagemaker.inputs import TrainingInput
from sagemaker.image_uris import retrieve

from .step import Step
from credit_fraud.pipeline.context import CreditFraudPipelineContext
from credit_fraud.pipeline.exceptions import InvalidAlgorithmFramework


class TrainingAlgorithmStrategy(ABC):
    """
    Abstract base class for training algorithm strategies.

    Attributes:
        context (CreditFraudPipelineContext): The context object for the 
            credit fraud pipeline.
        model_environment (dict): The model environment containing MLflow ARN 
            and run ID.
    """

    def __init__(self, context: CreditFraudPipelineContext) -> None:
        """
        Initializes a new instance of the TrainingAlgorithmStrategy class.

        Args:
            context (CreditFraudPipelineContext): The context object for the 
                credit fraud pipeline.
        """
        self.context = context
        self.model_environment = {
            "MLFLOW_ARN": self.context.mlflow.server_arn,
            "MLFLOW_RUN_ID": self.context.mlflow.experiment_run_id,
        }

    @abstractmethod
    def get_image_uri(self, scope: Literal["training", "inference"]) -> str:
        """
        Abstract method to get the image URI for the training or inference scope.

        Args:
            scope (str): The scope for which to get the image URI.
                Can be "training" or "inference".

        Returns:
            str: The image URI.
        """
        pass

    @abstractmethod
    def build(self, processed_data_uri: str) -> TrainingStep:
        """
        Abstract method to build the training step.

        Args:
            processed_data_uri (str): The URI of the processed data.

        Returns:
            TrainingStep: The training step.
        """
        pass


class XGBoostAlgorithmStrategy(TrainingAlgorithmStrategy):
    """
    A class representing the XGBoost algorithm strategy for training models
        in the Credit Fraud pipeline.

    Args:
        context (CreditFraudPipelineContext): The context object for the pipeline.

    Attributes:
        xgb_estimator (XGBoost): The XGBoost estimator for training the model.
    """

    def __init__(self, context: CreditFraudPipelineContext) -> None:
        super().__init__(context)
        self.context.logger.info("Configuring XGBoost Model")
        self.xgb_estimator = XGBoost(
            entry_point="train.py",
            output_path=f"{self.context.bucket_folder}/runs/{self.context.execution_name}/model",
            code_location=f"{self.context.bucket_folder}/runs/{self.context.execution_name}/scripts/",
            source_dir="credit_fraud/pipeline/jobs/xgboost",
            hyperparameters=self.context.model_params,
            environment=self.model_environment,
            role=self.context.sagemaker_role,
            instance_count=self.context.cfg["Training"]["TrainInstanceCount"],
            instance_type=self.context.cfg["Training"]["TrainInstanceType"],
            framework_version=self.context.cfg["Training"]["XGBoostFrameworkVersion"],
            disable_profiler=True,
        )

    def get_image_uri(self, scope: Literal["training", "inference"] = None) -> str:
        """
        Retrieves the image URI for the XGBoost framework.

        Args:
            scope (str, optional): The scope of the image URI. Defaults to None.

        Returns:
            str: The image URI for the XGBoost framework.
        """
        return retrieve(
            framework="xgboost",
            region=self.context.region,
            version=self.context.cfg["Training"]["XGBoostFrameworkVersion"],
        )

    def build(self, train_data_uri: str, validation_data_uri: str) -> TrainingStep:
        """
        Builds the training step for the XGBoost model.

        Args:
            train_data_uri (str): The S3 URI of the training data.
            validation_data_uri (str): The S3 URI of the validation data.

        Returns:
            TrainingStep: The training step for the XGBoost model.
        """
        training_dataset_s3_path = TrainingInput(
            s3_data=train_data_uri, content_type="parquet", s3_data_type="S3Prefix"
        )

        validation_dataset_s3_path = TrainingInput(
            s3_data=validation_data_uri, content_type="parquet", s3_data_type="S3Prefix"
        )

        train_step = TrainingStep(
            name="XGBModelTraining",
            estimator=self.xgb_estimator,
            inputs={
                "train_data_path": training_dataset_s3_path,
                "validation_data_path": validation_dataset_s3_path,
            },
        )
        return train_step


class LGBMAlgorithmStrategy(TrainingAlgorithmStrategy):
    """
    A class representing the LightGBM algorithm strategy for training in the 
        Credit Fraud pipeline.

    Args:
        context (CreditFraudPipelineContext): The context object for the pipeline.

    Attributes:
        train_model_id (str): The ID of the LightGBM classification model.
        train_model_version (str): The version of the LightGBM classification model.
        lgbm_estimator (Estimator): The estimator object for training 
            the LightGBM model.

    Methods:
        get_image_uri: Retrieves the image URI for the training scope.
        build: Builds the training step for the LightGBM algorithm.

    """

    def __init__(self, context: CreditFraudPipelineContext) -> None:
        super().__init__(context)
        self.context.logger.info("Configuring LGBM Model")

        self.train_model_id = "lightgbm-classification-model"
        self.train_model_version = "2.1.0"
        train_image_uri = self.get_image_uri()

        train_model_uri = model_uris.retrieve(
            model_id=self.train_model_id,
            model_version=self.train_model_version,
            model_scope="training",
        )

        # Build estimator
        self.lgbm_estimator = Estimator(
            role=self.context.sagemaker_role,
            image_uri=train_image_uri,
            output_path=f"{self.context.bucket_folder}/runs/{self.context.execution_name}/model",
            code_location=f"{self.context.bucket_folder}/runs/{self.context.execution_name}/scripts/",
            source_dir="credit_fraud/pipeline/jobs/lgbm",
            model_uri=train_model_uri,
            entry_point="train.py",
            instance_count=self.context.cfg["Training"]["TrainInstanceCount"],
            instance_type=self.context.cfg["Training"]["TrainInstanceType"],
            max_run=360000,
            hyperparameters=self.context.model_params,
            environment=self.model_environment,
        )

    def get_image_uri(
        self, scope: Literal["training", "inference"] = "training"
    ) -> str:
        """
        Retrieves the image URI for the training scope.

        Args:
            scope (str): The scope of the image URI. Defaults to "training".

        Returns:
            str: The image URI.
        """
        return retrieve(
            region=self.context.region,
            framework=None,
            model_id=self.train_model_id,
            model_version=self.train_model_version,
            image_scope=scope,
            instance_type=self.context.cfg["Training"]["TrainInstanceType"],
        )

    def build(self, train_data_uri: str, validation_data_uri: str) -> TrainingStep:
        """
        Builds the training step for the LightGBM algorithm.

        Args:
            train_data_uri (str): The URI of the training data.
            validation_data_uri (str): The URI of the validation data.

        Returns:
            TrainingStep: The training step object.
        """
        training_dataset_s3_path = TrainingInput(
            s3_data=train_data_uri,
            content_type="application/x-parquet",
        )

        validation_dataset_s3_path = TrainingInput(
            s3_data=validation_data_uri,
            content_type="application/x-parquet",
        )

        train_step = TrainingStep(
            name="LGBMTraining",
            estimator=self.lgbm_estimator,
            inputs={
                "train": training_dataset_s3_path,
                "validation": validation_dataset_s3_path,
            },
        )
        return train_step


class TrainStepJob(Step):
    """
    Constructs a step in the credit fraud pipeline for training the model.

    Args:
        context (CreditFraudPipelineContext): The context object for the pipeline.

    Attributes:
        context (CreditFraudPipelineContext): The context object for the pipeline.
        _strategy_algorithm (TrainingAlgorithmStrategy): 
            The strategy algorithm for training.
    """

    def __init__(self, context: CreditFraudPipelineContext) -> None:
        self.context = context
        self._strategy_algorithm = self._instance_strategy_algorithm(
            strategy=self.context.training_algorithm
        )

    @property
    def strategy_algorithm(self) -> TrainingAlgorithmStrategy:
        return self._strategy_algorithm

    @strategy_algorithm.setter
    def strategy_algorithm(self, strategy) -> None:
        return self._instance_strategy_algorithm(strategy)

    def _instance_strategy_algorithm(self, strategy) -> TrainingAlgorithmStrategy:
        """
        Instantiate the strategy algorithm based on the provided training algorithm.

        Args:
            strategy (str): The training algorithm.

        Returns:
            TrainingAlgorithmStrategy: The instantiated strategy algorithm.

        Raises:
            InvalidAlgorithmFramework: If the provided training algorithm is invalid.
        """
        self.context.logger.info("Configuring preprocessing framework...")
        if strategy.lower() in ("xgboost", "xgb"):
            return XGBoostAlgorithmStrategy(self.context)
        elif strategy.lower() in ("lightgbm", "lgbm"):
            return LGBMAlgorithmStrategy(self.context)
        else:
            raise InvalidAlgorithmFramework(
                "Invalid training algorithm." 
                + "Available algorithms are: xgboost, lightgbm."
            )

    def build(self, train_data_uri, validation_data_uri) -> TrainingStep:
        """
        Build the training step used in the pipeline.

        Args:
            train_data_uri (str): The S3 URI of the training data.
            validation_data_uri (str): The S3 URI of the validation data.

        Returns:
            TrainingStep: The built training step.
        """
        self.context.logger.info("Building training step.")
        return self._strategy_algorithm.build(train_data_uri, validation_data_uri)
