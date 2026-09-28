from abc import ABC, abstractmethod

from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.spark.processing import PySparkProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep

from .step import Step
from credit_fraud.utils import SecretManager
from credit_fraud.pipeline.context import CreditFraudPipelineContext
from credit_fraud.pipeline.exceptions import InvalidProcessingFramework


class ProcessingFrameworkStrategy(ABC):
    """Abstract base class for processing framework strategies."""

    @abstractmethod
    def build(self, *args, **kwargs) -> ProcessingStep:
        """Builds the processing step.

        Returns:
            ProcessingStep: The built processing step.
        """
        pass


class ScikitLearnFrameworkStrategy(ProcessingFrameworkStrategy):
    """Processing framework strategy for Scikit-Learn.

    Args:
        context (CreditFraudPipelineContext): The pipeline context.
    """

    def __init__(self, context: CreditFraudPipelineContext):
        self.context = context

        self.context.s3_script_manager.upload_script(
            source_directory=context.cfg["Global"]["JobsScriptsFolder"],
            script_name="preprocess_sklearn.py",
        )

        self.context.logger.info("Configuring Scikit-Learn processor")
        self.sklearn_processor = SKLearnProcessor(
            role=self.context.sagemaker_role,
            framework_version="1.2-1",
            instance_count=1,
            instance_type=self.context.pipeline_params[
                "preprocess_sklearn_instance_type"
            ],
            base_job_name=f"{self.context['Global']['BaseJobNamePrefix']}-preprocess-sklearn",
        )

    def build(self) -> ProcessingStep:
        """Builds the Scikit-Learn processing step.

        Returns:
            ProcessingStep: The built processing step.
        """
        preprocess_step = ProcessingStep(
            name="ScikitLearnDataPreprocess",
            processor=self.sklearn_processor,
            inputs=[
                ProcessingInput(
                    source=self.context.s3_raw_data_key,
                    destination="/opt/ml/processing/raw",
                ),
            ],
            outputs=[
                ProcessingOutput(
                    destination=self.context.processed_train_data_folder,
                    output_name="train.parquet",
                    source="/opt/ml/processing/train.parquet",
                ),
                ProcessingOutput(
                    destination=self.context.processed_validation_data_folder,
                    output_name="validation.parquet",
                    source="/opt/ml/processing/validation.parquet",
                ),
                ProcessingOutput(
                    destination=self.context.processed_test_data_folder,
                    output_name="test.parquet",
                    source="/opt/ml/processing/test.parquet",
                ),
            ],
            job_arguments=[
                "--raw-data-key",
                "/opt/ml/processing/raw",
                "--train-ratio",
                self.context.pipeline_params["preprocess_train_ratio"],
                "--validation-ratio",
                self.context.pipeline_params["preprocess_validation_ratio"],
                "--test-ratio",
                self.context.pipeline_params["preprocess_test_ratio"],
            ],
            code=self.context.s3_script_manager.get_script_uri("preprocess_sklearn.py"),
        )
        return preprocess_step


class PysparkFrameworkStrategy(ProcessingFrameworkStrategy):
    """Processing framework strategy for PySpark.

    Args:
        context (CreditFraudPipelineContext): The pipeline context.
    """

    def __init__(self, context: CreditFraudPipelineContext):
        self.context = context

        self.context.s3_script_manager.upload_script(
            source_directory=context.cfg["Global"]["JobsScriptsFolder"],
            script_name="preprocess_pyspark.py",
        )

        self._setup_spark_processor()

    def _setup_spark_processor(self):
        """Sets up the PySpark processor with necessary configurations."""
        env_vars = {"AWS_SPARK_CONFIG_MODE": "2"}
        if self.context.cfg["Preprocess"]["SourceMethod"] == "rds":
            rds_secret = SecretManager(
                region_name=self.context.region, logger=self.context.logger
            ).get_secret(self.context.rds_secret_name)
            env_vars["RDS_SECRET_USERNAME"] = rds_secret["username"]
            env_vars["RDS_SECRET_PASSWORD"] = rds_secret["password"]
            env_vars["RDS_HOST_URL"] = self.context.rds_host_url

        self.context.logger.info("Configuring PySpark processor")
        self.spark_processor = PySparkProcessor(
            dependency_location=f"{self.context.bucket_folder}/jars",
            base_job_name=f"{self.context.cfg['Global']['BaseJobNamePrefix']}-preprocess-pyspark",
            framework_version="3.3",
            py_version="py39",
            container_version="1",
            role=self.context.sagemaker_role,
            instance_count=self.context.cfg["Preprocess"][
                "PreprocessPysparkInstanceCount"
            ],
            instance_type=self.context.cfg["Preprocess"][
                "PreprocessPysparkInstanceType"
            ],
            max_runtime_in_seconds=3600,
            env=env_vars,
            sagemaker_session=self.context,
        )

    def build(self) -> ProcessingStep:
        """Builds the PySpark processing step.

        Returns:
            ProcessingStep: The built processing step.
        """
        run_args = self.spark_processor.run(
            submit_app=self.context.s3_script_manager.get_script_uri(
                "preprocess_pyspark.py"
            ),
            submit_jars=["dependencies/mysql-connector-j-9.0.0.jar"],
            arguments=[
                "--source-method",
                self.context.cfg["Preprocess"]["SourceMethod"],
                "--raw-data-key",
                self.context.s3_raw_data_key,
                "--train-data-folder",
                self.context.processed_train_data_folder,
                "--validation-data-folder",
                self.context.processed_validation_data_folder,
                "--test-data-folder",
                self.context.processed_test_data_folder,
                "--train-ratio",
                self.context.pipeline_params["preprocess_train_ratio"],
                "--validation-ratio",
                self.context.pipeline_params["preprocess_validation_ratio"],
                "--test-ratio",
                self.context.pipeline_params["preprocess_test_ratio"],
            ],
            outputs=[
                ProcessingOutput(
                    destination=self.context.processed_train_data_folder,
                    output_name="train.parquet",
                    source="/opt/ml/processing/train.parquet",
                ),
                ProcessingOutput(
                    destination=self.context.processed_validation_data_folder,
                    output_name="validation.parquet",
                    source="/opt/ml/processing/validation.parquet",
                ),
                ProcessingOutput(
                    destination=self.context.processed_test_data_folder,
                    output_name="test.parquet",
                    source="/opt/ml/processing/test.parquet",
                ),
            ],
        )

        preprocess_step = ProcessingStep(
            name="PySparkDataPreprocess",
            step_args=run_args,
        )
        return preprocess_step


class PreprocessStepJob(Step):
    """Constructs a step in the credit fraud pipeline for preprocessing data.

    Args:
        context (CreditFraudPipelineContext): The pipeline context.
    """

    def __init__(self, context: CreditFraudPipelineContext):
        self.context = context
        self._strategy_framework = self._instantiate_strategy_framework(
            strategy=self.context.cfg["Preprocess"]["PreprocessFramework"]
        )

    @property
    def strategy_framework(self):
        return self._strategy_framework

    @strategy_framework.setter
    def strategy_framework(self, strategy: str):
        return self._instantiate_strategy_framework(strategy)

    def _instantiate_strategy_framework(
        self, strategy: str
    ) -> ProcessingFrameworkStrategy:
        """Instantiates the processing framework strategy.

        Args:
            strategy (str): The processing framework strategy name.
            Accepts "scikit-learn" or "pyspark". Ignores uppercase.

        Returns:
            ProcessingFrameworkStrategy: The instantiated processing framework strategy.

        Raises:
            InvalidProcessingFramework: If the processing framework strategy is invalid.
        """
        self.context.logger.info("Configuring preprocessing framework...")
        if strategy.lower() in ("scikit-learn", "sklearn"):
            return ScikitLearnFrameworkStrategy(self.context)
        elif strategy.lower() in ("pyspark", "spark"):
            return PysparkFrameworkStrategy(self.context)
        else:
            raise InvalidProcessingFramework(
                "Invalid sagemaker processing framework. "
                + "Available frameworks are: scikit-learn, pyspark."
            )

    def build(self) -> ProcessingStep:
        """Builds the preprocessing step.

        Returns:
            ProcessingStep: The built preprocessing step.
        """
        self.context.logger.info("Building preprocessing step.")
        assert self.context.cfg["Preprocess"]["SourceMethod"].lower() in ("s3", "rds")
        return self._strategy_framework.build()


if __name__ == "__main__":
    pass
