from sagemaker.lambda_helper import Lambda
from sagemaker.workflow.lambda_step import (
    LambdaStep,
    LambdaOutput,
    LambdaOutputTypeEnum,
)

from .step import Step
from credit_fraud.pipeline.context import CreditFraudPipelineContext
from credit_fraud.pipeline.exceptions import InvalidAlgorithmFramework


class RegisterModelStepJob(Step):
    """
    Constructs a step in the credit fraud pipeline for registering the model
        in MLFlow using an Lambda Function.

    Args:
        context (CreditFraudPipelineContext): The context object for the pipeline.
    """

    def __init__(self, context: CreditFraudPipelineContext):
        self.context = context

        # Get Lambda Function Arn and build Lambda Sagemaker helper instance
        response = self.context.lambda_functions.client.get_function(
            FunctionName=self.context.lambda_functions.register_model_func_name
        )
        self.lambda_func = Lambda(function_arn=response["Configuration"]["FunctionArn"])

    def build(self, model_artifact_s3_uri: str):
        """
        Builds the RegisterModelStep.

        Args:
            model_artifact_s3_uri (str): The S3 URI of the model artifact.

        Returns:
            LambdaStep: The RegisterModelStep.

        Raises:
            InvalidAlgorithmFramework: If the provided training algorithm is invalid.
        """
        if self.context.training_algorithm.lower() in ("xgboost", "xgb"):
            model_name = "XGBoost"
        elif self.context.training_algorithm.lower() in ("lightgbm", "lgbm"):
            model_name = "LightGBM"
        else:
            raise InvalidAlgorithmFramework(
                "Invalid training algorithm."
                + "Available algorithms are: xgboost, lightgbm."
            )
        status_code = LambdaOutput(
            output_name="status_code", output_type=LambdaOutputTypeEnum.String
        )
        response_body = LambdaOutput(
            output_name="body", output_type=LambdaOutputTypeEnum.String
        )
        register_model_step = LambdaStep(
            name="RegisterModel",
            lambda_func=self.lambda_func,
            inputs={
                "mlflow_arn": self.context.mlflow.server_arn,
                "run_id": self.context.mlflow.experiment_run_id,
                "model_path": model_artifact_s3_uri,
                "model_name": model_name,
            },
            outputs=[status_code, response_body],
        )
        return register_model_step
