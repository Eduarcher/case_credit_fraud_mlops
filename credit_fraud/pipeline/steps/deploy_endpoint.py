from sagemaker.lambda_helper import Lambda
from sagemaker.workflow.lambda_step import LambdaStep

from .step import Step
from credit_fraud.pipeline.context import CreditFraudPipelineContext


class DeployEndpointStepJob(Step):
    """
    Constructs a step in the credit fraud pipeline for deploying the model endpoint.

    Args:
        context (CreditFraudPipelineContext): The context object for the pipeline.
    """

    def __init__(self, context: CreditFraudPipelineContext):
        self.context = context

        # Get Lambda Function Arn and build Lambda Sagemaker helper instance
        response = self.context.lambda_functions.client.get_function(
            FunctionName=self.context.lambda_functions.deploy_func_name,
        )
        self.lambda_func = Lambda(function_arn=response["Configuration"]["FunctionArn"])

    def build(self, model_name: str) -> LambdaStep:
        """
        Builds a LambdaStep object for deploying the endpoint.
        This Lambda function also enables auto-scalling.

        Args:
            model_name (str): The name of the model to be deployed.

        Returns:
            LambdaStep: The LambdaStep object for deploying the endpoint.
        """
        deploy_step = LambdaStep(
            name="LambdaStepRealTimeDeploy",
            lambda_func=self.lambda_func,
            inputs={
                "model_name": model_name,
                "endpoint_name": self.context.cfg["Deployment"]["EndpointName"],
                "instance_type": self.context.cfg["Deployment"]["DeployInstanceType"],
            },
        )
        return deploy_step
