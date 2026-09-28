from sagemaker.model import Model
from sagemaker.workflow.steps import CreateModelStep
from sagemaker.inputs import CreateModelInput

from .step import Step
from credit_fraud.pipeline.context import CreditFraudPipelineContext


class CreateModelStepJob(Step):
    """
     Constructs a step in the credit fraud pipeline for creating a Sagemaker model.
     This model will be used to deploy an endpoint.

    Args:
        context (CreditFraudPipelineContext): The context object for the pipeline.
        image_uri (str): The URI of the Docker image for the model.
    """

    def __init__(self, context: CreditFraudPipelineContext, image_uri):
        self.context = context
        self.image_uri = image_uri

    def build(self, model_artifact_s3_uri: str) -> CreateModelStep:
        """
        Builds the CreateModelStep object used by the pipeline.

        Args:
            model_artifact_s3_uri (str): The S3 URI of the model artifact.

        Returns:
            CreateModelStep: The CreateModelStep object.
        """
        model = Model(
            image_uri=self.image_uri,
            model_data=model_artifact_s3_uri,
            sagemaker_session=self.context,
            role=self.context.sagemaker_role,
        )
        inputs = CreateModelInput(
            instance_type=self.context.cfg["Deployment"]["DeployInstanceType"]
        )
        create_model_step = CreateModelStep(
            name="CreateModel", model=model, inputs=inputs
        )
        return create_model_step
