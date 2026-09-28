from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.properties import PropertyFile

from .step import Step
from credit_fraud.pipeline.context import CreditFraudPipelineContext


class EvaluateStepJob(Step):
    """
    Represents a step in the credit fraud pipeline for model evaluation.

    Args:
        context (CreditFraudPipelineContext): The pipeline context.
        image_uri (str): The URI of the Docker image for the script processor.

    Attributes:
        context (CreditFraudPipelineContext): The pipeline context.
        eval_processor (ScriptProcessor): The script processor for evaluation.
        evaluation_report (PropertyFile): The property file for evaluation results.

    """

    def __init__(self, context: CreditFraudPipelineContext, image_uri: str):
        self.context = context

        self.context.s3_script_manager.upload_script(
            source_directory=context.cfg["Global"]["JobsScriptsFolder"],
            script_name="evaluate.py",
        )

        self.eval_processor = ScriptProcessor(
            image_uri=image_uri,
            command=["python3"],
            instance_type=self.context.cfg["Evaluation"]["EvaluateInstanceType"],
            instance_count=1,
            base_job_name=f"{self.context.cfg['Global']['BaseJobNamePrefix']}-model-eval",
            sagemaker_session=self.context,
            role=self.context.sagemaker_role,
        )

        self.evaluation_report = PropertyFile(
            name="ModelEvaluationPropertyFile",
            output_name="evaluation",
            path="evaluation.json",
        )

    def build(self, model_artifact_s3_uri: str, test_data_uri: str) -> ProcessingStep:
        """
        Builds the evaluation step of the pipeline.

        Args:
            model_artifact_s3_uri (str): The S3 URI of the model artifact.
            test_data_uri (str): The URI of the test data.

        Returns:
            ProcessingStep: The evaluation step of the pipeline.

        """
        evaluation_step = ProcessingStep(
            name="ModelEvaluate",
            processor=self.eval_processor,
            inputs=[
                ProcessingInput(
                    source=model_artifact_s3_uri,
                    destination="/opt/ml/processing/model",
                ),
                ProcessingInput(
                    source=test_data_uri,
                    destination="/opt/ml/processing/test.parquet",
                ),
            ],
            outputs=[
                ProcessingOutput(
                    destination=f"{self.context.bucket_folder}/runs/{self.context.execution_name}/evaluation",
                    output_name="evaluation",
                    source="/opt/ml/processing/evaluation",
                )
            ],
            code=self.context.s3_script_manager.get_script_uri("evaluate.py"),
            property_files=[self.evaluation_report],
            job_arguments=[
                "--model-algorithm",
                self.context.training_algorithm,
                "--mlflow-arn",
                self.context.mlflow.server_arn,
                "--mlflow-run-id",
                self.context.mlflow.experiment_run_id,
            ],
        )
        return evaluation_step
