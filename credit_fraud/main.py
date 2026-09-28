"""Main pipeline file used for creating or updating with latest configurations"""

import argparse

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet

from credit_fraud.utils import Logger
from credit_fraud.pipeline.context import CreditFraudPipelineContext
from credit_fraud.pipeline.steps.preprocess import PreprocessStepJob
from credit_fraud.pipeline.steps.train import TrainStepJob
from credit_fraud.pipeline.steps.evaluate import EvaluateStepJob
from credit_fraud.pipeline.steps.create_model import CreateModelStepJob
from credit_fraud.pipeline.steps.register_model import RegisterModelStepJob
from credit_fraud.pipeline.steps.deploy_endpoint import DeployEndpointStepJob


def run():
    """
    Runs the credit fraud pipeline.

    This function sets up and executes the steps of the credit fraud pipeline. 
    It takes command line arguments to control the behavior of the pipeline. 
    The pipeline includes preprocessing, training, evaluation, and
    deployment steps.
    """
    logger = Logger()

    parser = argparse.ArgumentParser()
    parser.add_argument("-cp", "--cache-preprocess", action="store_true")
    parser.add_argument("-ct", "--cache-training", action="store_true")
    parser.add_argument("-ce", "--cache-evaluate", action="store_true")
    parser.add_argument("-dr", "--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--local-run", action="store_true")
    args, _ = parser.parse_known_args()

    logger.info("Loading pipeline context.")
    context = CreditFraudPipelineContext(args=args, logger=logger)

    logger.info("Building pipeline steps.")
    preprocess_step = PreprocessStepJob(context).build()

    train_step_job = TrainStepJob(context)
    train_step = train_step_job.build(
        train_data_uri=preprocess_step.properties.ProcessingOutputConfig.Outputs[
            "train.parquet"
        ].S3Output.S3Uri,
        validation_data_uri=preprocess_step.properties.ProcessingOutputConfig.Outputs[
            "validation.parquet"
        ].S3Output.S3Uri,
    )

    evaluation_model_image_uri = train_step_job.strategy_algorithm.get_image_uri(
        scope="training"
    )
    evaluation_step = EvaluateStepJob(context, evaluation_model_image_uri).build(
        model_artifact_s3_uri=train_step.properties.ModelArtifacts.S3ModelArtifacts,
        test_data_uri=preprocess_step.properties.ProcessingOutputConfig.Outputs[
            "test.parquet"
        ].S3Output.S3Uri,
    )

    inference_model_image_uri = train_step_job.strategy_algorithm.get_image_uri(
        scope="inference"
    )
    create_model_step = CreateModelStepJob(context, inference_model_image_uri).build(
        model_artifact_s3_uri=train_step.properties.ModelArtifacts.S3ModelArtifacts
    )

    register_model_step = RegisterModelStepJob(context).build(
        model_artifact_s3_uri=train_step.properties.ModelArtifacts.S3ModelArtifacts,
    )

    deploy_step = DeployEndpointStepJob(context).build(
        model_name=create_model_step.properties.ModelName,
    )

    cond_gte = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=evaluation_step.name,
            property_file=evaluation_step.property_files[0],
            json_path="classification_metrics.ROC-AUC.value",
        ),
        right=context.pipeline_params["roc_auc_min_threshold"],
    )
    validate_performance_condition_step = ConditionStep(
        name="ValidatePerformanceConditional",
        conditions=[cond_gte],
        if_steps=[create_model_step, register_model_step, deploy_step],
        else_steps=[],
    )

    pipeline = Pipeline(
        name=context.cfg["Global"]["PipelineName"],
        parameters=list(context.pipeline_params.values()),
        steps=[
            preprocess_step,
            train_step,
            evaluation_step,
            validate_performance_condition_step,
        ],
        sagemaker_session=context,
    )

    if args.verbose:
        logger.info(f"Full pipeline description: {pipeline.definition()}")
    if not args.dry_run:
        logger.info("Upserting pipeline manifest.")
        pipeline.upsert(role_arn=context.sagemaker_role)
        logger.info("Starting pipeline.")
        start_response = pipeline.start(execution_display_name=context.execution_name)
        if args.verbose:
            logger.info(f"Pipeline start response: {start_response.describe()}")


if __name__ == "__main__":
    run()
