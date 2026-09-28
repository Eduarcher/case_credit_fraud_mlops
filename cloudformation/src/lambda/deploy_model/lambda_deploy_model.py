"""
Lambda function creates an endpoint configuration 
and deploys a model to real-time endpoint.
Required parameters for deployment are retrieved from the event object
"""

import json
import os
import logging

import boto3
from botocore.exceptions import ClientError


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

MODEL_MIN_CAPACITY = int(os.environ.get("MODEL_MIN_CAPACITY"))
MODEL_MAX_CAPACITY = int(os.environ.get("MODEL_MAX_CAPACITY"))


def _await_endpoint(sm_client, event):
    waiter = sm_client.get_waiter("endpoint_in_service")
    logger.info("Waiting for endpoint to create...")
    waiter.wait(EndpointName=event["endpoint_name"])
    resp = sm_client.describe_endpoint(EndpointName=event["endpoint_name"])
    logger.info(f"Endpoint Status: {resp['EndpointStatus']}")


def _create_endpoint_config(sm_client, event):
    endpoint_config_name = f"config-{event['model_name']}"
    sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "main",
                "ModelName": event["model_name"],
                "InitialInstanceCount": MODEL_MIN_CAPACITY,
                "InstanceType": event["instance_type"],
                "InitialVariantWeight": 1,
            }
        ],
    )
    return endpoint_config_name


def _update_endpoint(sm_client, event, endpoint_config_name):
    sm_client.update_endpoint(
        EndpointName=event["endpoint_name"],
        EndpointConfigName=endpoint_config_name,
        DeploymentConfig={
            "BlueGreenUpdatePolicy": {
                "TrafficRoutingConfiguration": {
                    "Type": "CANARY",
                    "CanarySize": {"Type": "CAPACITY_PERCENT", "Value": 50},
                    "WaitIntervalInSeconds": 300,
                },
                "TerminationWaitInSeconds": 300,
                "MaximumExecutionTimeoutInSeconds": 1800,
            }
        },
    )
    return True


def _apply_autoscaling(as_client, event):
    resource_id = f"endpoint/{event['endpoint_name']}/variant/main"

    as_client.register_scalable_target(
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        MinCapacity=MODEL_MIN_CAPACITY,
        MaxCapacity=MODEL_MAX_CAPACITY,
    )
    as_client.put_scaling_policy(
        PolicyName="Invocations-ScalingPolicy",
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        PolicyType="TargetTrackingScaling",
        TargetTrackingScalingPolicyConfiguration={
            "TargetValue": 10.0,
            "PredefinedMetricSpecification": {
                "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance",
            },
            "ScaleInCooldown": 600,
            "ScaleOutCooldown": 300,
        },
    )
    return True


def lambda_handler(event, context):
    sm_client = boto3.client("sagemaker")
    as_client = boto3.client("application-autoscaling")

    logger.info("Received event: " + json.dumps(event, indent=2))

    logger.info("Creating endpoint config")
    endpoint_config_name = _create_endpoint_config(sm_client, event)

    try:
        logger.info("Trying to update endpoint")
        _update_endpoint(sm_client, event, endpoint_config_name)
        _await_endpoint(sm_client, event)
        logger.info("Endpoint updated successfully!")
    except ClientError as ex:
        if "Could not find endpoint" not in str(ex):
            raise
        logger.warning("Endpoint not found! Creating new endpoint")
        sm_client.create_endpoint(
            EndpointName=event["endpoint_name"], EndpointConfigName=endpoint_config_name
        )
        _await_endpoint(sm_client, event)
        logger.info("Endpoint created successfully!")

        logger.info("Applying application auto-scaling configuration")
        _apply_autoscaling(as_client, event)
        _await_endpoint(sm_client, event)


if __name__ == "__main__":
    pass
