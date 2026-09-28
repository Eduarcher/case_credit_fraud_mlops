import logging
import os

import boto3


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

ECS_TASK_DEFINITION_NAME = os.environ.get("ECS_TASK_DEFINITION_NAME")
ECS_FARGATE_CLUSTER_ARN = os.environ.get("ECS_FARGATE_CLUSTER_ARN")
VPC_ID = os.environ.get("VPC_ID")


def lambda_handler(event, context):
    cli = boto3.client("ecs")

    # New Image URI
    new_image_uri = event["CodePipeline.job"]["data"]["actionConfiguration"][
        "configuration"
    ]["UserParameters"]

    # Describe existing Task Definition
    task_definition = cli.describe_task_definition(
        taskDefinition=ECS_TASK_DEFINITION_NAME,
    )["taskDefinition"]

    # Remove invalid args and register updated task definition
    task_definition["containerDefinitions"][0]["image"] = new_image_uri
    remove_args = [
        "compatibilities",
        "registeredAt",
        "registeredBy",
        "status",
        "revision",
        "taskDefinitionArn",
        "requiresAttributes",
    ]
    task_definition = {
        arg: value for arg, value in task_definition.items() if arg not in remove_args
    }
    reg_task_def_response = cli.register_task_definition(**task_definition)

    # Get VPC subnets list
    ec2_resource = boto3.resource("ec2")
    vpc_subnets_collection = ec2_resource.subnets.filter(
        Filters=[{"Name": "vpc-id", "Values": [VPC_ID]}]
    )
    vpc_subnets = [subnet.id for subnet in vpc_subnets_collection]

    # Run new task definition
    task_revision = reg_task_def_response["taskDefinition"]["revision"]
    cli.run_task(
        taskDefinition=f"{ECS_TASK_DEFINITION_NAME}:{task_revision}",
        cluster=ECS_FARGATE_CLUSTER_ARN,
        launchType="FARGATE",
        count=1,
        platformVersion="LATEST",
        networkConfiguration={
            "awsvpcConfiguration": {"subnets": vpc_subnets, "assignPublicIp": "ENABLED"}
        },
    )
    response = boto3.client("codepipeline").put_job_success_result(
        jobId=event["CodePipeline.job"]["id"]
    )
    return response
