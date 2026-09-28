import os
import logging

import boto3


# Grab environment variables
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME")

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def lambda_handler(event, context=None):
    client = boto3.client("sagemaker")

    endpoints_list = client.list_endpoints()["Endpoints"]

    for endpoint in endpoints_list:
        if ENDPOINT_NAME == endpoint["EndpointName"]:
            return {"status": endpoint["EndpointStatus"] == "InService"}
    return {"status": False}
