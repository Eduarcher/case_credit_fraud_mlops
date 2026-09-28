import json
import os
import logging
from typing import Any
from functools import reduce
from operator import add
import ast

import boto3


# Grab environment variables
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME")

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def _dict_to_csv_bytes(data_body: dict):
    # Convert dict to list of lists (rows)
    rows = list(zip(*data_body["data"].values()))
    # Convert each row to a comma-separated string and join rows with newline
    csv_string = "\n".join([",".join(map(str, row)) for row in rows])
    return csv_string.encode()


def _parse_response(query_response):
    query_body = query_response["Body"].read()
    if query_body[0] == 123:
        model_predictions_lists = json.loads(query_body)["probabilities-1d"]
        model_predictions = reduce(add, model_predictions_lists)
    else:
        body_decoded_list = query_body.decode("utf-8").split("\n")
        model_predictions = [float(num) for num in body_decoded_list if num]
    return model_predictions


def lambda_handler(event: dict, context: Any = None):
    client = boto3.client("runtime.sagemaker")

    data_body = ast.literal_eval(event["body"])
    if type(data_body) is not dict or "data" not in data_body.keys():
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "*/*"},
            "body": "Invalid data on the request body.",
        }

    try:
        payload = _dict_to_csv_bytes(data_body)
        response = client.invoke_endpoint(
            EndpointName=ENDPOINT_NAME, Body=payload, ContentType="text/csv"
        )
        preds = _parse_response(response)
        response = {
            "statusCode": 200,
            "headers": {"Content-Type": "*/*"},
            "body": str(preds),
        }
        return response
    except Exception:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "*/*"},
            "body": "An error occurred while processing the request.",
        }
