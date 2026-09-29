"""API Gateway token authorizer.

Validates the ``Authorization`` header against a shared secret stored in AWS
Secrets Manager and returns an IAM policy that allows or denies the request.
"""

import hmac
import json
import logging
import os

import boto3

SECRET_NAME = os.environ.get("API_KEY_SECRET_NAME")

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def _read_expected_token():
    client = boto3.client("secretsmanager")
    secret = client.get_secret_value(SecretId=SECRET_NAME)
    return json.loads(secret["SecretString"])["api_key"]


def lambda_handler(event, context=None):
    token = event.get("authorizationToken") or ""
    method_arn = event.get("methodArn") or ""

    effect = "Deny"
    try:
        expected_token = _read_expected_token()
        if expected_token and hmac.compare_digest(token, expected_token):
            effect = "Allow"
    except Exception:
        logger.exception("Failed to read or validate the API key secret")

    return {
        "principalId": "user",
        "policyDocument": {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Action": "execute-api:Invoke",
                    "Effect": effect,
                    "Resource": method_arn,
                }
            ],
        },
    }
