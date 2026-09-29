"""Focused checks for the API Gateway token authorizer.

`lambda_authorizer.py` is packaged standalone for AWS Lambda, so it cannot rely
on the `credit_fraud` package. These tests import the script by path and verify
the allow/deny policy returned for valid, invalid, and missing tokens, and that
secret-read failures fail closed.
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "cloudformation"
    / "src"
    / "lambda"
    / "authorizer"
    / "lambda_authorizer.py"
)

METHOD_ARN = "arn:aws:execute-api:us-east-1:123456789012:api-id/dev/POST/"


@pytest.fixture(scope="module")
def authorizer_module():
    spec = importlib.util.spec_from_file_location("lambda_authorizer", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["lambda_authorizer"] = module
    spec.loader.exec_module(module)
    return module


def _policy_effect(response):
    return response["policyDocument"]["Statement"][0]["Effect"]


def test_valid_token_is_allowed(authorizer_module):
    with patch.object(authorizer_module, "_read_expected_token", return_value="secret"):
        response = authorizer_module.lambda_handler(
            {"authorizationToken": "secret", "methodArn": METHOD_ARN}
        )
    assert _policy_effect(response) == "Allow"
    assert response["policyDocument"]["Statement"][0]["Resource"] == METHOD_ARN


def test_invalid_token_is_denied(authorizer_module):
    with patch.object(authorizer_module, "_read_expected_token", return_value="secret"):
        response = authorizer_module.lambda_handler(
            {"authorizationToken": "wrong", "methodArn": METHOD_ARN}
        )
    assert _policy_effect(response) == "Deny"


def test_missing_token_is_denied(authorizer_module):
    with patch.object(authorizer_module, "_read_expected_token", return_value="secret"):
        response = authorizer_module.lambda_handler({"methodArn": METHOD_ARN})
    assert _policy_effect(response) == "Deny"


def test_secret_read_failure_fails_closed(authorizer_module):
    with patch.object(
        authorizer_module, "_read_expected_token", side_effect=RuntimeError("boom")
    ):
        response = authorizer_module.lambda_handler(
            {"authorizationToken": "secret", "methodArn": METHOD_ARN}
        )
    assert _policy_effect(response) == "Deny"


def test_policy_shape(authorizer_module):
    with patch.object(authorizer_module, "_read_expected_token", return_value="secret"):
        response = authorizer_module.lambda_handler(
            {"authorizationToken": "secret", "methodArn": METHOD_ARN}
        )
    assert response["principalId"] == "user"
    statement = response["policyDocument"]["Statement"][0]
    assert statement["Action"] == "execute-api:Invoke"
    assert statement["Resource"] == METHOD_ARN
