import logging
import json

import mlflow


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def lambda_handler(event, context=None):
    mlflow.set_tracking_uri(event.get("mlflow_arn"))

    run_id = event.get("run_id")
    model_path = event.get("model_path")
    model_name = event.get("model_name")

    # Construct the full path to the model artifact
    model_uri = f"runs:/{run_id}/{model_path}"

    # Register the model
    try:
        result = mlflow.register_model(model_uri=model_uri, name=model_name)
        logger.info(f"Model registered successfully: {result}")
    except Exception as e:
        logger.error(f"Failed to register model: {e}")
        raise
    return {
        "status_code": 200,
        "body": json.dumps("Model Registered Successfully"),
    }
