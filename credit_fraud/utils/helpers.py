import os
import logging
import json

import boto3
import toml


class PyProjectHelper:
    @classmethod
    def get_version(cls, path="pyproject.toml"):
        with open(path, "r") as file:
            pyproject_toml = toml.load(file)
            python_version = pyproject_toml.get("project", {}).get("version")
            if python_version:
                return python_version
            else:
                raise ValueError(
                    "Project version not found inside pyproject.toml file."
                )


class EnvironHelper:
    @classmethod
    def environ_get_bool(cls, key):
        return os.environ.get(key, "false").lower() == "true"


class S3ScriptManager:
    def __init__(
        self,
        region: str,
        destination_bucket_name: str,
        destination_folder: str,
        logger=logging,
    ):
        self.logger = logger
        self.region = region
        self.destination_bucket_name = destination_bucket_name
        self.destination_folder = destination_folder
        self.s3_client = boto3.client("s3", region_name=region)

    def upload_script(self, source_directory, script_name):
        self.logger.info(f"Uploading script '{script_name}' to S3")
        res = self.s3_client.upload_file(
            Filename=f"{source_directory}/{script_name}",
            Bucket=self.destination_bucket_name,
            Key=f"{self.destination_folder}/{script_name}",
        )
        return res

    def get_script_uri(self, script_name):
        return f"s3://{self.destination_bucket_name}/{self.destination_folder}/{script_name}"


class SecretManager:
    def __init__(self, region_name="us-east-1", logger=logging):
        session = boto3.session.Session()
        self.client = session.client(
            service_name="secretsmanager", region_name=region_name
        )
        self.logger = logger

    def get_secret(self, secret_name):
        try:
            get_secret_value_response = self.client.get_secret_value(
                SecretId=secret_name
            )
        except Exception as e:
            self.logger.error(f"Error getting secret: {secret_name}")
            raise e
        secret = json.loads(get_secret_value_response["SecretString"])
        return secret


if __name__ == "__main__":
    pass
