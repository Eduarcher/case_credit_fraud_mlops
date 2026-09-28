import os

import pytest
from unittest.mock import patch, mock_open
import credit_fraud.utils.helpers as helpers


def test_get_version_success():
    mocked_content = """
    [project]
    name = "Mocked Project"
    version = "1.0.0"
    """
    with patch("builtins.open", mock_open(read_data=mocked_content)) as mock_file:
        version = helpers.PyProjectHelper.get_version("pyproject.toml")
        assert version == "1.0.0"
        mock_file.assert_called_once_with("pyproject.toml", "r")


def test_get_version_fail():
    mocked_content = """
    [project]
    name = "Mocked Project"
    """
    with pytest.raises(ValueError) as _:
        with patch("builtins.open", mock_open(read_data=mocked_content)) as _:
            helpers.PyProjectHelper.get_version("pyproject.toml")


def test_environ_get_bool():
    os.environ["TEST"] = "true"
    assert helpers.EnvironHelper.environ_get_bool("TEST")
    os.environ["TEST"] = "True"
    assert helpers.EnvironHelper.environ_get_bool("TEST")
    os.environ["TEST"] = ""
    assert not helpers.EnvironHelper.environ_get_bool("TEST")


def test_S3ScriptManager_instance(mocker):
    script_manager = helpers.S3ScriptManager(
        region="us-east-1",
        destination_bucket_name="dest-bucket",
        destination_folder="dest-folder",
    )
    assert script_manager.region == "us-east-1"
    assert script_manager.destination_bucket_name == "dest-bucket"
    assert script_manager.destination_folder == "dest-folder"
    assert script_manager.logger
    assert type(script_manager.get_script_uri("test.py")) is str
