"""Focused checks for the PySpark preprocessing feature mapping.

`preprocess_pyspark.py` is uploaded and executed standalone inside a SageMaker
PySpark processing container, so it cannot rely on the `credit_fraud` package
being installed. These tests import the script by path and stub the minimal
`pyspark` surface it touches, then verify that the 28 scaled PCA features are
mapped to the correct 0-based vector elements.
"""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "credit_fraud"
    / "pipeline"
    / "jobs"
    / "preprocess_pyspark.py"
)

_PYSPARK_MODULES = (
    "pyspark",
    "pyspark.sql",
    "pyspark.sql.functions",
    "pyspark.sql.window",
    "pyspark.sql.types",
    "pyspark.ml",
    "pyspark.ml.feature",
    "pyspark.ml.functions",
)


def _install_fake_pyspark():
    """Register a minimal fake ``pyspark`` package tree in ``sys.modules``."""
    pyspark = types.ModuleType("pyspark")
    pyspark_sql = types.ModuleType("pyspark.sql")
    pyspark_sql_functions = types.ModuleType("pyspark.sql.functions")
    pyspark_sql_window = types.ModuleType("pyspark.sql.window")
    pyspark_sql_types = types.ModuleType("pyspark.sql.types")
    pyspark_ml = types.ModuleType("pyspark.ml")
    pyspark_ml_feature = types.ModuleType("pyspark.ml.feature")
    pyspark_ml_functions = types.ModuleType("pyspark.ml.functions")

    pyspark.sql = pyspark_sql
    pyspark.ml = pyspark_ml
    pyspark_sql.functions = pyspark_sql_functions
    pyspark_sql.window = pyspark_sql_window
    pyspark_sql.types = pyspark_sql_types
    pyspark_ml.feature = pyspark_ml_feature
    pyspark_ml.functions = pyspark_ml_functions

    pyspark_sql.SparkSession = MagicMock()
    pyspark_sql.DataFrame = MagicMock()
    pyspark_sql_window.Window = MagicMock()
    pyspark_sql_types.StructField = MagicMock()
    pyspark_sql_types.StructType = MagicMock()
    pyspark_sql_types.DoubleType = MagicMock()
    pyspark_sql_types.IntegerType = MagicMock()
    pyspark_sql_types.FloatType = MagicMock()
    pyspark_ml_feature.MinMaxScaler = MagicMock()
    pyspark_ml_feature.RobustScaler = MagicMock()
    pyspark_ml_feature.VectorAssembler = MagicMock()
    pyspark_ml.Pipeline = MagicMock()
    pyspark_ml_functions.vector_to_array = MagicMock()

    sys.modules["pyspark"] = pyspark
    sys.modules["pyspark.sql"] = pyspark_sql
    sys.modules["pyspark.sql.functions"] = pyspark_sql_functions
    sys.modules["pyspark.sql.window"] = pyspark_sql_window
    sys.modules["pyspark.sql.types"] = pyspark_sql_types
    sys.modules["pyspark.ml"] = pyspark_ml
    sys.modules["pyspark.ml.feature"] = pyspark_ml_feature
    sys.modules["pyspark.ml.functions"] = pyspark_ml_functions


@pytest.fixture()
def preprocess_module():
    original_modules = {name: sys.modules.get(name) for name in _PYSPARK_MODULES}
    _install_fake_pyspark()
    try:
        spec = importlib.util.spec_from_file_location("preprocess_pyspark", SCRIPT_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules["preprocess_pyspark"] = module
        spec.loader.exec_module(module)
        yield module
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
        sys.modules.pop("preprocess_pyspark", None)


class _VectorArray:
    """Fake replacement for ``vector_to_array(...).getItem(i)``."""

    def __init__(self, column):
        self.column = column
        self.index = None

    def getItem(self, index):
        self.index = index
        return self


class _FakeDataFrame:
    """Minimal DataFrame fake that records ``withColumn`` calls."""

    def __init__(self, columns):
        self._columns = list(columns)
        self.with_calls = []

    @property
    def columns(self):
        return self._columns

    def withColumn(self, name, expression):
        self.with_calls.append((name, expression))
        return self

    def drop(self, *columns):
        self._columns = [c for c in self._columns if c not in columns]
        return self

    def select(self, *columns):
        self._columns = list(columns)
        return self


def test_scaled_feature_vector_index_covers_all_28_features(preprocess_module):
    expected = {f"V{i}": i - 1 for i in range(1, 29)}
    assert preprocess_module.SCALED_FEATURE_VECTOR_INDEX == expected


def test_transform_dataframe_maps_features_to_correct_vector_elements(
    preprocess_module,
):
    preprocess_module.vector_to_array = lambda column: _VectorArray(column)

    input_columns = [f"V{i}" for i in range(1, 29)] + [
        "Amount",
        "Class",
        "min_max_features",
        "min_max_features_scaled",
        "Amount_vec",
        "Amount_scaled",
    ]
    dataframe = _FakeDataFrame(input_columns)

    scaler_model = MagicMock()
    scaler_model.transform.return_value = dataframe
    preprocess_module.scalerModel = scaler_model

    preprocess_module.transform_dataframe(dataframe)

    records = {
        name: (expression.column, expression.index)
        for name, expression in dataframe.with_calls
    }

    expected = {f"V{i}": ("min_max_features_scaled", i - 1) for i in range(1, 29)}
    expected["Amount"] = ("Amount_scaled", 0)

    assert records == expected
