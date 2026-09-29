# Vendored Dependencies

This project includes a small set of dependencies bundled directly into the repository due to AWS SageMaker limitations at the time of development (2024). These are intentionally minimal and documented here for transparency.

## SageMaker JumpStart Utilities (LGBM training job)

**Location:** `credit_fraud/pipeline/jobs/lgbm/lib/`

**Packages** (2 wheels, ~12 KB total):
- `sagemaker_jumpstart_tabular_script_utilities-1.0.0-py2.py3-none-any.whl`
- `sagemaker_jumpstart_prepack_script_utilities-1.0.0-py2.py3-none-any.whl`

**Why vendored:** These are internal AWS SageMaker utility packages distributed inside the LightGBM built-in training container. They are required by the LGBM training script for data preparation (`prepare_data`, `get_categorical_features_index`), model info saving (`save_model_info`), and inference code packaging (`copy_inference_code`). They are not published to PyPI and therefore cannot be declared as regular `pip` dependencies.

**Source:** Extracted from the SageMaker JumpStart LightGBM built-in algorithm container (`lightgbm-classification-model`, version `2.1.0`).

**How to update:** These packages are versioned by AWS and tied to the SageMaker container lifecycle. If updating the LightGBM container version, extract the updated wheels from the new container image and replace them here, then update `requirements.txt` to reference the new filenames.

**License:** These wheels are distributed under the Apache License 2.0. The full license text and attribution are provided in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The rest of this repository is governed by the [LICENSE](LICENSE) file.

## All other dependencies

All other Python packages (LightGBM, Dask, Distributed, MLflow, etc.) are declared as standard PyPI dependencies in their respective `requirements.txt` files. The Lambda layer for MLflow is built at deployment time by `pip install` from its own `requirements.txt`.

## Former vendored items (removed for public release)

The following were removed in favor of standard dependency declarations or documented download URLs:

- **MySQL JDBC driver (`mysql-connector-j-9.0.0.jar`):** Required for PySpark RDS preprocessing. It is fetched from Maven Central at build time by `buildspec.yml` (into the git-ignored `dependencies/` directory) instead of being committed. Only needed when `SourceMethod: rds` is configured.
- **Mlflow-skinny Lambda layer:** Now built at deployment time via `pip install -r requirements.txt` in `cloudformation/install.sh`, using the canonical AWS Lambda layer build pattern.
