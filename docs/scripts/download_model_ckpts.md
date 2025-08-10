# Download Model Checkpoints

This document describes how to use the provided Python script to download model checkpoints from HuggingFace or GitHub, based on a YAML configuration file.

## Overview

The script allows users to download machine learning model checkpoints from platforms like HuggingFace or GitHub. It supports downloading either full model checkpoints or specific files for inference, with options to filter based on whether the model is a base model or intended for inference. The configuration is provided via a YAML file, and the script uses command-line arguments for additional flexibility.

## Requirements

- Python 3.x
- Required Python packages:
  - `argparse` (standard library)
  - `yaml` (install via `pip install pyyaml`)
  - `os` (standard library)
  - `requests` (install via `pip install requests`)
  - `huggingface_hub` (install via `pip install huggingface_hub`)

## Usage

Run the script from the command line with the following options:

```bash
python script_name.py [--config CONFIG_PATH] [--full_ckpts] [--include_base_model] [--base_model_only]
```

### Command-Line Arguments

| Argument               | Description                                                                 | Default Value                     |
|------------------------|-----------------------------------------------------------------------------|-----------------------------------|
| `--config`             | Path to the YAML configuration file.                                        | `configs/model_ckpts.yaml`       |
| `--full_ckpts`         | If specified, downloads all models using `snapshot_download`. Otherwise, downloads only specific files for models marked `for_inference` in the YAML. | Not set (False)                  |
| `--include_base_model` | If specified, downloads all models (both `base_model: true` and `false`). Otherwise, skips models with `base_model: true`. | Not set (False)                  |
| `--base_model_only`    | If specified, downloads only models with `base_model: true`, ignoring `for_inference`. | Not set (False)                  |

### YAML Configuration

The script expects a YAML configuration file (default: `configs/model_ckpts.yaml`) with a list of model configurations. Each model configuration should include:

| Key            | Description                                                                 | Required | Example Value                     |
|----------------|-----------------------------------------------------------------------------|----------|-----------------------------------|
| `model_id`     | Identifier for the model (e.g., HuggingFace repository ID or model name).   | Yes      | `bert-base-uncased`              |
| `local_dir`    | Local directory to save the downloaded model.                              | Yes      | `./models/bert`                  |
| `platform`     | Platform to download from (`HuggingFace` or `GitHub`).                     | Yes      | `HuggingFace`                    |
| `url`          | Direct URL to the model file (required for GitHub platform).                | No       | `https://github.com/.../model.pth` |
| `filename`     | Specific file to download (required for HuggingFace with `--full_ckpts` not set, or GitHub). | No       | `pytorch_model.bin`              |
| `base_model`   | Boolean indicating if the model is a base model.                           | No       | `true` or `false`                |
| `for_inference`| Boolean indicating if the model is used for inference.                     | No       | `true` or `false`                |

#### Example YAML Configuration

```yaml
- model_id: bert-base-uncased
  local_dir: ./models/bert
  platform: HuggingFace
  filename: pytorch_model.bin
  for_inference: true
  base_model: false
- model_id: custom-model
  local_dir: ./models/custom
  platform: GitHub
  url: https://github.com/user/repo/releases/download/v1.0/model.pth
  filename: model.pth
  for_inference: false
  base_model: true
```

## Script Workflow

1. **Parse Command-Line Arguments**:
   - The script uses `argparse` to handle command-line arguments for configuration file path and download options.

2. **Load YAML Configuration**:
   - Reads the specified YAML file to load model configurations using the `load_config` function.

3. **Filter Models**:
   - Based on the command-line arguments, the script filters models to download:
     - If `--base_model_only` is set, only models with `base_model: true` are downloaded.
     - If `--full_ckpts` is not set, only models with `for_inference: true` are downloaded.
     - If `--include_base_model` is not set, models with `base_model: true` are skipped unless explicitly included via `--base_model_only`.

4. **Download Models**:
   - For each model in the configuration that passes the filters, the `download_model` function is called.
   - **HuggingFace**:
     - If `--full_ckpts` is set, uses `snapshot_download` to download the entire model repository (files with `.pth`, `.bin`, or `.json` extensions).
     - Otherwise, uses `hf_hub_download` to download a specific file specified by `filename`.
   - **GitHub**:
     - Downloads the file from the provided `url` using `requests` and saves it to `local_dir` with the specified `filename` (or derived from the URL if not provided).

5. **Error Handling**:
   - Ensures the local directory exists before downloading.
   - Validates required fields (`url` for GitHub, `filename` for non-full HuggingFace downloads).
   - Raises errors for unsupported platforms or failed downloads (e.g., HTTP errors).

## Example Commands

1. **Download specific inference files** (default behavior):
   ```bash
   python script_name.py --config configs/model_ckpts.yaml
   ```
   Downloads only models with `for_inference: true` and `base_model: false`.

2. **Download full model checkpoints**:
   ```bash
   python script_name.py --config configs/model_ckpts.yaml --full_ckpts
   ```
   Downloads full checkpoints for models with `for_inference: true` and `base_model: false`.

3. **Include base models**:
   ```bash
   python script_name.py --config configs/model_ckpts.yaml --include_base_model
   ```
   Downloads models regardless of `base_model` status, but still respects `for_inference` unless `--full_ckpts` is set.

4. **Download only base models**:
   ```bash
   python script_name.py --config configs/model_ckpts.yaml --base_model_only
   ```
   Downloads only models with `base_model: true`, ignoring `for_inference`.

## Notes

- Ensure the YAML configuration file is correctly formatted and accessible.
- For HuggingFace downloads, an internet connection and sufficient disk space are required.
- For GitHub downloads, the provided URL must be a direct link to a downloadable file.
- The script creates the `local_dir` if it does not exist.
- Downloaded files are saved to the specified `local_dir` with their original filenames (or as specified in the YAML).