# Inference Documentation

This document provides instructions for using the Real-ESRGAN inference script to perform super-resolution on anime images. The script uses a pre-trained Real-ESRGAN model to upscale images, with configurable input and output options.

## Prerequisites

- **Python Libraries**: Ensure the following Python packages are installed:
  - `argparse`
  - `PIL` (Pillow)
  - `numpy`
  - `torch`
  - `opencv-python` (cv2)
  - `pyyaml`
  - `huggingface_hub`
- **Model Configuration**: A YAML file specifying model details (model ID, local directory, and filename).
- **Input Image**: A valid image file (e.g., PNG, JPEG) in RGB format.
- **Hardware**: CUDA-compatible GPU (optional, for faster processing) or CPU.

## Script Overview

The script (`inference.py`) performs super-resolution on an input image using the Real-ESRGAN model. It supports:
- Downloading model weights from Hugging Face if not available locally.
- Upscaling images using an inner scale (model-specific) and an optional outer scale (post-processing resizing).
- Saving the upscaled image to a specified output path or a default location.

## Command-Line Arguments

The script accepts the following command-line arguments:

| Argument                | Type | Required | Default | Description                                                                 |
|-------------------------|------|----------|---------|-----------------------------------------------------------------------------|
| `--input_path`          | str  | Yes      | None    | Path to the input image file (e.g., `image.png`).                           |
| `--output_path`         | str  | No       | None    | Path to save the upscaled image. If not provided, the image is returned but not saved automatically. |
| `--model_id`            | str  | Yes      | None    | Model ID for the Real-ESRGAN model (e.g., `danhtran2mind/Real-ESRGAN-Anime-finetuning`). |
| `--models_config_path`  | str  | Yes      | None    | Path to the YAML configuration file containing model details.                |
| `--batch_size`          | int  | No       | 1       | Batch Ascertain batch size (not used in this implementation).                    |
| `--outer_scale`        | int  | Yes      | None    | Desired final scale factor for super-resolution (e.g., 4, 8).               |
| `--inner_scale`        | int  | No       | 4       | Internal scale factor used by the model (typically 4).                      |

## Usage

1. **Prepare the Models Configuration File**:
   Create a YAML file (e.g., `models_config.yaml`) with the following structure:

   ```yaml
   - model_id: "danhtran2mind/Real-ESRGAN-Anime-finetuning"
     local_dir: "./weights"
     filename: "model.pth"
   ```

   This file specifies the model ID, local directory for weights, and the filename of the model checkpoint.

2. **Run the Script**:
   Use the following command to run the inference:

   ```bash
   python inference.py \
      --input_path path/to/input/image.png \
      --output_path path/to/output/image.png \
      --model_id danhtran2mind/Real-ESRGAN-Anime-finetuning \
      --models_config_path path/to/models_config.yaml \
      --outer_scale 4
   ```

   Example:

   ```bash
   python inference.py \
      --input_path input.png \
      --output_path output.png \
      --model_id danhtran2mind/Real-ESRGAN-Anime-finetuning \
      --models_config_path models_config.yaml \
      --outer_scale 8
   ```

3. **Output**:
   - The script processes the input image and applies super-resolution.
   - If `--output_path` is provided, the upscaled image is saved to the specified path.
   - If `--outer_scale` differs from `--inner_scale`, the output image is resized using OpenCV's `INTER_CUBIC` (for upscaling) or `INTER_AREA` (for downscaling) interpolation.

## How It Works

1. **Model Loading**:
   - The script reads the `models_config_path` YAML file to locate the model configuration.
   - If the model weights are not found locally, they are downloaded from the Hugging Face Hub using the specified `model_id` and `filename`.
   - The Real-ESRGAN model is initialized with the specified `inner_scale` and loaded with the weights.

2. **Image Processing**:
   - The input image is opened and converted to RGB format using Pillow.
   - The Real-ESRGAN model upscales the image by the `inner_scale` factor.
   - If `outer_scale` differs from `inner_scale`, the image is further resized to achieve the desired scale using OpenCV.

3. **Output Handling**:
   - The upscaled image is saved to `output_path` if provided.
   - The processed image is returned as a Pillow Image object.

## Notes

- **Device Selection**: The script automatically uses CUDA if available; otherwise, it falls back to CPU.
- **Model Weights**: Ensure the `local_dir` specified in the YAML file exists or is writable for downloading weights.
- **Outer vs. Inner Scale**:
  - `inner_scale` is the scale factor used by the Real-ESRGAN model (typically fixed at 4).
  - `outer_scale` is the final desired scale, achieved through additional resizing if necessary.
- **Batch Size**: The `--batch_size` argument is included but not used in this implementation, as the script processes one image at a time.

## Example Models Configuration File

Here is an example `models_config.yaml`:

<xaiArtifact artifact_id="0b60a214-8c91-48ed-ad50-fae3467a0508" artifact_version_id="35ae4a0a-da96-44d0-b8ed-7c1d62b59527" title="models_config.yaml" contentType="text/yaml">

```yaml
- model_id: "danhtran2mind/Real-ESRGAN-Anime-finetuning"
  local_dir: "./weights"
  filename: "model.pth"
```