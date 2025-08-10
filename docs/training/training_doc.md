# Training Guide

This document provides instructions on how to run the Real-ESRGAN training script using the provided Python code. The script allows you to train a Real-ESRGAN model with customizable configurations through command-line arguments.

## Prerequisites
- Python 3.10+ installed
- Real-ESRGAN repository cloned and dependencies installed
- A valid YAML configuration file for training
- Required Python packages: `yaml`, `shutil`, and other dependencies listed in the Real-ESRGAN repository

## Script Overview
The script (`train.py`) executes the Real-ESRGAN training process by invoking the `train.py` script from the Real-ESRGAN repository. It supports various command-line arguments to customize the training process and handles moving the experiment output to a specified directory.

## Command-Line Arguments
The script accepts the following command-line arguments:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | String | `configs/Real-ESRGAN-Anime-finetuning.yml` | Path to the configuration YAML file that defines training parameters. |
| `--launcher` | Choice (`none`, `pytorch`, `slurm`) | `none` | Job launcher for distributed training. Use `none` for single-node training, `pytorch` for PyTorch distributed, or `slurm` for SLURM-based clusters. |
| `--auto_resume` | Flag | `False` | If specified, automatically resumes training from the latest checkpoint. |
| `--debug` | Flag | `False` | If specified, enables debug mode for additional logging or debugging information. |
| `--local_rank` | Integer | `0` | Local rank for distributed training (used in multi-GPU setups). |
| `--force_yml` | List of strings | `None` | Force updates to the YAML configuration file. Example: `train:ema_decay=0.999` to override specific settings. |
| `--output_model_dir` | String | `ckpts` | Path to move the experiment directory (containing model checkpoints and logs) after training completes. |

## Usage Instructions
1. **Prepare the Environment**:
   - Ensure the Real-ESRGAN repository is available in the `third_party/Real-ESRGAN` directory relative to the script.
   - Install required dependencies (refer to the Real-ESRGAN repository's documentation).

2. **Create or Modify a Configuration File**:
   - Prepare a YAML configuration file (e.g., `Real-ESRGAN-Anime-finetuning.yml`) specifying training parameters like dataset paths, model architecture, and hyperparameters.
   - The configuration file must include a `name` field to identify the experiment.

3. **Run the Training Script**:
   Use the following command to start training with default settings:
   ```bash
   python train.py
   ```

   To customize the training, use the command-line arguments. Examples:
   - Train with a specific configuration file and enable auto-resume:
     ```bash
     python train.py --config configs/my_config.yml --auto_resume
     ```
   - Override YAML settings and specify an output directory:
     ```bash
     python train.py --config configs/my_config.yml \
        --output_model_dir "</path/to/output>"
     ```

4. **Output**:
   - The script runs the Real-ESRGAN training process using the specified configuration.
   - After training, the experiment directory (named after the `name` field in the YAML file) is moved from `third_party/Real-ESRGAN/experiments/` to the directory specified by `--output_model_dir`.
   - If the source experiment directory does not exist, a warning is printed.
   - Errors during training or directory moving are caught and reported, with the script exiting on failure.

## Example Workflow
To train a Real-ESRGAN model for anime-style image upscaling:
1. Ensure the `Real-ESRGAN-Anime-finetuning.yml` file is configured with the correct dataset paths and model settings.
2. Run the following command:
   ```bash
   python train.py --config configs/Real-ESRGAN-Anime-finetuning.yml --output_model_dir models/anime_model --auto_resume
   ```
3. The training process will start, and upon completion, the experiment directory will be moved to `models/anime_model`.

## Notes
- Ensure the YAML configuration file exists at the specified path, or the script will raise a `FileNotFoundError`.
- The `PYTHONPATH` environment variable is modified to include the Real-ESRGAN directory for proper module resolution.
- If using distributed training (`--launcher pytorch` or `--launcher slurm`), ensure the environment is set up for multi-GPU or cluster-based training.
- The `--force_yml` argument allows dynamic updates to the YAML configuration without modifying the file directly.

## Troubleshooting
- **Error: Configuration file not found**:
  - Verify the `--config` path is correct and the file exists.
- **Error: Training failed**:
  - Check the Real-ESRGAN repository's documentation for troubleshooting training issues.
  - Ensure all dependencies are installed and compatible.
- **Warning: Source directory does not exist**:
  - Confirm that the experiment name in the YAML file matches the expected directory in `third_party/Real-ESRGAN/experiments/`.
- **Permission errors when moving directories**:
  - Ensure the script has write permissions for the `--output_model_dir` path.

For further details on Real-ESRGAN, refer to the official repository documentation.