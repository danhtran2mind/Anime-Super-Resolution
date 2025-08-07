import argparse
import yaml
from huggingface_hub import snapshot_download

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def download_model(model_id, local_dir, platform):
    if platform == "HuggingFace":
        print(f"Downloading model {model_id} to {local_dir}")
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            allow_patterns=["*.pth", "*.bin", "*.json"],  # Common model file extensions
            ignore_patterns=["*.md", "*.txt"],  # Ignore non-model files
        )
        print(f"Successfully downloaded {model_id} to {local_dir}")
    else:
        raise ValueError(f"Unsupported platform: {platform}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download model checkpoints from HuggingFace.")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()

    # Load the YAML configuration
    config = load_config(args.config)

    # Iterate through models in the config
    for model_config in config:
        model_id = model_config['model_id']
        local_dir = model_config['local_dir']
        platform = model_config['platform']
        download_model(model_id, local_dir, platform)
