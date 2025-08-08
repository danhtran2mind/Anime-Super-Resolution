import argparse
import yaml
import os
import requests
from huggingface_hub import snapshot_download

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def download_model(model_id, local_dir, platform, url=None):
    # Ensure the local directory exists
    os.makedirs(local_dir, exist_ok=True)
    
    if platform == "HuggingFace":
        print(f"Downloading model {model_id} from HuggingFace to {local_dir}")
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            allow_patterns=["*.pth", "*.bin", "*.json"],  # Common model file extensions
            ignore_patterns=["*.md", "*.txt"],  # Ignore non-model files
        )
        print(f"Successfully downloaded {model_id} to {local_dir}")
    elif platform == "GitHub":
        if not url:
            raise ValueError(f"No URL provided for GitHub model: {model_id}")
        print(f"Downloading model {model_id} from GitHub URL {url} to {local_dir}")
        # Extract filename from URL
        filename = os.path.join(local_dir, os.path.basename(url))
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"Successfully downloaded {model_id} to {filename}")
        else:
            raise ValueError(f"Failed to download {model_id} from {url}: HTTP {response.status_code}")
    else:
        raise ValueError(f"Unsupported platform: {platform}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download model checkpoints from HuggingFace or GitHub.")
    parser.add_argument('--config', type=str, default="configs/model_ckpts.yaml",
                        help="Path to the YAML configuration file")
    args = parser.parse_args()

    # Load the YAML configuration
    config = load_config(args.config)

    # Iterate through models in the config
    for model_config in config:
        model_id = model_config['model_id']
        local_dir = model_config['local_dir']
        platform = model_config['platform']
        url = model_config.get('url')  # Get URL if it exists, None otherwise
        download_model(model_id, local_dir, platform, url)