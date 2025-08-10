import argparse
import yaml
import os
import requests
from huggingface_hub import snapshot_download, hf_hub_download

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def download_model(model_config, full_ckpts=False):
    model_id = model_config['model_id']
    local_dir = model_config['local_dir']
    platform = model_config['platform']
    url = model_config.get('url')  # Get URL if it exists, None otherwise
    filename = model_config.get('filename')

    # Ensure the local directory exists
    os.makedirs(local_dir, exist_ok=True)
    
    if platform == "HuggingFace":
        if full_ckpts:
            print(f"Downloading full model {model_id} from HuggingFace to {local_dir}")
            snapshot_download(
                repo_id=model_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                allow_patterns=["*.pth", "*.bin", "*.json"],  # Common model file extensions
                ignore_patterns=["*.md", "*.txt"],  # Ignore non-model files
            )
            print(f"Successfully downloaded {model_id} to {local_dir}")
        else:
            if not filename:
                raise ValueError(f"No filename provided for model: {model_id}")
            print(f"Downloading file {filename} for model {model_id} from HuggingFace to {local_dir}")
            hf_hub_download(
                repo_id=model_id,
                filename=filename,
                local_dir=local_dir,
            )
            print(f"Successfully downloaded {filename} to {local_dir}")
    elif platform == "GitHub":
        if not url:
            raise ValueError(f"No URL provided for GitHub model: {model_id}")
        if not filename:
            filename = os.path.basename(url)
        full_path = os.path.join(local_dir, filename)
        print(f"Downloading model {model_id} from GitHub URL {url} to {full_path}")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(full_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"Successfully downloaded {model_id} to {full_path}")
        else:
            raise ValueError(f"Failed to download {model_id} from {url}: HTTP {response.status_code}")
    else:
        raise ValueError(f"Unsupported platform: {platform}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download model checkpoints from HuggingFace or GitHub.")
    parser.add_argument('--config', type=str, default="configs/model_ckpts.yaml",
                        help="Path to the YAML configuration file")
    parser.add_argument('--full_ckpts', action='store_true',
                        help="if true download all models using snapdownload, else just download model with for_inference in yaml")
    parser.add_argument('--include_base_model', action='store_true',
                        help="if true download all model base_model true and false, else just download base_model is false")
    args = parser.parse_args()

    # Load the YAML configuration
    config = load_config(args.config)

    # Iterate through models in the config
    for model_config in config:
        if not args.full_ckpts and not model_config.get('for_inference', False):
            continue
        if not args.include_base_model and model_config.get('base_model', False):
            continue
        download_model(model_config, full_ckpts=args.full_ckpts)