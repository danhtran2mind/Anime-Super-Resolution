"""
Refer to docs/scripts/download_dataset_doc.md for detailed instructions on usage.
"""

import argparse
import os
from huggingface_hub import HfApi
from huggingface_hub import snapshot_download
import zipfile

def download_and_extract_dataset(repo_id, huggingface_token, output_dir):
    # Initialize the API
    api = HfApi()

    # Get the repository ID and token from arguments
    repo_id = args.dataset_id
    huggingface_token = args.huggingface_token
    # Define the save path
    save_path = output_dir

    # Create the directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Download the dataset
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=save_path,
        token=huggingface_token  # Pass the token if provided
    )

    # Look for zip files in the output directory and extract them
    for file_name in os.listdir(save_path):
        if file_name.endswith('.zip'):
            zip_path = os.path.join(save_path, file_name)
            extract_path = os.path.join(save_path, file_name.replace('.zip', '-raw'))
            
            # Create extraction directory
            os.makedirs(extract_path, exist_ok=True)
            
            # Extract the zip file
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
                print(f"Extracted {zip_path} to {extract_path}")
                
                # Remove the zip file after extraction
                os.remove(zip_path)
                print(f"Removed {zip_path}")
            except Exception as e:
                print(f"Error extracting {zip_path}: {e}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Download a dataset from Hugging Face and extract any zip files")
    parser.add_argument('--dataset_id', type=str, required=True, help="Hugging Face dataset repository ID (e.g., ejhf743b/anime-images)")
    parser.add_argument('--huggingface_token', type=str, default=None, help="Hugging Face API token (optional, can also use HF_TOKEN env variable)")
    parser.add_argument('--output_dir', type=str, default="./data", help="Directory to save the downloaded dataset (default: ./data)")
    
    # Parse arguments
    args = parser.parse_args()
    download_and_extract_dataset(args.dataset_id, args.huggingface_token, args.output_dir)