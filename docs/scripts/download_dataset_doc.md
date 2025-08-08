# Download Dataset Script

## Overview
Python script to download a dataset from Hugging Face Hub and extract zip files using `huggingface_hub`.

## Accessing the Anime Images Dataset

To download the Anime Images dataset, please contact me through the Issue tab on GitHub: [https://github.com/danhtran2mind/Anime-Super-Resolution/issues](https://github.com/danhtran2mind/Anime-Super-Resolution/issues).

Once you reach out, I will provide:

-   A direct link to the dataset.
-   Access permissions for the dataset.
-   Detailed instructions for downloading.

To download the dataset, use the following command after receiving the necessary credentials:

```bash
python scripts/download_datasets.py \
    --dataset_id "<huggingface_dataset_id>" \
    --huggingface_token "<your_huggingface_token>"
```
**Notes**:

-   Replace <huggingface_dataset_id> with the dataset ID provided.
-   Replace <your_huggingface_token> with the Hugging Face token I share with you.
-   Ensure you have the required dependencies installed (e.g., Python, Hugging Face CLI).
-   For any issues, refer to the GitHub repository or contact me via the Issue tab.

## Prerequisites
- Python 3.10+
- Install: `pip install huggingface_hub`
- Optional: Hugging Face API token for private datasets

## Usage
```bash
python download_dataset.py --dataset_id <dataset_id> [--huggingface_token <token>] [--output_dir <directory>]
```

### Arguments
| Argument            | Type   | Required | Description                                           |
|---------------------|--------|----------|-------------------------------------------------------|
| `--dataset_id`      | String | Yes      | Dataset ID (e.g., `ejhf743b/anime-images`)            |
| `--huggingface_token`| String | No       | API token for private datasets                        |
| `--output_dir`      | String | No       | Save directory (default: `./data`)                    |

### Example
```bash
python download_dataset.py --dataset_id ejhf743b/anime-images --output_dir ./my_datasets
```

## Functionality
1. Initializes Hugging Face API client.
2. Creates output directory if needed.
3. Downloads dataset to `output_dir` using `snapshot_download`.
4. Extracts `.zip` files to `<zip_filename>-raw` subdirectories and deletes zips.
5. Prints extraction status or errors.

## Notes
- Use `HF_TOKEN` env variable instead of `--huggingface_token` if preferred.
- Handles only `.zip` files.
- Errors during extraction are logged but do not stop the script.

## Example Output
```bash
Extracted ./data/dataset.zip to ./data/dataset-raw
Removed ./data/dataset.zip
```

## License
Provided as-is. Check dataset license on Hugging Face Hub.