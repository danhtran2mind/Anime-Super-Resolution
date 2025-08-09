import argparse
from inference.real_esrgan_inference import RealESRGAN
from PIL import Image
import numpy as np
import torch
import os
import yaml
from huggingface_hub import hf_hub_download

def get_model_checkpoint(model_id, models_config):
    try:
        config_list = yaml.safe_load(models_config)
    except yaml.YAMLError as e:
        print(f"Error loading YAML: {e}")
        exit(1)
    # Find the specific model configuration
    model_config = next((item for item in config_list if item['model_id'] == model_id), None)
    if model_config is None:
        print("Error: Model ID 'danhtran2mind/Real-ESRGAN-Anime-finetuning' not found in configuration.")
        exit(1)
    model_path = os.path.join(model_config["local_dir"], model_config["filename"])
    if not os.path.exists(model_path):
        hf_hub_download(repo_id=model_config["model_id"],
                        filename=model_config["filename"], 
                        cache_dir=model_config["local_dir"],
                        local_dir_use_symlinks=False)
        
        print('Weights downloaded to:', model_path)
    return model_path

def infer(input_path, model_id, models_config, outer_scale, inner_scale=4, output_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = RealESRGAN(device, scale=inner_scale)
    model_path =  get_model_checkpoint(model_id, models_config)
    model.load_weights(model_path)

    image = Image.open(input_path).convert('RGB')
    
    output_image = model.predict(image)
    
    if output_path:
        output_image.save(output_path)
    # else:
    #     # If no output path is provided, create a default output path
    #     output_path = input_path.rsplit('.', 1)[0] + '_out.png'
    #     output_image.save(output_path)
    
    return output_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Super-resolution for anime images using Real-ESRGAN")
    parser.add_argument('--input_path', type=str, required=True, help="Path to the input image")
    parser.add_argument('--output_path', type=str, default=None, help="Path to save the output image")
    parser.add_argument('--model_id', type=str, required=True, help="Model ID for Real-ESRGAN")
    parser.add_argument('--models_config', type=str, required=True, help="Path to the models configuration YAML file")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for inference (not used in this implementation)")
    parser.add_argument('--outer_scale', type=int, required=True, help="Outer scale for super-resolution")
    parser.add_argument('--inner_scale', type=int, default=4, help="Inner scale for the model")
    
    args = parser.parse_args()
    
    # Read the models_config file
    with open(args.models_config, 'r') as file:
        models_config = file.read()
    
    # Call infer with the correct arguments
    infer(args.input_path, args.model_id, models_config, 
          args.outer_scale, args.inner_scale, args.output_path)