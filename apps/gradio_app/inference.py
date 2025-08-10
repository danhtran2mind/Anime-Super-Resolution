import gradio as gr
from PIL import Image
import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.anime_super_resolution.infer import infer

def run_inference(input_image, model_id, outer_scale):
    if not input_image:
        return None, "Please upload an image."
    
    models_config_path = "configs/model_ckpts.yaml"
    try:
        output_image = infer(
            input_path=input_image,
            model_id=model_id,
            models_config_path=models_config_path,
            outer_scale=outer_scale,
        )
        return output_image, "Inference completed successfully!"
    except Exception as e:
        return None, f"Error during inference: {str(e)}"