import os
import json
from PIL import Image
import gradio as gr

def load_examples():
    examples = []
    examples_base_path = os.path.join(os.path.dirname(__file__), "..", "assets", "examples", "Real-ESRGAN-Anime-finetuning")
    
    for folder in ["1", "2", "3", "4"]:
        folder_path = os.path.join(examples_base_path, folder)
        config_path = os.path.join(folder_path, "config.json")
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    input_filename = config.get("input_file", "input.jpg")
                    output_filename = config.get("output_file", "output.jpg")
                    outer_scale = config.get("outer_scale", 4)
                    
                    input_image_path = os.path.join(folder_path, input_filename)
                    output_image_path = os.path.join(folder_path, output_filename)
                    
                    if os.path.exists(input_image_path) and os.path.exists(output_image_path):
                        # Verify images can be opened
                        Image.open(input_image_path).close()
                        Image.open(output_image_path).close()
                        examples.append([input_image_path, output_image_path, outer_scale])
        except (json.JSONDecodeError, OSError) as e:
            print(f"Error loading example in folder {folder}: {e}")
            continue
    
    return examples

def select_example(evt: gr.SelectData, examples_data):
    if not examples_data:
        return None, 2, None, "No examples available"
    example_index = evt.index
    input_image_path, output_image_path, outer_scale = examples_data[example_index]
    return input_image_path, outer_scale, output_image_path, f"Loaded example with Outer Scale: {outer_scale}"