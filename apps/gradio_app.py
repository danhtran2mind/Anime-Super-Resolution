import gradio as gr
import os
import sys
import json
from PIL import Image

from gradio_app.project_info import (
    CONTENT_DESCRIPTION,
    CONTENT_IN_1, CONTENT_IN_2,
    CONTENT_OUT_1, CONTENT_OUT_2
)
# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

def load_examples():
    examples = []
    examples_base_path = os.path.join("apps", "assets", "examples", "Real-ESRGAN-Anime-finetuning")
    
    for folder in ["1", "2", "3", "4"]:
        folder_path = os.path.join(examples_base_path, folder)
        config_path = os.path.join(folder_path, "config.json")
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                input_filename = config.get("input_file", "input.jpg")
                output_filename = config.get("output_file", "output.jpg")
                outer_scale = config.get("outer_scale", 4)
                
                input_image_path = os.path.join(folder_path, input_filename)
                output_image_path = os.path.join(folder_path, output_filename)
                
                if os.path.exists(input_image_path) and os.path.exists(output_image_path):
                    input_image_data = Image.open(input_image_path)
                    output_image_data = Image.open(output_image_path)
                    examples.append([input_image_data, output_image_data, outer_scale])
    
    return examples

def select_example(evt: gr.SelectData, examples_data):
    example_index = evt.index
    input_image_data, output_image_data, outer_scale = examples_data[example_index]
    return input_image_data, outer_scale, output_image_data, f"Loaded example with Outer Scale: {outer_scale}"

# Load custom CSS
custom_css = open("apps/gradio_app/static/styles.css").read()

# JavaScript function to update warning_text Markdown component
outer_scale_warning = open("apps/gradio_app/static/outer_scale_warning.js").read()

# Define Gradio interface
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# Anime Super Resolution 🖼️")
    gr.Markdown(CONTENT_DESCRIPTION)
    gr.Markdown(CONTENT_IN_1)
    gr.HTML(CONTENT_IN_2)
    with gr.Row():
        with gr.Column(scale=2):
            input_image = gr.Image(
                type="filepath",
                label="Input Image",
                elem_classes="input-image"
            )
            model_id = gr.Textbox(
                label="Model ID",
                value="danhtran2mind/Real-ESRGAN-Anime-finetuning"
            )
            
            outer_scale = gr.Slider(
                minimum=1,
                maximum=16,
                step=1,
                value=4,
                label="Outer Scale",
                elem_id="outer-scale-slider"
            )
            warning_text = gr.HTML(elem_id="warning-text")  # Added elem_id for JS targeting
            gr.Markdown(
                "**Note:** For optimal output quality, set `Outer Scale` to a value between 1 and 4. "
                "Values greater than 4 are not recommended. "
                "Please ensure `Outer Scale` is greater than or equal to `Inner Scale` (default: 4)."
            )
            
            examples_data = load_examples()
            submit_button = gr.Button("Run Inference")
        
        with gr.Column(scale=3):
            output_image = gr.Image(
                label="Output Image",
                elem_classes="output-image"
            )
            output_text = gr.Textbox(label="Status")
    
    # Client-side warning update for warning_text
    outer_scale.change(
        fn=lambda x: x,  # Pass-through function to satisfy Gradio
        inputs=outer_scale,
        outputs=outer_scale,  # Return value to maintain slider functionality
        js=outer_scale_warning  # Update warning_text via JavaScript
    )
    
    gr.Examples(
        examples=[[input_img, output_img, outer_scale] for input_img, output_img, outer_scale in examples_data],
        inputs=[input_image, output_image, outer_scale],
        label="Example Inputs",
        examples_per_page=4
    )
        
    submit_button.click(
        fn=run_inference,
        inputs=[input_image, model_id, outer_scale],
        outputs=[output_image, output_text]
    )
    gr.HTML(CONTENT_OUT_1)
    gr.Markdown(CONTENT_OUT_2)
    
if __name__ == "__main__":
    demo.launch(share=True)  # Changed to local launch for safety; use share=True for public URL if needed