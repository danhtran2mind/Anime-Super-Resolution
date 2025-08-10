import gradio as gr
import os
import sys
import json
from PIL import Image

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

# JavaScript to handle outer_scale change
# JavaScript to handle outer_scale change
custom_js = """
<script>
document.addEventListener('DOMContentLoaded', function() {
    // Find the outer_scale number input by data-testid or aria-label
    const numberInput = document.querySelector('input[data-testid="number-input"]');
    const warningText = document.querySelector('#warning-text');

    if (numberInput && warningText) {
        numberInput.addEventListener('input', function() {
            const value = parseInt(numberInput.value);
            if (value > 4) {
                warningText.innerHTML = '<span style="color:red">To ensure optimal output quality, please set the <code>Outer Scale</code> to a value of 4 or less. The suggested range is from 1 to 4.</span>';
            } else {
                warningText.innerHTML = '';
            }
        });
    }
});
</script>
"""

# Define Gradio interface
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# Anime Image Super-Resolution with Real-ESRGAN")
    
    gr.Markdown("## Example Inputs")
    gr.Markdown("Select an example below to load its input image and outer scale. The corresponding output image will appear under 'Output Image'.")
    
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
                label="Outer Scale"
            )
            warning_text = gr.Markdown(elem_id="warning-text")  # Assign elem_id for JavaScript targeting
            gr.Markdown(
                "**Note:** For optimal output quality, set `Outer Scale` to a value between 1 and 4. "
                "Values greater than 4 are not recommended. "
                "Please ensure `Outer Scale` is greater than or equal to `Inner Scale` (default: 4)."
            )
            
            # Load examples
            examples_data = load_examples()
            
            submit_button = gr.Button("Run Inference")
        
        with gr.Column(scale=3):
            output_image = gr.Image(
                label="Output Image",
                elem_classes="output-image"
            )
            output_text = gr.Textbox(label="Status")
    
    # Inject JavaScript
    gr.HTML(custom_js)
    
    # Update input image, outer scale, and output image when an example is selected
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

if __name__ == "__main__":
    demo.launch(share=True)  # Changed to local launch for safety