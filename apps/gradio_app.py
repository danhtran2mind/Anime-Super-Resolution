import gradio as gr
import os
import sys
import json

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.anime_super_resolution.infer import infer

def run_inference(input_image, model_id, outer_scale):
    if not input_image:
        return None, "Please upload an image."
    
    models_config_path = "configs/model_ckpts.yaml"
    try:
        # Run inference without specifying output_path
        output_image = infer(
            input_path=input_image,
            model_id=model_id,
            models_config=models_config_path,
            outer_scale=outer_scale,
        )
        
        return output_image, "Inference completed successfully!"
    except Exception as e:
        return None, f"Error during inference: {str(e)}"

def update_warning(outer_scale):
    if outer_scale > 4:
        return '<span style="color:red">To ensure optimal output quality, please set the <code>Outer Scale</code> to a value of 4 or less. The suggested range is from 1 to 4.</span>'
    return ""

def load_examples():
    """
    Load example inputs from apps/assets/examples/Real-ESRGAN-Anime-finetuning/{1,2,3,4}.
    Returns a list of lists: [input_image_path, outer_scale] for each example.
    """
    examples = []
    examples_base_path = os.path.join("apps", "assets", "examples", "Real-ESRGAN-Anime-finetuning")
    
    for folder in ["1", "2", "3", "4"]:
        folder_path = os.path.join(examples_base_path, folder)
        config_path = os.path.join(folder_path, "config.json")
        input_image_path = os.path.join(folder_path, "input.jpg")
        
        if os.path.exists(input_image_path) and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                outer_scale = config.get("outer_scale", 4)  # Default to 4 if not specified
            examples.append([input_image_path, outer_scale])
    
    return examples

custom_css = open("apps/gradio_app/static/styles.css").read()
# Define Gradio interface
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# Anime Image Super-Resolution with Real-ESRGAN")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="filepath", label="Input Image")
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
            warning_text = gr.Markdown()
            gr.Markdown(
                "**Note:** For optimal output quality, set `Outer Scale` to a value between 1 and 4. "
                "Values greater than 4 are not recommended. "
                "Please ensure `Outer Scale` is greater than or equal to `Inner Scale` (default: 4)."
            )
            
            # Load and display examples
            examples_data = load_examples()
            gr.Examples(
                examples=examples_data,
                inputs=[input_image, outer_scale],
                label="Example Inputs",
                examples_per_page=4
            )
            
            submit_button = gr.Button("Run Inference")
        
        with gr.Column():
            output_image = gr.Image(label="Output Image")
            output_text = gr.Textbox(label="Status")
    
    # Update warning text when outer_scale changes
    outer_scale.change(
        fn=update_warning,
        inputs=outer_scale,
        outputs=warning_text
    )
    
    submit_button.click(
        fn=run_inference,
        inputs=[input_image, model_id, outer_scale],
        outputs=[output_image, output_text]
    )

if __name__ == "__main__":
    demo.launch(share=True)