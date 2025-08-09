import gradio as gr
import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.anime_super_resolution.infer import infer

def run_inference(input_image, model_id, models_config_path, outer_scale, inner_scale):
    if not input_image:
        return None, "Please upload an image."
    
    if not os.path.exists(models_config_path):
        return None, f"Models configuration file not found at: {models_config_path}"
    
    with open(models_config_path, 'r') as file:
        models_config = file.read()
    
    try:
        # Run inference without specifying output_path
        output_image = infer(
            input_path=input_image,
            model_id=model_id,
            models_config=models_config,
            outer_scale=outer_scale,
            inner_scale=inner_scale
        )
        
        return output_image, "Inference completed successfully!"
    except Exception as e:
        return None, f"Error during inference: {str(e)}"

# Define Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Anime Image Super-Resolution with Real-ESRGAN")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="filepath", label="Input Image")
            model_id = gr.Textbox(
                label="Model ID",
                value="danhtran2mind/Real-ESRGAN-Anime-finetuning"
            )
            models_config_path = gr.Textbox(
                label="Models Config Path",
                value="path/to/your/models_config.yaml"
            )
            outer_scale = gr.Slider(
                minimum=1,
                maximum=16,
                step=1,
                value=4,
                label="Outer Scale"
            )
            inner_scale = gr.Slider(
                minimum=1,
                maximum=4,
                step=1,
                value=4,
                label="Inner Scale"
            )
            submit_button = gr.Button("Run Inference")
        
        with gr.Column():
            output_image = gr.Image(label="Output Image")
            output_text = gr.Textbox(label="Status")
    
    submit_button.click(
        fn=run_inference,
        inputs=[input_image, model_id, models_config_path, outer_scale, inner_scale],
        outputs=[output_image, output_text]
    )

if __name__ == "__main__":
    demo.launch(share=True)