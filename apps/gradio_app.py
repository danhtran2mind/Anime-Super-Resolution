import os
import subprocess
import gradio as gr
import random
from gradio_app.inference import run_inference
from gradio_app.examples import load_examples, select_example
from gradio_app.project_info import (
    NAME, 
    CONTENT_DESCRIPTION, 
    CONTENT_IN_1, 
    CONTENT_OUT_1
)

def run_setup_script():
    setup_script = os.path.join(os.path.dirname(__file__), "gradio_app", "setup_scripts.py")
    try:
        result = subprocess.run(["python", setup_script], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Setup script failed with error: {e.stderr}")
        return f"Setup script failed: {e.stderr}"

def stop_app():
    """Function to stop the Gradio app."""
    try:
        gr.Interface.close_all()  # Attempt to close all running Gradio interfaces
        return "Application stopped successfully."
    except Exception as e:
        return f"Error stopping application: {str(e)}"

def create_gui():
    try:
        custom_css = open("apps/gradio_app/static/style.css").read()
    except FileNotFoundError:
        print("Error: style.css not found at gradio_app/static/style.css")
        custom_css = ""  # Fallback to empty CSS if file is missing

    with gr.Blocks(css=custom_css) as demo:
        gr.Markdown(NAME)
        gr.HTML(CONTENT_DESCRIPTION)
        gr.HTML(CONTENT_IN_1)

        with gr.Row():
            with gr.Column(scale=2):
                input_image = gr.Image(type="filepath", label="Input Image")
                prompt = gr.Textbox(
                    label="Prompt",
                    value="a man is doing yoga"
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value="monochrome, lowres, bad anatomy, worst quality, low quality"
                )
                
                with gr.Row():
                    width = gr.Slider(
                        minimum=256,
                        maximum=1024,
                        value=512,
                        step=64,
                        label="Width"
                    )
                    height = gr.Slider(
                        minimum=256,
                        maximum=1024,
                        value=512,
                        step=64,
                        label="Height"
                    )
                
                with gr.Accordion("Advanced Settings", open=False):
                    num_steps = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=30,
                        step=1,
                        label="Number of Inference Steps"
                    )
                    use_random_seed = gr.Checkbox(label="Use Random Seed", value=False)
                    seed = gr.Slider(
                        minimum=0,
                        maximum=2**32 - 1,
                        value=42,
                        step=1,
                        label="Random Seed",
                        visible=True
                    )
                    
                    guidance_scale = gr.Slider(
                        minimum=1.0,
                        maximum=20.0,
                        value=7.5,
                        step=0.1,
                        label="Guidance Scale"
                    )
                    controlnet_conditioning_scale = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=1.0,
                        step=0.1,
                        label="ControlNet Conditioning Scale"
                    )  
                    
            with gr.Column(scale=3):
                output_images = gr.Image(label="Generated Images")
                output_message = gr.Textbox(label="Status")
                
                submit_button = gr.Button("Generate Images", elem_classes="submit-btn")
                stop_button = gr.Button("Stop Application", elem_classes="stop-btn")

        def update_seed_visibility(use_random):
            return gr.update(visible=not use_random)
        
        use_random_seed.change(
            fn=update_seed_visibility,
            inputs=use_random_seed,
            outputs=seed
        )
        
        # Load examples
        examples_data = load_examples(os.path.join("apps", "gradio_app", 
            "assets", "examples", "Stable-Diffusion-2.1-Openpose-ControlNet"))
        examples_component = gr.Examples(
            examples=examples_data,
            inputs=[
                input_image,
                prompt,
                negative_prompt,
                output_images,
                num_steps,
                seed,
                width,
                height,
                guidance_scale,
                controlnet_conditioning_scale,
                use_random_seed
            ],
            outputs=[
                input_image,
                prompt,
                negative_prompt,
                output_images,
                num_steps,
                seed,
                width,
                height,
                guidance_scale,
                controlnet_conditioning_scale,
                use_random_seed,
                output_message
            ],
            fn=select_example,
            cache_examples=False,
            label="Examples: Yoga Poses"
        )
        
        submit_button.click(
            fn=run_inference,
            inputs=[
                input_image,
                prompt,
                negative_prompt,
                num_steps,
                seed,
                width,
                height,
                guidance_scale,
                controlnet_conditioning_scale,
                use_random_seed,
            ],
            outputs=[output_images, output_message]
        )
        
        stop_button.click(
            fn=stop_app,
            inputs=[],
            outputs=[output_message]
        )
        
        gr.HTML(CONTENT_OUT_1)
        
    return demo

if __name__ == "__main__":
    run_setup_script()
    demo = create_gui()
    demo.launch()
