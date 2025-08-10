import gradio as gr
import os
import subprocess
from gradio_app.project_info import (
    CONTENT_DESCRIPTION,
    CONTENT_IN_1, CONTENT_IN_2,
    CONTENT_OUT_1, CONTENT_OUT_2
)
from gradio_app.inference import run_inference
from gradio_app.examples import load_examples, select_example


def run_setup_script():
    setup_script = os.path.join(os.path.dirname(__file__),
                                "gradio_app", "setup_scripts.py")
    try:
        result = subprocess.run(["python", setup_script], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Setup script failed with error: {e.stderr}")
        return f"Setup script failed: {e.stderr}"
    
def create_gui():
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
                    maximum=8,
                    step=1,
                    value=2,
                    label="Outer Scale",
                    elem_id="outer-scale-slider"
                )
                warning_text = gr.HTML(elem_id="warning-text")
                gr.Markdown(
                    "**Note:** For optimal output quality, set `Outer Scale` to a value between 1 and 4. "
                    "**Values greater than 4 are not recommended**. "
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
            fn=lambda x: x,
            inputs=outer_scale,
            outputs=outer_scale,
            js=outer_scale_warning
        )
        
        gr.Examples(
            examples=[[input_img, output_img, outer_scale] for input_img, output_img, outer_scale in examples_data],
            inputs=[input_image, output_image, outer_scale],
            label="Example Inputs",
            examples_per_page=4,
            cache_examples=False,
            fn=select_example,
            outputs=[input_image, outer_scale, output_image, output_text]
        )
            
        submit_button.click(
            fn=run_inference,
            inputs=[input_image, model_id, outer_scale],
            outputs=[output_image, output_text]
        )
        gr.HTML(CONTENT_OUT_1)
        gr.HTML(CONTENT_OUT_2)

        return demo
    
if __name__ == "__main__":
    run_setup_script()
    demo = create_gui()
    demo.launch(debug=True)