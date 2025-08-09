import gradio as gr
import os
import sys

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
        return '<span style="color:red">Warning: outer_scale > 4 may reduce output quality. Recommended range is (1, 4].</span>'
    return ""

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
            
            outer_scale = gr.Slider(
                minimum=1,
                maximum=16,
                step=1,
                value=4,
                label="Outer Scale"
            )
            warning_text = gr.Markdown()
            gr.Markdown(
                "**Note**: For best output quality, set `outer_scale` in the range (1, 4]. "
                "Values greater than 4 may still work but are not recommended, as `outer_scale` "
                "should be greater than or equal to `inner_scale` (default is 4)."
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