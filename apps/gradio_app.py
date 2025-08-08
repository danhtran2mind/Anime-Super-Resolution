import gradio as gr
import os
import subprocess
from pathlib import Path
import tempfile

def run_super_resolution(image, upscale_rate, model_path):
    # Create temporary directories for input and output
    temp_dir = Path(tempfile.mkdtemp())
    input_path = temp_dir / "input_image.png"
    output_dir = temp_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Save uploaded image
    image.save(str(input_path))
    
    # Construct the command
    command = [
        "python", "src/anime_super_resolution/infer.py",
        "--input_path", str(input_path),
        "--output_dir", str(output_dir),
        "--suffix", "real_esrgan_anime",
        "--outscale", str(upscale_rate),
        "--model_path", model_path
    ]
    
    # Run the inference command
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(result.stdout)
        
        # Find the output image
        output_files = list(output_dir.glob("*real_esrgan_anime*.png"))
        if output_files:
            return str(output_files[0])
        else:
            return "Error: No output image found"
            
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"

# Define model path options
model_paths = [
    "ckpts/Real-ESRGAN-Anime-finetuning/net_g_latest.pth",
    # Add more model paths here if available
]

# Create Gradio interface
iface = gr.Interface(
    fn=run_super_resolution,
    inputs=[
        gr.Image(type="pil", label="Input Image"),
        gr.Dropdown(choices=[2, 4, 8], label="Upscale Rate", value=2),
        gr.Dropdown(choices=model_paths, label="Model Path", value=model_paths[0])
    ],
    outputs=gr.Image(type="filepath", label="Upscaled Image"),
    title="Anime Super Resolution",
    description="Upload an image and select parameters to upscale using Real-ESRGAN"
)

if __name__ == "__main__":
    iface.launch()