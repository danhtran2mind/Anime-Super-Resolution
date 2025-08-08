# this file is src\anime_super_resolution\infer.py

import sys
import os
import subprocess
import argparse

# Append the Real-ESRGAN directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'third_party', 'Real-ESRGAN'))

def infer(args):
    # Construct the path to inference_realesrgan.py
    third_party_project_dir = os.path.join(os.path.dirname(__file__), '..', 'third_party', 'Real-ESRGAN')
    inference_script = os.path.join(third_party_project_dir, 'inference_realesrgan.py')

    # Construct the command using parsed arguments
    cmd = (
        f"python \"{inference_script}\" "
        f"-n {args.model_name} "
        f"-i \"{args.input_path}\" "
        f"-o \"{args.output_dir}\" "
        f"--suffix {args.suffix} "
        f"--ext {args.ext} "
        f"--outscale {args.outscale} "
        f"--model_path \"{args.model_path}\""
    )

    # Run the command
    subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run Real-ESRGAN inference for anime super-resolution')
    parser.add_argument('--model_name', type=str, default='RealESRGAN_x4plus', help='Name of the model to use')
    parser.add_argument('--input_path', type=str, default='tests/test_data/input.jpg', help='Path to input image')
    parser.add_argument('--output_dir', type=str, default='tests/test_data/output.jpg', help='Path to output directory or file')
    parser.add_argument('--outscale', type=int, default=2, help='Output scale factor')
    parser.add_argument('--ext', type=str, default='auto', help='Output file extension')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix for output file name')
    parser.add_argument('--model_path', type=str, default='./ckpts/Real-ESRGAN-Anime-finetuning/net_g_latest.pth', help='Path to model checkpoint')

    # Parse arguments
    args = parser.parse_args()

    infer(args)