# this file is src\anime_super_resolution\infer.py

# sys append to src\anime_super_resolution\third_party\Real-ESRGAN

# cmd = f"python inference_realesrgan.py -n {model_name} -i {input_path} -o {output_dir} --ext {ext} --outscale {outscale} --model_path {model_path}"

# run cmd
# input_img_path = "tests/test_data/input.jpg"
# my_fintune_demo_path = "tests/test_data/output.jpg"
# model_name="RealESRGAN_x4plus",
# input_path=input_img_path,
# output_dir=my_fintune_demo_path,
# outscale=2,
# ext="auto",
# model_path="./ckpts/Real-ESRGAN-Anime-finetuning/net_g_latest.pth"

import sys
import os
import subprocess
import argparse

# Append the Real-ESRGAN directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'third_party', 'Real-ESRGAN'))

def infer(args):
    # Construct the command using parsed arguments
    cmd = (
        f"python inference_realesrgan.py "
        f"-n {args.model_name} "
        f"-i {args.input_path} "
        f"-o {args.output_dir} "
        f"--suffix {args.suffix} "
        f"--ext {args.ext} "
        f"--outscale {args.outscale} "
        f"--model_path {args.model_path}"
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
