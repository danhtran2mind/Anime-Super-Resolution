import sys
import os
import subprocess
import argparse

# Append the Real-ESRGAN directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'third_party', 'Real-ESRGAN'))

def infer(args):
    # Construct the command using parsed arguments
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), 'src', 'third_party', 'Real-ESRGAN', 'inference_realesrgan.py'),
        '-n', args.model_name,
        '-i', args.input_path,
        '-o', args.output_dir,
        '--suffix', args.suffix,
        '--ext', args.ext,
        '-s', str(args.outscale),
        '--model_path', args.model_path
    ]

    # Add optional arguments if specified
    if args.denoise_strength is not None:
        cmd.extend(['-dn', str(args.denoise_strength)])
    if args.tile is not None:
        cmd.extend(['-t', str(args.tile)])
    if args.tile_pad is not None:
        cmd.extend(['--tile_pad', str(args.tile_pad)])
    if args.pre_pad is not None:
        cmd.extend(['--pre_pad', str(args.pre_pad)])
    if args.face_enhance:
        cmd.append('--face_enhance')
    if args.fp32:
        cmd.append('--fp32')
    if args.alpha_upsampler is not None:
        cmd.extend(['--alpha_upsampler', args.alpha_upsampler])
    if args.gpu_id is not None:
        cmd.extend(['-g', str(args.gpu_id)])

    # Run the command
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run Real-ESRGAN inference for anime super-resolution')
    parser.add_argument('--model_name', type=str, default='RealESRGAN_x4plus', 
                        help='Name of the model to use: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | realesr-animevideov3 | realesr-general-x4v3')
    parser.add_argument('--input_path', type=str, default='tests/test_data/input.jpg', help='Path to input image or folder')
    parser.add_argument('--output_dir', type=str, default='tests/test_data/output', help='Path to output directory')
    parser.add_argument('--outscale', type=float, default=4, help='Output scale factor')
    parser.add_argument('--ext', type=str, default='auto', help='Image extension. Options: auto | jpg | png')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix for output file name')
    parser.add_argument('--model_path', type=str, default='./ckpts/Real-ESRGAN-Anime-finetuning/net_g_latest.pth', 
                        help='Path to model checkpoint')
    parser.add_argument('--denoise_strength', type=float, default=None, 
                        help='Denoise strength (0 to 1). Only used for realesr-general-x4v3 model')
    parser.add_argument('--tile', type=int, default=None, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=None, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=None, help='Pre padding size at each border')
    parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
    parser.add_argument('--fp32', action='store_true', help='Use fp32 precision during inference')
    parser.add_argument('--alpha_upsampler', type=str, default=None, 
                        help='The upsampler for alpha channels. Options: realesrgan | bicubic')
    parser.add_argument('--gpu_id', type=int, default=None, help='GPU device to use (e.g., 0,1,2 for multi-GPU)')

    # Parse arguments
    args = parser.parse_args()

    infer(args)