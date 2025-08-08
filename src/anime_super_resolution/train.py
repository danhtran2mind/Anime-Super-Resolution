import sys
import os
import subprocess
import argparse

# Get the absolute path to the Real-ESRGAN directory
real_esrgan_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'third_party', 'Real-ESRGAN'))

# Append the Real-ESRGAN directory to sys.path if not already present
if real_esrgan_dir not in sys.path:
    sys.path.insert(0, real_esrgan_dir)

def train(args):
    # Ensure the config file exists
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file {args.config} not found")

    # Execute the Real-ESRGAN training command
    try:
        subprocess.run([
            sys.executable,  # Use the current Python executable
            os.path.join(real_esrgan_dir, 'realesrgan', 'train.py'),
            '-opt', args.config,
            '--auto_resume'
        ], check=True)  # Raise an exception if the command fails
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run Real-ESRGAN training with specified config')
    parser.add_argument('--config', type=str, default='configs/finetune_anime.yml', 
                        help='Path to the configuration YAML file')
    args = parser.parse_args()

    train(args)