import sys
import os
import subprocess
import argparse

# Get the absolute path to the Real-ESRGAN directory
real_esrgan_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'third_party', 'Real-ESRGAN'))

def train(args):
    # Ensure the config file exists
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file {args.config} not found")

    # Set up environment for subprocess
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{real_esrgan_dir}{os.pathsep}{env.get('PYTHONPATH', '')}"

    # Execute the Real-ESRGAN training command
    try:
        subprocess.run([
            sys.executable,  # Use the current Python executable
            os.path.join(real_esrgan_dir, 'realesrgan', 'train.py'),
            '-opt', args.config,
            '--auto_resume'
        ], env=env, check=True)  # Pass the modified environment
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