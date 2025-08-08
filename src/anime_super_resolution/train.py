import sys
import os
import subprocess
import argparse

# Append the Real-ESRGAN directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'third_party', 'Real-ESRGAN'))

def train(args):
    # Execute the Real-ESRGAN training command
    subprocess.run([
        'python',
        os.path.join(os.path.dirname(__file__), '..', 'third_party', 'Real-ESRGAN', 'realesrgan', 'train.py'),
        '-opt', args.config,
        '--auto_resume'
    ])

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run Real-ESRGAN training with specified config')
    parser.add_argument('--config', type=str, default='configs/finetune_anime.yml', 
                       help='Path to the configuration YAML file')
    args = parser.parse_args()

    train(args)

    