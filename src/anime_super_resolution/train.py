import sys
import os
import subprocess
import argparse
import yaml
import shutil

# Get the absolute path to the Real-ESRGAN directory
real_esrgan_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'third_party', 'Real-ESRGAN'))

def train(args):
    # Ensure the config file exists
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file {args.config} not found")

    # Read the YAML config file to get the experiment name
    with open(args.config, 'r') as f:
        config_data = yaml.safe_load(f)
    experiment_name = config_data.get('name', 'default_experiment')

    # Set up environment for subprocess
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{real_esrgan_dir}{os.pathsep}{env.get('PYTHONPATH', '')}"

    # Execute the Real-ESRGAN training command
    try:
        command = [
            sys.executable,  # Use the current Python executable
            os.path.join(real_esrgan_dir, 'realesrgan', 'train.py'),
            '-opt', args.config
        ]
        if args.auto_resume:
            command.append('--auto_resume')
        if args.launcher != 'none':
            command.append(f'--launcher={args.launcher}')
        if args.debug:
            command.append('--debug')
        command.append(f'--local_rank={args.local_rank}')
        if args.force_yml:
            command.extend(['--force_yml'] + args.force_yml)
        
        subprocess.run(command, env=env, check=True)  # Pass the modified environment

        # Move the entire experiment directory to output_model_dir
        if args.output_model_dir:
            source_dir = os.path.join(real_esrgan_dir, 'experiments', experiment_name)
            target_dir = os.path.abspath(args.output_model_dir)
            
            # Create target directory if it doesn't exist
            os.makedirs(target_dir, exist_ok=True)
            
            # Move the entire source directory to target
            if os.path.exists(source_dir):
                shutil.move(source_dir, target_dir)
                print(f"Moved experiment directory from {source_dir} to {args.output_model_dir}")
            else:
                print(f"Warning: Source directory {source_dir} does not exist")

    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error moving directory: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run Real-ESRGAN training with specified config')
    parser.add_argument('--config', type=str, default='configs/Real-ESRGAN-Anime-finetuning.yml', 
                        help='Path to the configuration YAML file')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none', 
                        help='job launcher')
    parser.add_argument('--auto_resume', action='store_true', 
                        help='Automatically resume training from the latest checkpoint')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--force_yml', nargs='+', default=None, 
                        help='Force to update yml files. Examples: train:ema_decay=0.999')
    parser.add_argument('--output_model_dir', type=str, default='ckpts', 
                        help='Path to move experiment directory after training')
    args = parser.parse_args()

    train(args)