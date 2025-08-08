# this file path is scripts\setup_third_party.py
# sys.append to src\anime_super_resolution\third_party

# !git clone https://github.com/danhtran2mind/Real-ESRGAN.git
# # copy Real-ESRGAN/realesrgan to 

import sys
import os
import subprocess
import shutil
import argparse

def main():
    parser = argparse.ArgumentParser(description="Setup third-party dependencies for anime super resolution")
    parser.add_argument('--src_path', default=os.path.join('src', 'anime_super_resolution', 'third_party'),
                      help='Path to append to sys.path and copy Real-ESRGAN to')
    parser.add_argument('--repo_url', default='https://github.com/danhtran2mind/Real-ESRGAN.git',
                      help='URL of the Real-ESRGAN repository to clone')
    parser.add_argument('--clone_dir', default='Real-ESRGAN', help='Directory to clone Real-ESRGAN into')
    
    args = parser.parse_args()

    # Append src/anime_super_resolution/third_party to sys.path
    sys.path.append(args.src_path)

    # Create third_party directory if it doesn't exist
    os.makedirs(args.src_path, exist_ok=True)

    # Clone Real-ESRGAN repository
    subprocess.run(['git', 'clone', args.repo_url, args.clone_dir], check=True)

    # Copy realesrgan directory to third_party
    src_realesrgan = os.path.join(args.clone_dir, 'realesrgan')
    dest_realesrgan = os.path.join(args.src_path, 'realesrgan')
    
    if os.path.exists(src_realesrgan):
        shutil.copytree(src_realesrgan, dest_realesrgan, dirs_exist_ok=True)
    else:
        raise FileNotFoundError(f"Directory {src_realesrgan} not found after cloning")

if __name__ == '__main__':
    main()