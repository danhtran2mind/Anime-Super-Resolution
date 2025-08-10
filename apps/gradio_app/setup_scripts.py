import subprocess
import sys
import os

def run_script(script_path, args=None):
    """
    Run a Python script using subprocess with optional arguments and handle errors.
    Returns True if successful, False otherwise.
    """
    if not os.path.isfile(script_path):
        print(f"Script not found: {script_path}")
        return False

    try:
        command = [sys.executable, script_path]
        if args:
            command.extend(args)
        result = subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True
        )
        print(f"Successfully executed {script_path}")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing {script_path}:")
        print(e.stderr)
        return False
    except Exception as e:
        print(f"Unexpected error executing {script_path}: {str(e)}")
        return False

def main():
    """
    Main function to execute download_ckpts.py with proper error handling.
    """
    scripts_dir = "scripts"
    scripts = [
        {
            "path": os.path.join(scripts_dir, "download_ckpts.py"),
            "args": []  # Empty list for args to avoid NoneType issues
        },
        {
            "path": os.path.join(scripts_dir, "setup_third_party.py"),
            "args": []
        }
    ]

    for script in scripts:
        script_path = script["path"]
        args = script.get("args", [])  # Safely get args with default empty list
        print(f"Starting execution of {script_path}{' with args: ' + ' '.join(args) if args else ''}\n")
        
        if not run_script(script_path, args):
            print(f"Stopping execution due to error in {script_path}")
            sys.exit(1)
        
        print(f"Completed execution of {script_path}\n")

if __name__ == "__main__":
    main()