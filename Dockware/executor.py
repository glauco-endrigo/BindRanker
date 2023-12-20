import subprocess
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Get the current script's directory
current_directory = os.path.dirname(os.path.realpath(__file__))

# List of scripts to run
scripts_to_run = ["remove_non_standard_residues.py",
    "make_pdbqt.py",
    "config_generator.py",
    "VinaParallel.py",
    "sep_poses.py"
]

# Run each script
for script in scripts_to_run:
    script_path = os.path.join(current_directory, script)
    logging.info(f"Running {script_path}")
    try:
        subprocess.run([sys.executable, script_path], check=True)
    except Exception as e:
        logging.error(f"Error running {script_path}: {e}")
        break
    logging.info(f"{script_path} completed")
