import os
import glob
import shutil
import logging
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from default_config.masif_opts import masif_opts

# Set up logging
logging.basicConfig(level=logging.ERROR)

# Assume masif_opts is a dictionary with the necessary key

# Get the directory to check
dir_to_check = masif_opts["site"]["masif_precomputation_dir"]

# Iterate over the subdirectories
for subdir in os.listdir(dir_to_check):
    subdir_path = os.path.join(dir_to_check, subdir)

    # Check if it's a directory
    if os.path.isdir(subdir_path):
        # Check for the required files
        required_files = ["*_X.npy", "*_list_indices.npy", "*_input_feat.npy"]
        missing_files = [
            rf for rf in required_files if not glob.glob(os.path.join(subdir_path, rf))
        ]

        # If any files are missing, log an error and delete the subdirectory
        if missing_files:
            logging.error(
                f"Missing files {missing_files} in subdirectory {subdir_path}"
            )
            shutil.rmtree(subdir_path)
            continue
