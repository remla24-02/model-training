"""
Download and extract training, testing and validation data.
"""

import os
import subprocess
import sys


def pull_specific_files(output_file):
    """
    Pull specific files from the DVC remote storage.
    """
    try:
        dvc_repo_url = os.path.abspath('.')
        subprocess.run(["dvc", "get", dvc_repo_url, "models/trained_model.joblib",
                       "-o", output_file, "--force"], check=True)
        print("Successfully pulled the latest model.")
    except subprocess.CalledProcessError as e:
        print(f"Error pulling the latest model: {e}")
        sys.exit(1)


def main(output_file: str = 'models/trained_model.joblib'):
    """
    Main function to ensure the raw data directory exists and pull the latest data.
    """
    model_dir = os.path.join('models')

    # Ensure the raw data directory exists
    os.makedirs(model_dir, exist_ok=True)

    # Pull the specified files
    pull_specific_files(output_file)


if __name__ == '__main__':
    main()
