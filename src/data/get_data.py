"""
Download and extract training, testing and validation data.
"""

import os
import subprocess
import sys


def pull_specific_files(files):
    """
    Pull specific files from the DVC remote storage.
    """
    dvc_repo_url = os.path.abspath('.')
    try:
        for file in files:
            subprocess.run(["dvc", "get", dvc_repo_url, file,
                            "-o", file, "--force"], check=True)
        print("Successfully pulled the specified files.")
    except subprocess.CalledProcessError as e:
        print(f"Error pulling the specified files: {e}")
        sys.exit(1)


def main():
    """
    Main function to ensure the raw data directory exists and pull the latest data.
    """
    raw_data_dir = os.path.join('data', 'raw')

    # Ensure the raw data directory exists
    os.makedirs(raw_data_dir, exist_ok=True)

    # List of specific files to pull
    files_to_pull = [
        os.path.join('data', 'raw', 'train.txt'),
        os.path.join('data', 'raw', 'test.txt'),
        os.path.join('data', 'raw', 'val.txt')
    ]

    # Pull the specified files
    pull_specific_files(files_to_pull)

    # only keep the first 20000 lines of each file
    for file in files_to_pull:
        with open(file, 'r', encoding='utf-8') as f:
            lines = [f.readline() for _ in range(20000)]

        with open(file, 'w', encoding='utf-8') as f:
            f.writelines(lines)


if __name__ == '__main__':
    main()
