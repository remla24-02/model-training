"""
Download the raw data from the DVC remote storage.
"""

import os
import yaml
import boto3
from botocore import UNSIGNED
from botocore.config import Config


def download_data(bucket_name, file_name, output_file):
    """
    Pull specific files from the DVC remote storage.
    """
    s3 = boto3.client('s3', region_name='eu-north-1',
                      config=Config(signature_version=UNSIGNED))
    s3.download_file(bucket_name, file_name, output_file)


def main():
    """
    Main function to ensure the raw data directory exists and pull the latest data.
    """
    bucket_name = 'dvc-remla24-02'
    raw_data_dir = os.path.join('data', 'raw')

    # Ensure the raw data directory exists
    os.makedirs(raw_data_dir, exist_ok=True)

    # Read the dvc.lock file
    with open('dvc.lock', 'r', encoding='utf-8') as file:
        dvc_lock_data = yaml.safe_load(file)

    # List of specific files to pull
    files_to_pull = [
        'data/raw/train.txt',
        'data/raw/test.txt',
        'data/raw/val.txt'
    ]

    # Iterate through each stage and download the outs for the specified files
    for stage in dvc_lock_data['stages'].values():
        for out in stage.get('outs', []):
            if out['path'] in files_to_pull:
                md5_hash = out['md5']
                key = f'data/files/md5/{md5_hash[:2]}/{md5_hash[2:]}'
                output_file = out['path']

                # Ensure directory exists
                os.makedirs(os.path.dirname(output_file), exist_ok=True)

                download_data(bucket_name, key, output_file)

    # Only keep the first 20000 lines of each file
    for file in files_to_pull:
        with open(file, 'r', encoding='utf-8') as f:
            lines = [f.readline() for _ in range(20000)]

        with open(file, 'w', encoding='utf-8') as f:
            f.writelines(lines)


if __name__ == '__main__':
    main()
