"""
Download and extract training, testing and validation data.
"""

import boto3
from botocore import UNSIGNED
from botocore.client import Config
import os


def download_data(bucket_name, file_name, output_file):
    """
    Download data from S3 bucket.
    """
    s3 = boto3.client('s3', region_name='eu-north-1',
                      config=Config(signature_version=UNSIGNED))
    s3.download_file(bucket_name, file_name, output_file)


def main():
    """
    Main function.
    """
    bucket_name = 'dvc-remla24-02'

    dvc_files = [file for file in os.listdir(
        os.path.join('data', 'raw')) if file.endswith('.dvc')]

    files = {}

    for dvc_file in dvc_files:
        with open(os.path.join('data', 'raw', dvc_file), 'r') as file:
            file_name = os.path.splitext(os.path.basename(file.name))[0]
            md5_hash = file.read()
            md5_hash = md5_hash.split(' ')[2]
            key = 'data/files/md5/' + md5_hash[:2] + '/' + md5_hash[2:]
            files[file_name] = key.rstrip('\n')

    for name, key in files.items():
        download_data(bucket_name, key, os.path.join(
            'data', 'raw', f"{name}.txt"))


if __name__ == '__main__':
    main()
