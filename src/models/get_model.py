"""
Download and extract training, testing and validation data.
"""

import boto3
import os


def download_data(bucket_name, file_name, output_file):
    """
    Download data from S3 bucket.
    """
    s3 = boto3.client('s3', region_name='eu-north-1')
    s3.download_file(bucket_name, file_name, output_file)


def main():
    """
    Main function.
    """
    bucket_name = 'dvc-remla24-02'

    # read the .dvc file in models folder
    with open('models/trained.joblib_model.dvc', 'r') as file:
        md5_hash = file.read()
        md5_hash = md5_hash.split(' ')[2]
        key = 'models/files/md5/' + md5_hash[:2] + '/' + md5_hash[2:]

    download_data(bucket_name, key.rstrip('\n'),
                  'models/trained_model.joblib')


if __name__ == '__main__':
    main()
