"""
Download the trained model from the DVC remote storage.
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


def main(model_name: str = 'trained_model'):
    """
    Main function to download the specific trained model from dvc.lock.
    """
    bucket_name = 'dvc-remla24-02'

    # Read the dvc.lock file
    with open('dvc.lock', 'r', encoding='utf-8') as file:
        dvc_lock_data = yaml.safe_load(file)

    # Find the path and md5 hash for models/trained_model.joblib
    for stage in dvc_lock_data['stages'].values():
        for out in stage.get('outs', []):
            if out['path'] == 'models/trained_model.joblib':
                md5_hash = out['md5']
                key = f'data/files/md5/{md5_hash[:2]}/{md5_hash[2:]}'
                output_file = os.path.join(f'{model_name}')

                # Ensure directory exists
                os.makedirs(os.path.dirname(output_file), exist_ok=True)

                download_data(bucket_name, key, output_file)
                return

    print("The specified model path 'models/trained_model.joblib' was not found in dvc.lock")


if __name__ == '__main__':
    main()
