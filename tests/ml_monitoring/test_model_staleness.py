import pytest
import os
import boto3
from botocore import UNSIGNED
from botocore.client import Config
from datetime import datetime

from src.models.get_model import main as get_model


MAX_MODEL_AGE = 60 # Maximum age of model in days before considered stale


@pytest.fixture(scope="session")
def setup_model():
    # Download the model
    get_model()

    # Yield to run the tests
    yield

    # Teardown the model
    os.remove(os.path.join('models', 'trained_model.joblib'))


def get_model_age(bucket_name, key):
    """
    Get the age of the model in days.
    """
    s3 = boto3.client('s3', region_name='eu-north-1',
                      config=Config(signature_version=UNSIGNED))
    response = s3.head_object(Bucket=bucket_name, Key=key)
    last_modified = response['LastModified']
    age_in_days = (datetime.now(last_modified.tzinfo) - last_modified).days
    return age_in_days


def test_check_for_staleness(setup_model):
    """
    Test if the model is considered stale.
    """
    bucket_name = 'dvc-remla24-02'

    with open(os.path.join('models', 'trained_model.joblib.dvc'), 'r', encoding='utf-8') as file:
        md5_hash = file.read()
        md5_hash = md5_hash.split(' ')[2]
        key = 'data/files/md5/' + md5_hash[:2] + '/' + md5_hash[2:]

    model_age = get_model_age(bucket_name, key.rstrip('\n'))
    assert model_age <= MAX_MODEL_AGE


if __name__ == "__main__":
    pytest.main()
    pass