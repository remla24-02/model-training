import os
from dvc.repo import Repo
import datetime
from datetime import datetime, timedelta

import pytest

MAX_MODEL_AGE = 60  # Maximum age of model in days before considered stale


def get_model_age(model_path):
    repo = Repo('.')

    # Get the path to the actual file in the cache
    outs = repo.find_outs_by_path(model_path)
    if not outs:
        raise RuntimeError(f"No outputs found for {model_path}")

    # Get the first output (assuming there's only one)
    out = outs[0]
    cache_path = out.fspath

    # Get the modification time of the file
    mod_time = datetime.fromtimestamp(os.path.getmtime(cache_path))

    return mod_time


def test_check_for_staleness():
    """
    Test if the model is considered stale.
    """
    model_path = 'models/trained_model.joblib'

    mod_time = get_model_age(model_path)
    current_time = datetime.now()
    age = current_time - mod_time
    assert age < timedelta(days=60)


if __name__ == "__main__":
    pytest.main()
