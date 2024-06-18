import pytest
import os

from src.models.get_model import main as get_model

MAX_MODEL_SIZE_MB = 10


@pytest.fixture(scope="module")
def setup_model():
    # Create a model
    model_path = os.path.join('models', 'trained_model.joblib')
    get_model()

    # Yield to run the tests
    yield model_path

    # Teardown the model
    # os.remove(model_path)


def test_model_memory_usage(setup_model):
    # Get the model path and check if it is below max model size (MB)
    model_path = setup_model
    model_size = os.path.getsize(model_path) / (1024**2)
    assert model_size <= MAX_MODEL_SIZE_MB


if __name__ == "__main__":
    pytest.main()
    pass
