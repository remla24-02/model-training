from lib_ml_remla24_team02 import data_preprocessing
import psutil
import pytest
import os

from src.models.define_model import main as define_model
from src.models.train_model import main as train_model
from tests.data.test_distribution import setup_data

TEST_EPOCHS = 1
MAX_ITERATION_MEMORY = 250


@pytest.fixture(scope="module")
def setup_training(setup_data):
    # Preprocess the data and define a model
    preprocessed_data_path = os.path.join('data', 'preprocessed')
    data_preprocessing.preprocess(os.path.join('data', 'raw'), preprocessed_data_path)
    define_model()

    yield

    # Remove the preprocessed data files and the defined/trained model
    for file in os.listdir(preprocessed_data_path):
        if file != '.gitkeep':
            os.remove(os.path.join(preprocessed_data_path, file))
    os.remove(os.path.join('models', 'defined_model.joblib'))


@pytest.fixture(scope="function")
def memory_usage():
    # Capture initial memory usage in MB
    process = psutil.Process()
    initial_memory = process.memory_info().rss / (1024**2)
    yield initial_memory


def test_train_memory_usage(setup_data, setup_training, memory_usage):
    # Get the initial memory usage (MB) and train the model
    initial_memory = memory_usage
    train_model(test_epochs=TEST_EPOCHS)

    # Capture memory usage after training in MB
    process = psutil.Process()
    final_memory = process.memory_info().rss / (1024**2)

    # Check if the used memory (increase) is below the threshold
    memory_increase = final_memory - initial_memory
    assert memory_increase < MAX_ITERATION_MEMORY


if __name__ == "__main__":
    pytest.main()
    pass
