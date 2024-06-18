import os
import pytest
from joblib import load

from src.data.data_preprocessing import main as preprocess_data
from src.data.get_data import main as get_data
from src.models.define_model import main as define_model, _get_parameters
from src.models.evaluate_model import evaluate_model
from src.models.train_model import main as train_model


ACCURACY_THRESHOLD = 0.05
BATCH_SIZE = 50
EPOCH = 1


@pytest.fixture()
def test_data():
    # Download and preprocess the data
    get_data()
    preprocess_data()

    yield

    # Teardown the data
    # for data_type in ["train", "val", "test"]:
    #     file_path = os.path.join("data", "raw", f"{data_type}.txt")
    #     if os.path.exists(file_path):
    #         os.remove(file_path)

    # output_dir = os.path.join('data', 'preprocessed')
    # for data_type in ["train", "val", "test"]:
    #     for axis in ["x", "y"]:
    #         file_path = os.path.join(output_dir, f'preprocessed_{
    #                                  axis}_{data_type}.joblib')
    #         if os.path.exists(file_path):
    #             os.remove(file_path)

    # os.remove(os.path.join(output_dir, 'char_index.joblib'))
    # os.remove(os.path.join(output_dir, 'label_encoder.joblib'))


def train_and_evaluate_model(params, x_test, y_test):
    define_model(params)
    train_model()
    model = load(os.path.join('models', 'trained_model.joblib'))
    metrics, _, _, _ = evaluate_model(
        model, x_test, y_test, batch_size=BATCH_SIZE)
    return metrics["accuracy"]


def test_non_determinism_robustness(test_data):
    """
    Given different seeds, tests non-determinism robustness of the model.
    """
    x_test = load('data/preprocessed/preprocessed_x_test.joblib')
    y_test = load('data/preprocessed/preprocessed_y_test.joblib')

    params_1 = _get_parameters(
        random_state=42, epoch=EPOCH, batch_size=BATCH_SIZE)
    accuracy_1 = train_and_evaluate_model(params_1, x_test, y_test)

    params_2 = _get_parameters(
        random_state=1, epoch=EPOCH, batch_size=BATCH_SIZE)
    accuracy_2 = train_and_evaluate_model(params_2, x_test, y_test)

    assert abs(accuracy_1 - accuracy_2) <= ACCURACY_THRESHOLD, \
        (f"The accuracy deviation for the model robustness' "
         f"exceeds the threshold of {ACCURACY_THRESHOLD}")

    # os.remove(os.path.join('models', 'defined_model.joblib'))
    # os.remove(os.path.join('models', 'trained_model.joblib'))


if __name__ == "__main__":
    pytest.main()
