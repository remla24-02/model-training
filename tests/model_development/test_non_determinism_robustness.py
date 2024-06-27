import os
import pytest
from joblib import load

from src.models.define_model import main as define_model, _get_parameters
from src.models.evaluate_model import evaluate_model
from src.models.train_model import main as train_model


ACCURACY_THRESHOLD = 0.05
BATCH_SIZE = 50
EPOCH = 1


def train_and_evaluate_model(params, x_test, y_test):
    define_model(params, model_name='test_defined_model')
    train_model(model_name='test_trained_model',
                defined_model='test_defined_model')
    model = load(os.path.join('models', 'test_trained_model.joblib'))
    metrics, _, _, _ = evaluate_model(
        model, x_test, y_test, batch_size=BATCH_SIZE, save_data=False)
    return metrics["accuracy"]


@pytest.fixture(scope="function")
def setup_and_cleanup():
    # Setup code: Load test data
    x_test = load('data/preprocessed/preprocessed_x_test.joblib')
    y_test = load('data/preprocessed/preprocessed_y_test.joblib')

    # Yield the test data to the test function
    yield x_test, y_test

    # Teardown code: Remove test models
    model_files = ['test_defined_model.joblib', 'test_trained_model.joblib']
    for model_file in model_files:
        model_path = os.path.join('models', model_file)
        if os.path.exists(model_path):
            os.remove(model_path)


def test_non_determinism_robustness(setup_and_cleanup):
    """
    Given different seeds, tests non-determinism robustness of the model.
    """
    x_test, y_test = setup_and_cleanup

    params_1 = _get_parameters(
        random_state=42, epoch=EPOCH, batch_size=BATCH_SIZE)
    accuracy_1 = train_and_evaluate_model(params_1, x_test, y_test)

    params_2 = _get_parameters(
        random_state=1, epoch=EPOCH, batch_size=BATCH_SIZE)
    accuracy_2 = train_and_evaluate_model(params_2, x_test, y_test)

    assert abs(accuracy_1 - accuracy_2) <= ACCURACY_THRESHOLD, \
        (f"The accuracy deviation for the model robustness' "
         f"exceeds the threshold of {ACCURACY_THRESHOLD}")


if __name__ == "__main__":
    pytest.main()
