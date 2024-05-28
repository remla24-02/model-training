import os
import pytest

from joblib import load
from lib_ml_remla24_team02.data_preprocessing import load_data, tokenize_data, encode_data

from src.data.get_data import main as get_data
from src.models.get_model import main as get_model
from src.models.evaluate_model import evaluate_model


ACCURACY_THRESHOLD = 0.07
SHORT_LENGTH_THRESHOLD = 40
LONG_LENGTH_THRESHOLD = 80


@pytest.fixture()
def trained_model():
    # Create a model
    model_path = os.path.join('models', 'trained_model.joblib')
    get_model()

    # Yield to run the tests
    yield load(model_path)

    # Teardown the model
    os.remove(model_path)


@pytest.fixture()
def test_data():
    # Download the data
    get_data()

    raw_x, raw_y = load_data(os.path.join('data', 'raw'))
    yield raw_x, raw_y

    # Teardown the data
    for data_type in ["train", "val", "test"]:
        file_path = os.path.join("data", "raw", f"{data_type}.txt")
        if os.path.exists(file_path):
            os.remove(os.path.join("data", "raw", f"{data_type}.txt"))


def preprocess_data_and_evaluate_model(raw_x, raw_y, model):
    _, _, x_test, _ = tokenize_data(raw_x)
    _, _, y_test, _ = encode_data(raw_y)

    accuracy, _, _, _ = evaluate_model(model, x_test, y_test)
    return accuracy


def test_data_slice(trained_model, test_data):
    """
    Divide test data into slices by using the length of the urls
    and check that model can handle urls of different lengths.
    Slices are of three types: short_urls, medium_urls and long_urls.
    """

    raw_x, raw_y = test_data
    original_accuracy = preprocess_data_and_evaluate_model(raw_x, raw_y, trained_model)

    raw_x_train, raw_x_test, raw_x_val = raw_x
    raw_y_train, raw_y_test, raw_y_val = raw_y

    indices_short_urls = [index for index, url in enumerate(raw_x_test)
                          if len(url) < SHORT_LENGTH_THRESHOLD]
    indices_medium_urls = [index for index, url in enumerate(raw_x_test)
                           if SHORT_LENGTH_THRESHOLD <= len(url) < LONG_LENGTH_THRESHOLD]
    indices_long_urls = [index for index, url in enumerate(raw_x_test)
                         if len(url) >= LONG_LENGTH_THRESHOLD]

    sliced_data = {
        'short_urls': indices_short_urls,
        'medium_urls': indices_medium_urls,
        'long_urls': indices_long_urls
    }

    for slice_name, indices in sliced_data.items():
        raw_x = [raw_x_train, [raw_x_test[index] for index in indices], raw_x_val]
        raw_y = [raw_y_train, [raw_y_test[index] for index in indices], raw_y_val]

        slice_accuracy = preprocess_data_and_evaluate_model(raw_x, raw_y, trained_model)

        assert abs(original_accuracy - slice_accuracy) <= ACCURACY_THRESHOLD, \
            (f"The accuracy deviation for slice '{slice_name}' "
             f"exceeds the threshold of {ACCURACY_THRESHOLD}")


if __name__ == "__main__":
    pytest.main()
