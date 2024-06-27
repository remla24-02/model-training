import shutil
from joblib import load
import pytest
import os
from src.data.get_data import main as get_data
from lib_ml_remla24_team02 import data_preprocessing
from src.models.define_model import main as define_model
from src.models.train_model import main as train_model
from src.models.evaluate_model import evaluate_model


def keep_first_20000_lines(file_path):
    with open(file_path, 'r') as file:
        lines = [file.readline() for _ in range(20000)]

    os.makedirs('data/raw/temp', exist_ok=True)
    temp_file_path = os.path.join('data/raw/temp', os.path.basename(file_path))
    with open(temp_file_path, 'w') as file:
        file.writelines(lines)


@pytest.fixture(scope="module")
def setup_and_cleanup():
    get_data()
    assert os.path.exists('data/raw/train.txt')
    assert os.path.exists('data/raw/test.txt')
    assert os.path.exists('data/raw/val.txt')

    keep_first_20000_lines('data/raw/train.txt')
    keep_first_20000_lines('data/raw/test.txt')
    keep_first_20000_lines('data/raw/val.txt')

    os.makedirs('data/temp_preprocessed', exist_ok=True)

    yield

    # Teardown: run after the test
    if os.path.exists('models/test_defined_model.joblib'):
        os.remove('models/test_defined_model.joblib')
    if os.path.exists('models/test_trained_model.joblib'):
        os.remove('models/test_trained_model.joblib')
    if os.path.exists('data/temp_preprocessed'):
        shutil.rmtree('data/temp_preprocessed')
    if os.path.exists('data/raw/temp'):
        shutil.rmtree('data/raw/temp')


def test_model_pipeline(setup_and_cleanup):
    data_preprocessing.preprocess(os.path.join(
        'data', 'raw', 'temp'), os.path.join('data', 'temp_preprocessed'))

    assert os.path.exists('data/temp_preprocessed/preprocessed_x_train.joblib')
    assert os.path.exists('data/temp_preprocessed/preprocessed_y_train.joblib')
    assert os.path.exists('data/temp_preprocessed/preprocessed_x_test.joblib')
    assert os.path.exists('data/temp_preprocessed/preprocessed_y_test.joblib')
    assert os.path.exists('data/temp_preprocessed/preprocessed_x_val.joblib')
    assert os.path.exists('data/temp_preprocessed/preprocessed_y_val.joblib')

    define_model(model_name='test_defined_model',
                 preprocessed_folder='data/temp_preprocessed')

    assert os.path.exists('models/test_defined_model.joblib')

    train_model(model_name='test_trained_model', defined_model='test_defined_model',
                preprocessed_folder='data/temp_preprocessed')

    assert os.path.exists('models/test_trained_model.joblib')

    model = load('models/test_trained_model.joblib')
    x_test = load('data/temp_preprocessed/preprocessed_x_test.joblib')
    y_test = load('data/temp_preprocessed/preprocessed_y_test.joblib')

    metrics, _, _, _ = evaluate_model(
        model, x_test, y_test, save_data=False)

    avg_accuracy = metrics["accuracy"]
    avg_precision = metrics["precision"]
    avg_recall = metrics["recall"]
    avg_f1 = metrics["f1"]
    roc_auc = metrics["roc_auc"]

    assert avg_accuracy > 0, "Model accuracy is 0"
    assert avg_precision > 0, "Model precision is 0"
    assert avg_recall > 0, "Model recall is 0"
    assert avg_f1 > 0, "Model f1 is 0"
    assert roc_auc > 0, "Model roc_auc is 0"


if __name__ == '__main__':
    pytest.main([__file__])
