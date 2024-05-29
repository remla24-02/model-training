from joblib import load
import pytest
import os
from src.data.get_data import main as get_data
from lib_ml_remla24_team02 import data_preprocessing
from src.models.define_model import main as define_model
from src.models.train_model import main as train_model
from src.models.evaluate_model import evaluate_model


def keep_first_1000_lines(file_path):
    with open(file_path, 'r') as file:
        lines = [file.readline() for _ in range(1000)]

    with open(file_path, 'w') as file:
        file.writelines(lines)


def test_model_pipeline():
    get_data()

    assert os.path.exists('data/raw/train.txt')
    assert os.path.exists('data/raw/test.txt')
    assert os.path.exists('data/raw/val.txt')

    keep_first_1000_lines('data/raw/train.txt')
    keep_first_1000_lines('data/raw/test.txt')
    keep_first_1000_lines('data/raw/val.txt')

    data_preprocessing.preprocess(os.path.join(
        'data', 'raw'), os.path.join('data', 'preprocessed'))

    assert os.path.exists('data/preprocessed/preprocessed_x_train.joblib')
    assert os.path.exists('data/preprocessed/preprocessed_y_train.joblib')
    assert os.path.exists('data/preprocessed/preprocessed_x_test.joblib')
    assert os.path.exists('data/preprocessed/preprocessed_y_test.joblib')
    assert os.path.exists('data/preprocessed/preprocessed_x_val.joblib')
    assert os.path.exists('data/preprocessed/preprocessed_y_val.joblib')

    define_model()

    assert os.path.exists('models/defined_model.joblib')

    train_model()

    assert os.path.exists('models/trained_model.joblib')

    model = load('models/trained_model.joblib')
    x_test = load('data/preprocessed/preprocessed_x_test.joblib')
    y_test = load('data/preprocessed/preprocessed_y_test.joblib')

    avg_accuracy, avg_precision, avg_recall, avg_f1, roc_auc = evaluate_model(
        model, x_test, y_test, None)

    assert avg_accuracy > 0, "Model accuracy is 0"
    assert avg_precision > 0, "Model precision is 0"
    assert avg_recall > 0, "Model recall is 0"
    assert avg_f1 > 0, "Model f1 is 0"
    assert roc_auc > 0, "Model roc_auc is 0"


if __name__ == '__main__':
    pytest.main([__file__])
