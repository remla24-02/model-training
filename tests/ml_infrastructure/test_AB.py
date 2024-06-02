# Run the old and new models and compare the results.
#
# Path: tests/ml_infrastructure/test_AB.py
import os
import pytest
from joblib import load
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.models.get_model import main as get_model

METRIC_WEIGHTS = {
    'accuracy': 0.25,
    'precision': 0.25,
    'recall': 0.25,
    'f1': 0.25
}

# Acceptable thresholds for each metric
THRESHOLDS = {
    'accuracy': 0.01,  # New model should not be more than 1% worse
    'precision': 0.01,
    'recall': 0.01,
    'f1': 0.01
}


@pytest.mark.skipif(not os.path.exists('models/trained_model.joblib'), reason="Trained model does not exist")
def test_AB():
    new_model = load('models/trained_model.joblib')
    get_model("old_model")
    old_model = load('models/old_model.joblib')

    x_test = load('data/preprocessed/preprocessed_x_test.joblib')
    y_test = load('data/preprocessed/preprocessed_y_test.joblib')

    y_pred_old = old_model.predict(x_test)
    y_pred_binary_old = (y_pred_old > 0.5).astype(int).reshape(-1, 1)

    y_pred_new = new_model.predict(x_test)
    y_pred_binary_new = (y_pred_new > 0.5).astype(int).reshape(-1, 1)

    metrics_old = {
        'accuracy': accuracy_score(y_test, y_pred_binary_old),
        'precision': precision_score(y_test, y_pred_binary_old),
        'recall': recall_score(y_test, y_pred_binary_old),
        'f1': f1_score(y_test, y_pred_binary_old)
    }

    metrics_new = {
        'accuracy': accuracy_score(y_test, y_pred_binary_new),
        'precision': precision_score(y_test, y_pred_binary_new),
        'recall': recall_score(y_test, y_pred_binary_new),
        'f1': f1_score(y_test, y_pred_binary_new)
    }

    total_weight = sum(METRIC_WEIGHTS.values())
    weighted_score = 0

    for metric, weight in METRIC_WEIGHTS.items():
        old_value = metrics_old[metric]
        new_value = metrics_new[metric]
        threshold = THRESHOLDS[metric]

        if new_value >= old_value - threshold:
            weighted_score += weight * (new_value / old_value)

    assert weighted_score / total_weight >= 1, f"New model's overall performance did not meet expectations. " \
                                               f"Weighted score: {weighted_score / total_weight}"

    print(f"Old Model Metrics: {metrics_old}")
    print(f"New Model Metrics: {metrics_new}")
    print(f"Weighted Score: {weighted_score / total_weight}")


if __name__ == '__main__':
    pytest.main()
