# Run the old and new models and compare the results.
#
# Path: tests/ml_infrastructure/test_AB.py
import pytest
from joblib import load
from src.models.evaluate_model import evaluate_model
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


def test_AB():
    new_model = load('models/trained_model.joblib')
    get_model("models/old_model.joblib")
    old_model = load('models/old_model.joblib')

    x_test = load('data/preprocessed/preprocessed_x_test.joblib')
    y_test = load('data/preprocessed/preprocessed_y_test.joblib')

    metrics_old, _, _, _ = evaluate_model(
        old_model, x_test, y_test, save_data=False)

    metrics_new, _, _, _ = evaluate_model(
        new_model, x_test, y_test, save_data=False)

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
