"""
Use trained models to make predictions.
"""

from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, precision_score
import numpy as np
from joblib import load
from dvclive import Live


def evaluate_model(model, x_test, y_test, batch_size=1000, live: Live = None):
    """
    Evaluate the model with the test data.
    """
    y_pred = model.predict(x_test, batch_size=batch_size)
    y_pred_binary = (np.array(y_pred) > 0.5).astype(int)
    y_test = y_test.reshape(-1, 1)

    avg_accuracy = accuracy_score(y_test, y_pred_binary)
    avg_precision = precision_score(y_test, y_pred_binary)
    avg_recall = recall_score(y_test, y_pred_binary)
    avg_f1 = f1_score(y_test, y_pred_binary)
    roc_auc = roc_auc_score(y_test, y_pred_binary)

    if live is not None:
        if not live.summary:
            live.summary = {"accuracy": avg_accuracy, "precision": avg_precision,
                            "recall": avg_recall, "f1": avg_f1, "roc_auc": roc_auc}

        live.log_sklearn_plot("roc", y_test, y_pred_binary,
                              name="roc", drop_intermediate=True)
        live.log_sklearn_plot("precision_recall", y_test.astype(float), y_pred_binary.astype(float),
                              name="prc", drop_intermediate=True)
        live.log_sklearn_plot("confusion_matrix", y_test.squeeze(),
                              y_pred_binary.squeeze(), name="cm")

    return avg_accuracy, avg_precision, avg_recall, avg_f1


def main():
    """
    Make predictions with the defined model.
    """
    model = load('models/trained_model.joblib')

    x_test = load('data/preprocessed/preprocessed_x_test.joblib')
    y_test = load('data/preprocessed/preprocessed_y_test.joblib')

    with Live("evaluation") as live:
        evaluate_model(model, x_test, y_test, live)


if __name__ == "__main__":
    main()
