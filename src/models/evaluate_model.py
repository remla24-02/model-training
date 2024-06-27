"""
Evaluate the trained model using various metrics and save the results to files.
"""

import json
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def save_metrics(metrics, filepath):
    """
    Save the metrics dictionary to a JSON file.

    Args:
        metrics (dict): The metrics dictionary to be saved.
        filepath (str): The path to the JSON file.

    Returns:
        None
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metrics, f)


def save_plot_data(data, filepath):
    """
    Save the plot data to a JSON file.

    Args:
        data (dict): The plot data to be saved.
        filepath (str): The path to the JSON file.

    Returns:
        None
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f)


def plot_and_save_roc(y_test, y_pred_binary, filepath):
    """
    Plots the Receiver Operating Characteristic (ROC) curve and saves it to a file.

    Parameters:
    - y_test (array-like): The true labels of the test data.
    - y_pred_binary (array-like): The predicted binary labels of the test data.
    - filepath (str): The file path to save the ROC curve plot.

    Returns:
    - dict: A dictionary containing the false positive rates (fpr)
            and true positive rates (tpr) as lists.

    """
    fpr, tpr, _ = roc_curve(y_test, y_pred_binary)
    plt.figure()
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig(filepath)
    plt.close()
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist()}


def plot_and_save_prc(y_test, y_pred_binary, filepath):
    """
    Plots the Precision-Recall Curve and saves it to a file.

    Args:
        y_test (array-like): The true labels of the test data.
        y_pred_binary (array-like): The predicted binary labels of the test data.
        filepath (str): The path to save the plot.

    Returns:
        dict: A dictionary containing the precision and recall values as lists.

    """
    precision, recall, _ = precision_recall_curve(y_test, y_pred_binary)
    plt.figure()
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(filepath)
    plt.close()
    return {"precision": precision.tolist(), "recall": recall.tolist()}


def plot_and_save_cm(y_test, y_pred_binary, filepath):
    """
    Plot and save the confusion matrix.

    Args:
        y_test (array-like): The true labels.
        y_pred_binary (array-like): The predicted labels.
        filepath (str): The file path to save the plot.

    Returns:
        dict: A dictionary containing the confusion matrix, actual labels, and predicted labels.
    """
    cm = confusion_matrix(y_test, y_pred_binary)
    plt.figure()
    plt.imshow(cm, interpolation='nearest',
               cmap=mcolors.ListedColormap(['blue']))
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Negative', 'Positive'], rotation=45)
    plt.yticks(tick_marks, ['Negative', 'Positive'])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filepath)
    plt.close()
    return {
        "confusion_matrix": cm.tolist(),
        "actual": y_test.flatten().tolist(),
        "predicted": y_pred_binary.flatten().tolist()
    }


def evaluate_model(model, x_test, y_test, batch_size=1000, save_data=True):
    """
    Evaluates a machine learning model using various metrics.

    Args:
        model (object): The trained machine learning model.
        x_test (numpy.ndarray): The input features for testing.
        y_test (numpy.ndarray): The target labels for testing.
        batch_size (int, optional): The batch size for prediction. Defaults to 1000.
        save_data (bool, optional): Whether to save the evaluation data. Defaults to True.

    Returns:
        tuple: A tuple containing the evaluation metrics, ROC data, PRC data, and CM data.

    Raises:
        None

    Examples:
        >>> model = create_model()
        >>> x_test = load_test_data()
        >>> y_test = load_test_labels()
        >>> metrics, roc_data, prc_data, cm_data = evaluate_model(model, x_test, y_test)
    """
    y_pred = model.predict(x_test, batch_size=batch_size)
    y_pred_binary = (np.array(y_pred) > 0.5).astype(int)
    y_test = y_test.reshape(-1, 1)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred_binary),
        "precision": precision_score(y_test, y_pred_binary),
        "recall": recall_score(y_test, y_pred_binary),
        "f1": f1_score(y_test, y_pred_binary),
        "roc_auc": roc_auc_score(y_test, y_pred_binary)
    }

    roc_data = None
    prc_data = None
    cm_data = None

    if save_data:
        os.makedirs('evaluation/plots', exist_ok=True)
        roc_data = plot_and_save_roc(
            y_test, y_pred_binary, 'evaluation/plots/roc.png')
        prc_data = plot_and_save_prc(
            y_test, y_pred_binary, 'evaluation/plots/prc.png')
        cm_data = plot_and_save_cm(
            y_test, y_pred_binary, 'evaluation/plots/cm.png')

    return metrics, roc_data, prc_data, cm_data


def main(save_data=True, trained_model='trained_model', preprocessed_folder='data/preprocessed'):
    """
    Main function for evaluating a trained model.

    Loads the trained model, test data, evaluates the model using the test data,
    and saves the evaluation metrics and plot data.

    Args:
        None

    Returns:
        None
    """
    model = load(f'models/{trained_model}.joblib')

    x_test = load(f'{preprocessed_folder}/preprocessed_x_test.joblib')
    y_test = load(f'{preprocessed_folder}/preprocessed_y_test.joblib')

    metrics, roc_data, prc_data, cm_data = evaluate_model(
        model, x_test, y_test, save_data)

    if save_data:
        os.makedirs('evaluation', exist_ok=True)
        save_metrics(metrics, 'evaluation/metrics.json')
        save_plot_data(roc_data, 'evaluation/plots/roc.json')
        save_plot_data(prc_data, 'evaluation/plots/prc.json')
        save_plot_data(cm_data, 'evaluation/plots/cm.json')


if __name__ == "__main__":
    main()
