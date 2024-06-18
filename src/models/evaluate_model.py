import os
import json
import numpy as np
from joblib import load
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, precision_score, roc_curve, precision_recall_curve, confusion_matrix


def save_metrics(metrics, filepath):
    with open(filepath, 'w') as f:
        json.dump(metrics, f)


def save_plot_data(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f)


def plot_and_save_roc(y_test, y_pred_binary, filepath):
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
    cm = confusion_matrix(y_test, y_pred_binary)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
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


def evaluate_model(model, x_test, y_test, batch_size=1000):
    y_pred = model.predict(x_test, batch_size=batch_size)
    y_pred_binary = (np.array(y_pred) > 0.5).astype(int)
    y_test = y_test.reshape(-1, 1)

    avg_accuracy = accuracy_score(y_test, y_pred_binary)
    avg_precision = precision_score(y_test, y_pred_binary)
    avg_recall = recall_score(y_test, y_pred_binary)
    avg_f1 = f1_score(y_test, y_pred_binary)
    roc_auc = roc_auc_score(y_test, y_pred_binary)

    metrics = {
        "accuracy": avg_accuracy,
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1,
        "roc_auc": roc_auc
    }

    os.makedirs('evaluation/plots', exist_ok=True)
    roc_data = plot_and_save_roc(
        y_test, y_pred_binary, 'evaluation/plots/roc.png')
    prc_data = plot_and_save_prc(
        y_test, y_pred_binary, 'evaluation/plots/prc.png')
    cm_data = plot_and_save_cm(
        y_test, y_pred_binary, 'evaluation/plots/cm.png')

    return metrics, roc_data, prc_data, cm_data


def main():
    model = load('models/trained_model.joblib')

    x_test = load('data/preprocessed/preprocessed_x_test.joblib')
    y_test = load('data/preprocessed/preprocessed_y_test.joblib')

    metrics, roc_data, prc_data, cm_data = evaluate_model(
        model, x_test, y_test)

    os.makedirs('evaluation', exist_ok=True)
    save_metrics(metrics, 'evaluation/metrics.json')
    save_plot_data(roc_data, 'evaluation/plots/roc.json')
    save_plot_data(prc_data, 'evaluation/plots/prc.json')
    save_plot_data(cm_data, 'evaluation/plots/cm.json')


if __name__ == "__main__":
    main()
