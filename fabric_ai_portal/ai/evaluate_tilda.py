
"""
Independent evaluation of the real TILDA fabric defect model.

Usage:
    python -m ai.evaluate_tilda
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report,
)

import config
from ai.model import build_model
from ai.tilda_data_prep import build_tilda_dataloaders


OUT = config.OUT_DIR / "tilda"
CHECKPOINT = OUT / "best.pt"


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Collect predicted probabilities and true labels."""

    model.eval()

    probabilities = []
    labels = []

    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        total_loss += loss.item()

        probs = torch.softmax(outputs, dim=1)[:, 1]

        probabilities.extend(probs.cpu().numpy())
        labels.extend(targets.cpu().numpy())

    average_loss = total_loss / len(loader.dataset)

    return (
        np.asarray(probabilities),
        np.asarray(labels),
        average_loss,
    )


def calculate_metrics(y_true, probabilities, threshold):
    """Calculate binary classification metrics."""

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "F1": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "TPR": float(tp / max(tp + fn, 1)),
        "FPR": float(fp / max(fp + tn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
    }


def plot_confusion(metrics):
    """Create the test confusion matrix."""

    matrix = np.array([
        [metrics["TN"], metrics["FP"]],
        [metrics["FN"], metrics["TP"]],
    ])

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.imshow(matrix, cmap="Blues")

    for row in range(2):
        for col in range(2):
            ax.text(
                col,
                row,
                matrix[row, col],
                ha="center",
                va="center",
                fontsize=15,
            )

    ax.set_xticks([0, 1], labels=["Good", "Defect"])
    ax.set_yticks([0, 1], labels=["Good", "Defect"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title("TILDA Test Confusion Matrix")

    plt.tight_layout()
    plt.savefig(OUT / "confusion_matrix.png", dpi=150)
    plt.close()


def plot_roc(y_true, probabilities, auc):
    """Create ROC curve using the independent test set."""

    fpr, tpr, _ = roc_curve(y_true, probabilities)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"Test AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("TILDA Test ROC Curve")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(OUT / "roc_curve.png", dpi=150)
    plt.close()


def plot_training_history():
    """Visualize training and validation curves."""

    history_path = OUT / "history.json"

    history = json.loads(history_path.read_text())

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history["train_loss"], label="Training")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, history["train_acc"], label="Training")
    axes[1].plot(epochs, history["val_acc"], label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(OUT / "training_curves.png", dpi=150)
    plt.close()


def plot_threshold_sweep(y_true, probabilities):
    """Show validation precision and recall at different thresholds."""

    thresholds = np.linspace(0.01, 0.99, 99)

    precisions = []
    recalls = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        precisions.append(
            precision_score(y_true, predictions, zero_division=0)
        )

        recalls.append(
            recall_score(y_true, predictions, zero_division=0)
        )

    plt.figure(figsize=(7, 5))

    plt.plot(thresholds, precisions, label="Precision")
    plt.plot(thresholds, recalls, label="Recall")

    plt.xlabel("Classification threshold")
    plt.ylabel("Score")
    plt.title("Validation Threshold Analysis")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(OUT / "threshold_sweep.png", dpi=150)
    plt.close()


def main():

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Best checkpoint not found: {CHECKPOINT}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 55)
    print("TILDA MODEL EVALUATION")
    print("=" * 55)

    loaders, info = build_tilda_dataloaders(
        imbalance="sampler"
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    model = build_model(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Device:", device)
    print("Best training epoch:", checkpoint["epoch"] + 1)

    # Validation set: choose the classification threshold.
    val_probabilities, val_labels, val_loss = collect_predictions(
        model,
        loaders["val"],
        device,
    )

    # Test set: independent final evaluation.
    test_probabilities, test_labels, test_loss = collect_predictions(
        model,
        loaders["test"],
        device,
    )

    # Choose threshold using validation Youden J statistic.
    val_fpr, val_tpr, thresholds = roc_curve(
        val_labels,
        val_probabilities,
    )

    valid = np.isfinite(thresholds)

    if not valid.any():
        raise RuntimeError("Unable to determine a validation threshold.")

    youden = val_tpr - val_fpr

    best_index = np.argmax(np.where(valid, youden, -np.inf))

    best_threshold = float(thresholds[best_index])

    # Calculate metrics on independent test data.
    test_metrics = calculate_metrics(
        test_labels,
        test_probabilities,
        best_threshold,
    )

    test_auc = float(
        roc_auc_score(test_labels, test_probabilities)
    )

    val_auc = float(
        roc_auc_score(val_labels, val_probabilities)
    )

    metrics = {
        "dataset": "TILDA",
        "best_epoch": int(checkpoint["epoch"] + 1),
        "val_loss": float(val_loss),
        "test_loss": float(test_loss),
        "auc_validation": val_auc,
        "auc_test": test_auc,
        "threshold_selection": "Validation Youden J",
        "test_metrics": test_metrics,
        "test_at_0.5": calculate_metrics(
            test_labels,
            test_probabilities,
            0.5,
        ),
    }

    # Save results.
    (OUT / "metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )

    plot_confusion(test_metrics)
    plot_roc(test_labels, test_probabilities, test_auc)
    plot_training_history()
    plot_threshold_sweep(val_labels, val_probabilities)

    report = classification_report(
        test_labels,
        (test_probabilities >= best_threshold).astype(int),
        labels=[0, 1],
        target_names=["Good", "Defect"],
        zero_division=0,
        output_dict=True,
    )

    (OUT / "classification_report.json").write_text(
        json.dumps(report, indent=2)
    )

    print("\nINDEPENDENT TEST RESULTS")
    print("=" * 55)
    print(json.dumps(metrics, indent=2))

    print("\nCLASSIFICATION REPORT")
    print(classification_report(
        test_labels,
        (test_probabilities >= best_threshold).astype(int),
        labels=[0, 1],
        target_names=["Good", "Defect"],
        zero_division=0,
    ))

    print("\nFiles saved in:", OUT)


if __name__ == "__main__":
    main()
