#imports
import json
import csv
from pathlib import Path
from collections import defaultdict

import torch
import matplotlib.pyplot as plt
from ai.model import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from ai.tilda_data_prep import build_tilda_dataloaders


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tilda"
MANIFEST_PATH = PROJECT_ROOT / "data" / "tilda_manifest.csv"
CHECKPOINT_PATH = OUTPUT_DIR / "best.pt"

DEFECT_TYPES = ["hole", "objects", "oil spot", "thread error"]


def load_manifest():
    """Map each prepared test filename to its original TILDA category."""
    mapping = {}

    with open(MANIFEST_PATH, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["split"] == "test":
                mapping[row["filename"]] = row["original_category"]

    return mapping


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    # Load the same test preprocessing used during evaluation.
    loaders, _ = build_tilda_dataloaders(imbalance="sampler")
    test_loader = loaders["test"]

    # Load the threshold selected using validation data.
    metrics_path = OUTPUT_DIR / "metrics.json"
    with open(metrics_path, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    threshold = metrics["test_metrics"]["threshold"]

    # Load the trained model.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model = build_model(device=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    filename_to_category = load_manifest()

    predictions = []
    actual_labels = []
    filenames = []

    # Generate predictions without changing the model.
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]

            predictions.extend(probabilities.cpu().tolist())
            actual_labels.extend(labels.tolist())

    # The test dataset is not shuffled, so its order matches the
    # dataset paths used by the DataLoader.
    filenames = [
        Path(path).name for path, _ in test_loader.dataset.samples
    ] if hasattr(test_loader.dataset, "samples") else None

    # Fall back to the dataset's stored paths if it uses a custom dataset.
    if filenames is None:
        filenames = [
            Path(path).name for path in test_loader.dataset.paths
        ]

    if not (len(filenames) == len(actual_labels) == len(predictions)):
        raise RuntimeError("Prediction and filename counts do not match.")

    results = defaultdict(lambda: {
        "support": 0,
        "detected": 0,
        "missed": 0,
        "probability_sum": 0.0
    })

    for filename, actual, probability in zip(
        filenames, actual_labels, predictions
    ):
        category = filename_to_category.get(filename)

        if category is None:
            raise ValueError(f"Filename missing from manifest: {filename}")

        if actual == 1:
            results[category]["support"] += 1
            results[category]["probability_sum"] += probability

            if probability >= threshold:
                results[category]["detected"] += 1
            else:
                results[category]["missed"] += 1

    report = {
        "dataset": "TILDA",
        "threshold": threshold,
        "threshold_source": "Validation Youden J",
        "defect_type_results": {}
    }

    print("\nDEFECT-TYPE DETECTION RESULTS")
    print(f"Threshold: {threshold:.4f}\n")

    for category in DEFECT_TYPES:
        values = results[category]
        support = values["support"]

        recall = (
            values["detected"] / support
            if support else 0.0
        )

        mean_probability = (
            values["probability_sum"] / support
            if support else 0.0
        )

        report["defect_type_results"][category] = {
            "support": support,
            "detected": values["detected"],
            "missed": values["missed"],
            "recall": recall,
            "mean_defect_probability": mean_probability
        }

        print(
            f"{category:15} | "
            f"Samples: {support:3} | "
            f"Detected: {values['detected']:3} | "
            f"Missed: {values['missed']:3} | "
            f"Recall: {recall:.4f}"
        )

    # Save numerical results.
    report_path = OUTPUT_DIR / "defect_type_analysis.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    # Create a defect-type recall chart.
    categories = DEFECT_TYPES
    recalls = [
        report["defect_type_results"][category]["recall"]
        for category in categories
    ]

    plt.figure(figsize=(9, 5))
    plt.bar(categories, recalls)
    plt.ylim(0, 1)
    plt.ylabel("Recall")
    plt.xlabel("Defect type")
    plt.title("TILDA Test Set: Defect Detection Recall")
    plt.xticks(rotation=15)
    plt.tight_layout()

    chart_path = OUTPUT_DIR / "defect_type_recall.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()

    print(f"\nReport saved: {report_path}")
    print(f"Chart saved:  {chart_path}")


if __name__ == "__main__":
    main()
