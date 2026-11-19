
import csv
import json
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from PIL import Image

from ai.model import build_model
from ai.tilda_data_prep import build_tilda_dataloaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "tilda"
MANIFEST_PATH = ROOT / "data" / "tilda_manifest.csv"


def load_manifest():
    mapping = {}

    with open(MANIFEST_PATH, "r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["split"] == "test":
                mapping[row["filename"]] = row

    return mapping


def main():
    loaders, _ = build_tilda_dataloaders(imbalance="sampler")
    test_loader = loaders["test"]

    with open(OUTPUT_DIR / "metrics.json", encoding="utf-8") as file:
        metrics = json.load(file)

    threshold = metrics["test_metrics"]["threshold"]

    checkpoint = torch.load(
        OUTPUT_DIR / "best.pt",
        map_location=DEVICE,
        weights_only=False
    )

    model = build_model(device=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    manifest = load_manifest()
    errors = []
    correct = []

    # Evaluate test images without changing the model.
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            probabilities = torch.softmax(model(images), dim=1)[:, 1]

            for i, probability in enumerate(probabilities.cpu().tolist()):
                actual = int(labels[i])
                predicted = int(probability >= threshold)

                path = Path(test_loader.dataset.paths[
                    len(errors) + len(correct)
                ])

                # The indexing above is replaced below by a stable
                # independent image counter.
                break

    # Collect predictions with their corresponding image paths.
    records = []

    with torch.no_grad():
        index = 0

        for images, labels in test_loader:
            images = images.to(DEVICE)
            probabilities = torch.softmax(model(images), dim=1)[:, 1]

            for i, probability in enumerate(probabilities.cpu().tolist()):
                path = Path(test_loader.dataset.paths[index])
                actual = int(labels[i])
                predicted = int(probability >= threshold)

                filename = path.name
                metadata = manifest.get(filename, {})

                record = {
                    "path": path,
                    "filename": filename,
                    "category": metadata.get("original_category", "unknown"),
                    "actual": actual,
                    "predicted": predicted,
                    "probability": probability
                }

                if actual == predicted:
                    correct.append(record)
                else:
                    errors.append(record)

                index += 1

    # Prioritise missed defects, then false alarms.
    errors.sort(
        key=lambda item: (
            item["actual"] == 0,
            -item["probability"]
        )
    )

    def plot_images(items, title, output_name, max_images=16):
        items = items[:max_images]

        if not items:
            print(f"No images to display for: {title}")
            return

        columns = 4
        rows = (len(items) + columns - 1) // columns

        fig, axes = plt.subplots(
            rows, columns,
            figsize=(14, rows * 3.5)
        )

        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for ax, item in zip(axes, items):
            image = Image.open(item["path"]).convert("RGB")
            ax.imshow(image)
            ax.axis("off")

            actual_label = "DEFECT" if item["actual"] else "GOOD"
            predicted_label = "DEFECT" if item["predicted"] else "GOOD"

            ax.set_title(
                f"{item['category']}\n"
                f"Actual: {actual_label}\n"
                f"Predicted: {predicted_label}\n"
                f"Defect probability: {item['probability']:.3f}",
                fontsize=8
            )

        for ax in axes[len(items):]:
            ax.axis("off")

        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / output_name, dpi=150)
        plt.close(fig)

    missed_defects = [
        item for item in errors
        if item["actual"] == 1 and item["predicted"] == 0
    ]

    false_alarms = [
        item for item in errors
        if item["actual"] == 0 and item["predicted"] == 1
    ]

    plot_images(
        missed_defects,
        "Missed Defects (False Negatives)",
        "missed_defects.png"
    )

    plot_images(
        false_alarms,
        "Good Fabric Incorrectly Flagged (False Positives)",
        "false_alarms.png"
    )

    plot_images(
        correct,
        "Correctly Classified Test Images",
        "correct_predictions.png"
    )

    print("\nERROR ANALYSIS COMPLETE")
    print(f"Threshold: {threshold:.4f}")
    print(f"Correct predictions: {len(correct)}")
    print(f"Total errors: {len(errors)}")
    print(f"Missed defects: {len(missed_defects)}")
    print(f"False alarms: {len(false_alarms)}")
    print("\nCharts saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
