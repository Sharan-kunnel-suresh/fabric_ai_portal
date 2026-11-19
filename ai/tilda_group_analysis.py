
import csv
import json
from pathlib import Path
from collections import defaultdict

import torch

from ai.model import build_model
from ai.tilda_data_prep import build_tilda_dataloaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "tilda"
MANIFEST_PATH = ROOT / "data" / "tilda_manifest.csv"


def load_test_manifest():
    records = {}

    with open(MANIFEST_PATH, "r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["split"] == "test":
                records[row["filename"]] = row

    return records


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

    manifest = load_test_manifest()

    # Store predictions grouped by original source image.
    groups = defaultdict(lambda: {
        "good_total": 0,
        "good_false_positives": 0,
        "defect_total": 0,
        "defect_detected": 0
    })

    index = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)

            probabilities = torch.softmax(model(images), dim=1)[:, 1]

            for i, probability in enumerate(probabilities.cpu().tolist()):
                path = Path(test_loader.dataset.paths[index])
                filename = path.name

                if filename not in manifest:
                    raise ValueError(
                        f"Missing manifest entry for {filename}"
                    )

                metadata = manifest[filename]
                group_id = metadata["group_id"]
                actual = int(labels[i])
                predicted = int(probability >= threshold)

                if actual == 0:
                    groups[group_id]["good_total"] += 1

                    if predicted == 1:
                        groups[group_id]["good_false_positives"] += 1

                else:
                    groups[group_id]["defect_total"] += 1

                    if predicted == 1:
                        groups[group_id]["defect_detected"] += 1

                index += 1

    # Rank source images by their number of false positives.
    ranked_groups = []

    for group_id, values in groups.items():
        false_positives = values["good_false_positives"]

        if false_positives > 0:
            ranked_groups.append({
                "group_id": group_id,
                **values,
                "good_false_positive_rate": (
                    false_positives / values["good_total"]
                    if values["good_total"] else 0
                )
            })

    ranked_groups.sort(
        key=lambda x: (
            x["good_false_positives"],
            x["good_false_positive_rate"]
        ),
        reverse=True
    )

    total_false_positives = sum(
        x["good_false_positives"] for x in ranked_groups
    )

    groups_with_fp = len(ranked_groups)

    total_test_groups = len(groups)

    print("\nFALSE-POSITIVE GROUP ANALYSIS")
    print(f"Threshold: {threshold:.4f}")
    print(f"Test source-image groups: {total_test_groups}")
    print(f"Groups containing false positives: {groups_with_fp}")
    print(
        f"Percentage of groups affected: "
        f"{groups_with_fp / total_test_groups * 100:.2f}%"
    )
    print(f"Total false positives: {total_false_positives}")

    print("\nTOP 15 SOURCE IMAGE GROUPS")
    print(
        f"{'Group':12} {'Good':>6} {'FP':>6} "
        f"{'FP rate':>10} {'Defects':>9} {'Detected':>10}"
    )

    for item in ranked_groups[:15]:
        print(
            f"{item['group_id']:12} "
            f"{item['good_total']:6} "
            f"{item['good_false_positives']:6} "
            f"{item['good_false_positive_rate'] * 100:9.1f}% "
            f"{item['defect_total']:9} "
            f"{item['defect_detected']:10}"
        )

    report = {
        "threshold": threshold,
        "total_test_groups": total_test_groups,
        "groups_with_false_positives": groups_with_fp,
        "percentage_groups_affected": (
            groups_with_fp / total_test_groups * 100
        ),
        "total_false_positives": total_false_positives,
        "ranked_groups": ranked_groups
    }

    report_path = OUTPUT_DIR / "group_error_analysis.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print(f"\nFull report saved: {report_path}")


if __name__ == "__main__":
    main()
