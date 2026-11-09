
"""
Prepare the TILDA textile dataset for binary fabric-defect classification.

Original categories:
    good
    hole
    objects
    oil spot
    thread error

Binary classes:
    good
    defect

All patches sharing an original filename prefix remain in the same split.
"""

import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

SOURCE_DIR = Path("data/raw/tilda")
OUTPUT_DIR = Path("data/tilda_binary")
MANIFEST_PATH = Path("data/tilda_manifest.csv")

CATEGORIES = {
    "good": "good",
    "hole": "defect",
    "objects": "defect",
    "oil spot": "defect",
    "thread error": "defect",
}

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def collect_images():
    """Collect image paths and their original categories."""

    records = []

    for category, binary_label in CATEGORIES.items():
        folder = SOURCE_DIR / category

        if not folder.exists():
            raise FileNotFoundError(f"Missing category folder: {folder}")

        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:

                if "_patch" not in path.stem:
                    raise ValueError(
                        f"Unexpected filename format: {path.name}"
                    )

                group_id = path.stem.split("_patch")[0]

                records.append({
                    "source": path,
                    "category": category,
                    "label": binary_label,
                    "group": group_id,
                })

    if not records:
        raise RuntimeError("No TILDA images were found.")

    return records


def split_groups(records):
    """Assign complete original groups to train, validation or test."""

    grouped = defaultdict(list)

    for record in records:
        grouped[record["group"]].append(record)

    group_ids = sorted(grouped)

    rng = random.Random(SEED)
    rng.shuffle(group_ids)

    # All groups contain 64 images in the inspected dataset.
    total_groups = len(group_ids)

    n_train = round(total_groups * TRAIN_FRAC)
    n_val = round(total_groups * VAL_FRAC)

    train_groups = set(group_ids[:n_train])
    val_groups = set(group_ids[n_train:n_train + n_val])
    test_groups = set(group_ids[n_train + n_val:])

    assignments = {}

    for group in train_groups:
        assignments[group] = "train"

    for group in val_groups:
        assignments[group] = "val"

    for group in test_groups:
        assignments[group] = "test"

    return assignments, grouped


def prepare_output():
    """Create fresh output directories without touching the demo."""

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    for split in ["train", "val", "test"]:
        for label in ["good", "defect"]:
            (OUTPUT_DIR / split / label).mkdir(
                parents=True, exist_ok=True
            )


def main():
    records = collect_images()

    assignments, grouped = split_groups(records)

    prepare_output()

    counts = {
        split: {"good": 0, "defect": 0}
        for split in ["train", "val", "test"]
    }

    category_counts = defaultdict(int)
    manifest_rows = []

    for record in records:

        split = assignments[record["group"]]
        label = record["label"]

        destination = (
            OUTPUT_DIR
            / split
            / label
            / f"{record['category'].replace(' ', '_')}_{record['source'].name}"
        )

        shutil.copy2(record["source"], destination)

        counts[split][label] += 1
        category_counts[record["category"]] += 1

        manifest_rows.append({
            "filename": destination.name,
            "original_category": record["category"],
            "binary_label": label,
            "group_id": record["group"],
            "split": split,
            "original_path": str(record["source"]),
        })

    # Save a complete image-level manifest.
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "filename",
                "original_category",
                "binary_label",
                "group_id",
                "split",
                "original_path",
            ],
        )

        writer.writeheader()
        writer.writerows(manifest_rows)

    print("\nTILDA DATASET PREPARATION COMPLETE")
    print("=" * 50)

    print("Total images:", len(records))
    print("Original groups:", len(grouped))

    print("\nOriginal category counts:")
    for category, count in category_counts.items():
        print(f"{category:15s}: {count}")

    print("\nBinary class counts by split:")
    for split, class_counts in counts.items():
        total = sum(class_counts.values())

        print(f"\n{split.upper()} ({total} images)")
        print("  Good:  ", class_counts["good"])
        print("  Defect:", class_counts["defect"])

    # Confirm that no group appears in multiple splits.
    group_sets = {
        split: {
            group for group, assigned_split in assignments.items()
            if assigned_split == split
        }
        for split in ["train", "val", "test"]
    }

    assert group_sets["train"].isdisjoint(group_sets["val"])
    assert group_sets["train"].isdisjoint(group_sets["test"])
    assert group_sets["val"].isdisjoint(group_sets["test"])

    assert sum(sum(c.values()) for c in counts.values()) == len(records)

    print("\nGroup leakage check: PASSED")
    print("Manifest:", MANIFEST_PATH)
    print("Prepared dataset:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
