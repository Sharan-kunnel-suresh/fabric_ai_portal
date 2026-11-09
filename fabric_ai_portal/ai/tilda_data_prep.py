
"""
TILDA DATA PREPARATION

Loads the already prepared group-based TILDA dataset.

Directory structure:
data/tilda_binary/
    train/good/
    train/defect/
    val/good/
    val/defect/
    test/good/
    test/defect/
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T

import config

TILDA_DIR = config.ROOT / "data" / "tilda_binary"

SPLITS = ["train", "val", "test"]
CLASSES = ["good", "defect"]

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def seed_everything(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collect_images(split):
    """Collect image paths and binary labels from one split."""

    paths = []
    labels = []

    for label_index, class_name in enumerate(CLASSES):
        folder = TILDA_DIR / split / class_name

        if not folder.exists():
            raise FileNotFoundError(f"Missing directory: {folder}")

        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                paths.append(str(path))
                labels.append(label_index)

    if not paths:
        raise RuntimeError(f"No images found in split: {split}")

    return paths, np.asarray(labels, dtype=np.int64)


def basic_transform(size=config.IMG_SIZE):
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
    ])


def compute_mean_std(paths, size=config.IMG_SIZE):
    """Calculate per-channel statistics using training images only."""

    transform = basic_transform(size)

    channel_sum = torch.zeros(3)
    channel_sum_sq = torch.zeros(3)
    pixel_count = 0

    for path in paths:
        with Image.open(path) as image:
            tensor = transform(image.convert("RGB"))

        channel_sum += tensor.sum(dim=(1, 2))
        channel_sum_sq += (tensor ** 2).sum(dim=(1, 2))
        pixel_count += tensor.shape[1] * tensor.shape[2]

    mean = channel_sum / pixel_count
    variance = channel_sum_sq / pixel_count - mean ** 2
    std = torch.sqrt(torch.clamp(variance, min=1e-8))

    return mean.tolist(), std.tolist()



def train_transform(mean, std):
    return T.Compose([
        T.Resize((config.IMG_SIZE, config.IMG_SIZE)),

        # Orientation augmentation
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(15),

        # Moderate appearance variation
        T.ColorJitter(
            brightness=0.15,
            contrast=0.15
        ),

        # Small positional variation
        T.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05)
        ),

        T.ToTensor(),
        T.Normalize(mean, std),
    ])



def evaluation_transform(mean, std):
    return T.Compose([
        T.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


class TildaDataset(Dataset):

    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)

        return tensor, int(self.labels[index])


def build_tilda_dataloaders(
    batch_size=config.BATCH_SIZE,
    imbalance="sampler",
    num_workers=0,
):
    """Create training, validation and test DataLoaders."""

    seed_everything()

    train_paths, train_labels = collect_images("train")
    val_paths, val_labels = collect_images("val")
    test_paths, test_labels = collect_images("test")

    # Normalization statistics are calculated from training data only.
    mean, std = compute_mean_std(train_paths)

    train_counts = np.bincount(train_labels, minlength=2)

    if np.any(train_counts == 0):
        raise ValueError("Both classes must be present in training data.")

    class_weights = train_counts.sum() / (2 * train_counts)

    sampler = None

    if imbalance == "sampler":
        sample_weights = class_weights[train_labels]

        generator = torch.Generator()
        generator.manual_seed(config.SEED)

        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_labels),
            replacement=True,
            generator=generator,
        )

    elif imbalance not in ("weights", "none"):
        raise ValueError("imbalance must be sampler, weights or none")

    datasets = {
        "train": TildaDataset(
            train_paths,
            train_labels,
            train_transform(mean, std),
        ),
        "val": TildaDataset(
            val_paths,
            val_labels,
            evaluation_transform(mean, std),
        ),
        "test": TildaDataset(
            test_paths,
            test_labels,
            evaluation_transform(mean, std),
        ),
    }

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=num_workers,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }

    info = {
        "dataset": "TILDA",
        "mean": mean,
        "std": std,
        "img_size": config.IMG_SIZE,
        "classes": CLASSES,
        "n_train": len(train_labels),
        "n_val": len(val_labels),
        "n_test": len(test_labels),
        "train_class_counts": train_counts.tolist(),
        "val_class_counts": np.bincount(val_labels, minlength=2).tolist(),
        "test_class_counts": np.bincount(test_labels, minlength=2).tolist(),
        "class_weights": class_weights.tolist(),
        "imbalance_strategy": imbalance,
    }

    stats_path = config.OUT_DIR / "tilda_data_stats.json"
    stats_path.write_text(json.dumps(info, indent=2))

    return loaders, info


if __name__ == "__main__":

    loaders, info = build_tilda_dataloaders()

    print("\nTILDA DATA LOADER TEST")
    print("=" * 40)
    print(json.dumps(info, indent=2))

    images, labels = next(iter(loaders["train"]))

    print("\nFirst training batch:")
    print("Image tensor shape:", tuple(images.shape))
    print("Label tensor shape:", tuple(labels.shape))
    print("Good samples:", (labels == 0).sum().item())
    print("Defect samples:", (labels == 1).sum().item())
