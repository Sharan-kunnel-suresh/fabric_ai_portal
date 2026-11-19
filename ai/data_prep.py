"""DATA PREPARATION (max 35 p)
Image resizer, standardization/normalizer, index splitter, data augmentation,
image transforms, imbalanced classes handling, training/validation/test sets, tensor transforms.
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

import config

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def seed_everything(seed=config.SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def list_images(data_dir=config.DATA_DIR):
    paths, labels = [], []
    for idx, cls in enumerate(config.CLASSES):
        for p in sorted((Path(data_dir) / cls).rglob("*")):
            if p.suffix.lower() in EXT:
                paths.append(str(p)); labels.append(idx)
    if not paths:
        raise FileNotFoundError(f"No images in {data_dir}. Run prepare_dataset.py or make_synthetic_dataset.py")
    return paths, np.array(labels)


# ---- Index splitter (stratified: keeps the class ratio in every split) ----
def index_splitter(labels, val_frac=config.VAL_FRAC, test_frac=config.TEST_FRAC, seed=config.SEED):
    idx = np.arange(len(labels))
    train_idx, rest = train_test_split(idx, test_size=val_frac + test_frac, stratify=labels, random_state=seed)
    val_idx, test_idx = train_test_split(rest, test_size=test_frac / (val_frac + test_frac),
                                         stratify=labels[rest], random_state=seed)
    return train_idx, val_idx, test_idx


# ---- Image resizer + tensor transform (used to compute the statistics) ----
def basic_tf(size=config.IMG_SIZE):
    return T.Compose([T.Resize((size, size)), T.ToTensor()])


# ---- Standardization: per-channel mean/std measured on the TRAINING set only ----
def compute_mean_std(paths, size=config.IMG_SIZE):
    tf = basic_tf(size)
    n, s, s2 = 0, torch.zeros(3), torch.zeros(3)
    for p in paths:
        x = tf(Image.open(p).convert("RGB"))
        s += x.sum((1, 2)); s2 += (x ** 2).sum((1, 2)); n += x.shape[1] * x.shape[2]
    mean = s / n
    std = (s2 / n - mean ** 2).sqrt()
    return mean.tolist(), std.tolist()


# ---- Image transforms: augmentation for training, deterministic for val/test/inference ----
def train_tf(mean, std, size=config.IMG_SIZE):
    return T.Compose([
        T.Resize((size, size)),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(),     # data augmentation
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(), T.Normalize(mean, std),                 # tensor transforms + normalizer
    ])


def eval_tf(mean, std, size=config.IMG_SIZE):
    return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(mean, std)])


class FabricDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = paths, labels, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), int(self.labels[i])


def load_splits():
    """Read outputs/splits.json and return absolute paths."""
    d = json.loads((config.OUT_DIR / "splits.json").read_text())
    return {k: [str(config.ROOT / p) for p in v] for k, v in d.items()}


def build_dataloaders(batch_size=config.BATCH_SIZE, imbalance="sampler", num_workers=0):
    """imbalance: 'sampler' (WeightedRandomSampler), 'weights' (class-weighted loss) or 'none'."""
    seed_everything()
    paths, labels = list_images()
    tr, va, te = index_splitter(labels)
    paths = np.array(paths)
    mean, std = compute_mean_std(paths[tr])

    counts = np.bincount(labels[tr], minlength=len(config.CLASSES))
    class_w = counts.sum() / (len(counts) * counts)           # inverse-frequency weights
    sampler = None
    if imbalance == "sampler":
        sampler = WeightedRandomSampler(class_w[labels[tr]], num_samples=len(tr), replacement=True)

    ds = {
        "train": FabricDataset(paths[tr], labels[tr], train_tf(mean, std)),
        "val": FabricDataset(paths[va], labels[va], eval_tf(mean, std)),
        "test": FabricDataset(paths[te], labels[te], eval_tf(mean, std)),
    }
    loaders = {
        "train": DataLoader(ds["train"], batch_size, sampler=sampler, shuffle=sampler is None, num_workers=num_workers),
        "val": DataLoader(ds["val"], batch_size, num_workers=num_workers),
        "test": DataLoader(ds["test"], batch_size, num_workers=num_workers),
    }
    info = {
        "mean": mean, "std": std, "img_size": config.IMG_SIZE, "classes": config.CLASSES,
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "train_class_counts": counts.tolist(), "class_weights": class_w.tolist(),
        "imbalance_strategy": imbalance,
    }
    (config.OUT_DIR / "data_stats.json").write_text(json.dumps(info, indent=2))
    rel = lambda a: [Path(x).relative_to(config.ROOT).as_posix() for x in a]      # portable paths
    (config.OUT_DIR / "splits.json").write_text(json.dumps(
        {"train": rel(paths[tr]), "val": rel(paths[va]), "test": rel(paths[te])}))
    return loaders, info


if __name__ == "__main__":
    _, info = build_dataloaders()
    print(json.dumps(info, indent=2))
