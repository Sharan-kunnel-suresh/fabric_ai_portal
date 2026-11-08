"""Generate a synthetic fabric dataset (good / defect) so the whole pipeline can be tested
without downloading anything.  For the real project use the Kaggle WFDD / TILDA data
(see prepare_dataset.py).

    python make_synthetic_dataset.py --n 1500 --defect-frac 0.3
"""
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import config

TINTS = [(0.95, 0.85, 0.75), (0.8, 0.9, 1.0), (0.85, 1.0, 0.85), (1.0, 0.9, 0.9), (0.9, 0.9, 0.9)]


def make_fabric(size, rng):
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    p = rng.integers(4, 9)
    ph = rng.uniform(0, 6.28)
    base = 0.55 + 0.18 * np.sin(2 * np.pi * x / p + ph) * np.sin(2 * np.pi * y / p)
    base += 0.06 * np.sin(2 * np.pi * x / (2 * p)) + rng.normal(0, 0.03, (size, size))
    tint = np.array(TINTS[rng.integers(len(TINTS))])
    img = np.clip(base[..., None] * tint, 0, 1)
    return Image.fromarray((img * 255).astype(np.uint8))


def add_defect(img, rng):
    s = img.size[0]
    kind = rng.choice(["hole", "stain", "thread", "tear"])
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    cx, cy = rng.integers(25, s - 25, 2)
    if kind == "hole":
        r = int(rng.integers(6, 13))
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(15, 15, 15, 255))
    elif kind == "stain":
        r = int(rng.integers(10, 20))
        col = tuple(int(c) for c in rng.integers(20, 110, 3)) + (215,)
        d.ellipse([cx - r, cy - r * 0.7, cx + r, cy + r * 0.7], fill=col)
    elif kind == "thread":
        y0 = int(cy)
        x0, x1 = int(rng.integers(0, s // 3)), int(rng.integers(2 * s // 3, s))
        d.line([x0, y0, x1, y0 + int(rng.integers(-3, 4))], fill=(255, 255, 255, 255), width=4)
    else:  # tear
        d.line([cx, cy - 20, cx + int(rng.integers(-8, 8)), cy + 20], fill=(10, 10, 10, 255), width=5)
    layer = layer.filter(ImageFilter.GaussianBlur(0.8))
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--defect-frac", type=float, default=0.3)
    ap.add_argument("--size", type=int, default=config.IMG_SIZE)
    a = ap.parse_args()
    rng = np.random.default_rng(config.SEED)
    for cls in config.CLASSES:
        (config.DATA_DIR / cls).mkdir(parents=True, exist_ok=True)
    n_def = int(a.n * a.defect_frac)
    for i in range(a.n):
        img = make_fabric(a.size, rng)
        cls = "good"
        if i < n_def:
            img, cls = add_defect(img, rng), "defect"
        img.save(config.DATA_DIR / cls / f"{cls}_{i:05d}.png")
    print(f"Saved {a.n - n_def} good + {n_def} defect images to {config.DATA_DIR}")


if __name__ == "__main__":
    main()
