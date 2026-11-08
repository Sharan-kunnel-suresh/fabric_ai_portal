"""Turn a raw Kaggle fabric download (WFDD / TILDA) into  data/dataset/{good,defect}.

Rule: any image whose path contains a folder called 'good' is GOOD, every other image is
DEFECT.  Ground-truth masks and mask folders are skipped.

    python prepare_dataset.py --src data/raw/WFDD
"""
import argparse
import shutil
from pathlib import Path
import config

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(config.RAW_DIR))
    a = ap.parse_args()
    src = Path(a.src)
    for cls in config.CLASSES:
        (config.DATA_DIR / cls).mkdir(parents=True, exist_ok=True)
    counts = {"good": 0, "defect": 0}
    for p in sorted(src.rglob("*")):
        if p.suffix.lower() not in EXT:
            continue
        parts = [x.lower() for x in p.relative_to(src).parts[:-1]]
        if any("ground_truth" in x or "mask" in x for x in parts):
            continue
        cls = "good" if "good" in parts else "defect"
        name = "_".join(p.relative_to(src).parts)
        shutil.copy2(p, config.DATA_DIR / cls / name)
        counts[cls] += 1
    print("Dataset ready:", counts, "->", config.DATA_DIR)
    if sum(counts.values()) < 1000:
        print("WARNING: the assignment asks for at least 1000 images.")


if __name__ == "__main__":
    main()
