"""TRAINED MODEL INTEGRATION (max 35 p)
Loads the trained checkpoint and exposes Classifier.predict(image) -> dict.
The SCADA worker sends a camera image in, receives the prediction and writes it to the PLC.

    python -m ai.inference path/to/image.png
"""
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

import config
from ai.data_prep import eval_tf
from ai.model import build_model


class Classifier:
    def __init__(self, ckpt=config.OUT_DIR / "best.pt", device="cpu"):
        self.device = device
        stats = json.loads((config.OUT_DIR / "data_stats.json").read_text())
        thr_file = config.OUT_DIR / "threshold.json"
        self.threshold = json.loads(thr_file.read_text())["threshold"] if thr_file.exists() else 0.5
        self.tf = eval_tf(stats["mean"], stats["std"], stats["img_size"])
        self.model = build_model(device)
        self.model.load_state_dict(torch.load(ckpt, map_location=device)["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, image):
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        t0 = time.perf_counter()
        x = self.tf(image.convert("RGB")).unsqueeze(0).to(self.device)
        p_def = torch.softmax(self.model(x), 1)[0, 1].item()
        is_def = p_def >= self.threshold
        return {"label": config.CLASSES[int(is_def)], "is_defect": bool(is_def), "p_defect": p_def,
                "p_good": 1 - p_def, "threshold": self.threshold,
                "latency_ms": (time.perf_counter() - t0) * 1000}


if __name__ == "__main__":
    clf = Classifier()
    for f in sys.argv[1:]:
        print(f, clf.predict(f))
