
"""
TILDA V1 MODEL INFERENCE
Loads the independently evaluated TILDA checkpoint
and provides predictions to the SCADA application.
"""

import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms as T

import config
from ai.model import build_model


class Classifier:

    def __init__(
        self,
        ckpt=config.OUT_DIR / "tilda" / "best.pt",
        device="cpu"
    ):
        self.device = torch.device(device)
        self.checkpoint_path = Path(ckpt)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"TILDA V1 checkpoint not found: {self.checkpoint_path}"
            )

        # Load the trained TILDA checkpoint.
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device
        )

        # Read normalisation statistics saved during TILDA training.
        stats_path = config.OUT_DIR / "tilda_data_stats.json"
        if not stats_path.exists():
            raise FileNotFoundError(f"TILDA normalisation statistics not found: {stats_path}"
                                    )
        with open(stats_path, "r", encoding="utf-8") as file:
            stats = json.load(file)
        mean = stats["mean"]
        std = stats["std"]
        image_size = checkpoint["img_size"]

        # Load the threshold selected using validation data.
        metrics_path = config.OUT_DIR / "tilda" / "metrics.json"

        if not metrics_path.exists():
            raise FileNotFoundError(
                f"TILDA evaluation metrics not found: {metrics_path}"
            )

        with open(metrics_path, "r", encoding="utf-8") as file:
            metrics = json.load(file)

        self.threshold = float(
            metrics["test_metrics"]["threshold"]
        )

        # Use exactly the evaluation preprocessing used for TILDA.
        self.tf = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

        # Build the same FabricNet architecture used during training.
        self.model = build_model(self.device)

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

        print("TILDA V1 model loaded successfully.")
        print("Checkpoint:", self.checkpoint_path)
        print("Device:", self.device)
        print("Image size:", image_size)
        print("Classification threshold:", self.threshold)

    @torch.no_grad()
    def predict(self, image):

        if isinstance(image, (str, Path)):
            with Image.open(image) as img:
                image = img.convert("RGB")
        else:
            image = image.convert("RGB")

        start_time = time.perf_counter()

        tensor = self.tf(image).unsqueeze(0).to(self.device)

        outputs = self.model(tensor)

        probabilities = torch.softmax(outputs, dim=1)[0]

        p_good = probabilities[0].item()
        p_defect = probabilities[1].item()

        is_defect = p_defect >= self.threshold

        latency = (time.perf_counter() - start_time) * 1000

        return {
            "label": "defect" if is_defect else "good",
            "is_defect": bool(is_defect),
            "p_defect": p_defect,
            "p_good": p_good,
            "threshold": self.threshold,
            "latency_ms": latency
        }


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python -m ai.inference path/to/image.png")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    classifier = Classifier(device=device)

    for image_path in sys.argv[1:]:
        result = classifier.predict(image_path)
        print(image_path, result)
