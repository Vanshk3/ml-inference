"""
Model loader — loads the Fleet AI MobileNetV2 model once at startup
and keeps it in memory for all requests.
"""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path("models/best_model.pth")
META_PATH  = Path("models/metadata.json")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _build_model(num_classes: int = 2) -> nn.Module:
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )
    return model


class ModelRegistry:
    """Singleton that holds the loaded model and metadata."""

    _instance = None

    def __init__(self):
        self.model = None
        self.meta  = None
        self.classes = []
        self.load_time_ms = 0
        self.loaded = False

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self):
        if self.loaded:
            return
        t0 = time.time()

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Copy best_model.pth from your Fleet AI project into models/"
            )

        with open(META_PATH) as f:
            self.meta = json.load(f)

        self.classes = self.meta.get("classes", ["defective", "good"])
        self.model = _build_model(num_classes=len(self.classes))
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        self.model.to(DEVICE)
        self.model.eval()
        self.load_time_ms = round((time.time() - t0) * 1000, 2)
        self.loaded = True
        print(f"Model loaded in {self.load_time_ms}ms on {DEVICE}")

    def predict(self, image: Image.Image) -> dict:
        img_tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)
        t0 = time.perf_counter()

        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs   = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        pred_idx   = int(np.argmax(probs))
        pred_class = self.classes[pred_idx]
        confidence = round(float(probs[pred_idx]) * 100, 1)

        return {
            "class":      pred_class,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "all_probs":  {c: round(float(p) * 100, 1) for c, p in zip(self.classes, probs)},
        }