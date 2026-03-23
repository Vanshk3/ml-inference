"""
Load testing with Locust.
Simulates concurrent users hitting the inference API.

Run with:
    locust -f tests/load_test.py --host=http://localhost:8000

Then open http://localhost:8089 to control the test.
Or run headless:
    locust -f tests/load_test.py --host=http://localhost:8000 \
           --users 20 --spawn-rate 5 --run-time 30s --headless
"""

import os
import random
from pathlib import Path
from locust import HttpUser, task, between
from PIL import Image
import io


def make_dummy_tyre_image(size=(224, 224)) -> bytes:
    """Generate a synthetic tyre-like image for load testing."""
    import numpy as np
    arr = np.random.randint(60, 130, (*size, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def get_test_images(n: int = 5) -> list[bytes]:
    """Load real test images if available, else use synthetic ones."""
    images = []
    test_dirs = [
        Path("../fleet_ai/data/test/good"),
        Path("../fleet_ai/data/test/defective"),
    ]
    for d in test_dirs:
        if d.exists():
            for p in list(d.glob("*.jpg"))[:n]:
                images.append(p.read_bytes())

    if not images:
        images = [make_dummy_tyre_image() for _ in range(n)]

    return images


TEST_IMAGES = get_test_images(10)


class InferenceUser(HttpUser):
    """
    Simulates a fleet management system hitting the inference API.
    Each user waits 0.5–2s between requests, mimicking realistic traffic.
    """
    wait_time = between(0.5, 2)

    @task(5)
    def predict_single(self):
        """Most common operation — single tyre inspection."""
        img_bytes = random.choice(TEST_IMAGES)
        self.client.post(
            "/predict",
            files={"file": ("tyre.jpg", img_bytes, "image/jpeg")},
        )

    @task(2)
    def check_health(self):
        """Health checks happen frequently in production."""
        self.client.get("/health")

    @task(1)
    def get_metrics(self):
        """Monitoring system polls metrics periodically."""
        self.client.get("/metrics")

    @task(1)
    def predict_batch(self):
        """Less frequent — batch inspection of multiple tyres."""
        n = random.randint(2, 5)
        files = [
            ("files", (f"tyre_{i}.jpg", random.choice(TEST_IMAGES), "image/jpeg"))
            for i in range(n)
        ]
        self.client.post("/predict/batch", files=files)