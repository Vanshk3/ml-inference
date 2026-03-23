"""
Request logger — every inference is logged to logs/requests.jsonl
One JSON object per line for easy streaming reads.
"""

import json
import time
import uuid
from pathlib import Path
from datetime import datetime, timezone
from threading import Lock

LOG_PATH = Path("logs/requests.jsonl")
LOG_PATH.parent.mkdir(exist_ok=True)
_lock = Lock()


def log_request(
    endpoint: str,
    prediction: dict,
    status_code: int,
    error: str = None,
):
    record = {
        "id":           str(uuid.uuid4())[:8],
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "endpoint":     endpoint,
        "status_code":  status_code,
        "prediction":   prediction.get("class") if prediction else None,
        "confidence":   prediction.get("confidence") if prediction else None,
        "latency_ms":   prediction.get("latency_ms") if prediction else None,
        "error":        error,
    }
    with _lock:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")


def read_logs(n: int = None) -> list[dict]:
    if not LOG_PATH.exists():
        return []
    with _lock:
        lines = LOG_PATH.read_text().strip().splitlines()
    records = [json.loads(l) for l in lines if l.strip()]
    return records[-n:] if n else records


def get_stats() -> dict:
    records = read_logs()
    if not records:
        return {
            "total_requests": 0,
            "success_rate":   0,
            "avg_latency_ms": 0,
            "p95_latency_ms": 0,
            "p99_latency_ms": 0,
            "defective_pct":  0,
            "good_pct":       0,
        }

    latencies   = [r["latency_ms"] for r in records if r["latency_ms"]]
    predictions = [r["prediction"] for r in records if r["prediction"]]
    successes   = [r for r in records if r["status_code"] == 200]

    import numpy as np
    return {
        "total_requests": len(records),
        "success_rate":   round(len(successes) / len(records) * 100, 1),
        "avg_latency_ms": round(float(np.mean(latencies)), 2) if latencies else 0,
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2) if latencies else 0,
        "p99_latency_ms": round(float(np.percentile(latencies, 99)), 2) if latencies else 0,
        "defective_pct":  round(predictions.count("defective") / len(predictions) * 100, 1) if predictions else 0,
        "good_pct":       round(predictions.count("good") / len(predictions) * 100, 1) if predictions else 0,
    }