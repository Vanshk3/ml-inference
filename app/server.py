"""
Fleet AI — Real-Time ML Inference Server
FastAPI server with latency tracking, request logging, and health monitoring.

Run with:
    uvicorn app.server:app --host 0.0.0.0 --port 8000 --workers 1
"""

import io
import time
import psutil
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

from core.model import ModelRegistry
from core.logger import log_request, get_stats, read_logs


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry = ModelRegistry.get()
    registry.load()
    yield


app = FastAPI(
    title="Fleet AI — Inference API",
    description="Real-time tyre defect detection. Upload an image, get a prediction with confidence score and latency.",
    version="1.0.0",
    lifespan=lifespan
)

SERVER_START = time.time()


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    """
    System health check. Returns model status, uptime, CPU/memory usage.
    Use this endpoint to verify the server is ready before sending predictions.
    """
    registry = ModelRegistry.get()
    process  = psutil.Process()
    return {
        "status":         "healthy" if registry.loaded else "loading",
        "model_loaded":   registry.loaded,
        "model_load_ms":  registry.load_time_ms,
        "uptime_seconds": round(time.time() - SERVER_START, 1),
        "cpu_percent":    psutil.cpu_percent(interval=0.1),
        "memory_mb":      round(process.memory_info().rss / 1024 / 1024, 1),
        "timestamp":      datetime.now(timezone.utc).isoformat(),
    }


# ── Single prediction ─────────────────────────────────────────────────────────

@app.post("/predict", tags=["Inference"])
async def predict(file: UploadFile = File(...)):
    """
    Predict whether a tyre image is good or defective.

    Returns:
    - class: 'good' or 'defective'
    - confidence: model confidence as a percentage
    - latency_ms: inference time in milliseconds
    - all_probs: probability for each class
    - recommended_action: what to do with this tyre
    """
    registry = ModelRegistry.get()
    if not registry.loaded:
        raise HTTPException(status_code=503, detail="Model not ready yet")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Expected an image, got {file.content_type}")

    try:
        contents = await file.read()
        image    = Image.open(io.BytesIO(contents))
        result   = registry.predict(image)

        result["recommended_action"] = (
            "Recommend immediate physical inspection. Do not deploy vehicle until cleared."
            if result["class"] == "defective"
            else "No immediate action required. Schedule routine inspection per fleet protocol."
        )
        result["filename"] = file.filename

        log_request("/predict", result, 200)
        return JSONResponse(content=result)

    except Exception as e:
        log_request("/predict", None, 500, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ── Batch prediction ──────────────────────────────────────────────────────────

@app.post("/predict/batch", tags=["Inference"])
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Predict on multiple tyre images in a single request.
    Returns individual results plus a fleet summary.
    """
    registry = ModelRegistry.get()
    if not registry.loaded:
        raise HTTPException(status_code=503, detail="Model not ready yet")

    if len(files) > 100:
        raise HTTPException(status_code=400, detail="Max 100 images per batch request")

    results   = []
    total_t0  = time.perf_counter()

    for f in files:
        try:
            contents = await f.read()
            image    = Image.open(io.BytesIO(contents))
            result   = registry.predict(image)
            result["filename"] = f.filename
            result["status"]   = "ok"
            log_request("/predict/batch", result, 200)
        except Exception as e:
            result = {"filename": f.filename, "status": "error", "error": str(e)}
            log_request("/predict/batch", None, 500, error=str(e))
        results.append(result)

    total_ms  = round((time.perf_counter() - total_t0) * 1000, 2)
    ok        = [r for r in results if r.get("status") == "ok"]
    defective = [r for r in ok if r.get("class") == "defective"]

    return JSONResponse(content={
        "results": results,
        "summary": {
            "total":          len(results),
            "successful":     len(ok),
            "defective":      len(defective),
            "good":           len(ok) - len(defective),
            "total_ms":       total_ms,
            "avg_latency_ms": round(total_ms / len(results), 2) if results else 0,
        }
    })


# ── Metrics ───────────────────────────────────────────────────────────────────

@app.get("/metrics", tags=["Monitoring"])
def metrics():
    """
    Live server metrics — total requests, latency percentiles (avg/p95/p99),
    success rate, and prediction distribution.
    Use this to monitor server performance over time.
    """
    return JSONResponse(content=get_stats())


@app.get("/metrics/logs", tags=["Monitoring"])
def recent_logs(n: int = 50):
    """
    Return the last N request logs.
    Useful for debugging and real-time monitoring.
    """
    return JSONResponse(content={"logs": read_logs(n=n), "count": n})


# ── Model info ────────────────────────────────────────────────────────────────

@app.get("/model/info", tags=["System"])
def model_info():
    """
    Returns model architecture, training metadata, and performance metrics.
    """
    registry = ModelRegistry.get()
    if not registry.loaded or not registry.meta:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return JSONResponse(content={
        "architecture":     "MobileNetV2 (transfer learning, ImageNet pretrained)",
        "classes":          registry.classes,
        "test_accuracy":    registry.meta.get("test_accuracy"),
        "val_accuracy":     registry.meta.get("best_val_accuracy"),
        "epochs_trained":   registry.meta.get("epochs"),
        "device":           registry.meta.get("device"),
        "classification_report": registry.meta.get("classification_report"),
    })