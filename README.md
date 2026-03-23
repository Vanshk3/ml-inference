# Fleet AI — Real-Time ML Inference System

Production-style REST API for serving the Fleet AI tyre defect detection model. Built to show what happens *after* training — how you actually get a model into a state where it can handle real traffic reliably.

---

## What this is

Most ML projects stop at training. This one goes further — it wraps the trained MobileNetV2 model in a proper API server with:

- Sub-100ms inference latency on CPU
- Request logging with unique IDs and timestamps
- Live latency metrics (avg, p95, p99)
- Load testing to prove it under concurrent traffic
- A real-time monitoring dashboard
- Docker container for reproducible deployment

---

## Performance (from load test)

| Metric | Result |
|---|---|
| Avg inference latency | 15.87ms |
| p95 latency | 21.8ms |
| p99 latency | 31.44ms |
| Requests/sec (20 users) | 13–15 req/s |
| Total requests served | 375 |
| Success rate | 100% |
| Model load time | 72ms |

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Server health, uptime, CPU/memory |
| POST | `/predict` | Single tyre image → prediction + confidence |
| POST | `/predict/batch` | Multiple images → results + fleet summary |
| GET | `/metrics` | Latency stats, success rate, prediction distribution |
| GET | `/metrics/logs` | Last N request logs |
| GET | `/model/info` | Model architecture and training metadata |
| GET | `/docs` | Auto-generated Swagger UI |

---

## Quickstart

```bash
git clone https://github.com/vanshk3/ml-inference
cd ml-inference
pip install -r requirements.txt
```

Copy your trained model weights from Fleet AI:

```bash
mkdir models
cp ../fleet_ai/models/best_model.pth models/
cp ../fleet_ai/models/metadata.json models/
```

Start the server:

```bash
uvicorn app.server:app --host 0.0.0.0 --port 8000
```

Open Swagger docs at: `http://localhost:8000/docs`

---

## Run the monitoring dashboard

```bash
streamlit run monitoring/dashboard.py
```

Opens at `localhost:8501` — shows live latency, prediction distribution, and request logs.

---

## Load testing

```bash
locust -f tests/load_test.py --host=http://localhost:8000 \
       --users 20 --spawn-rate 5 --run-time 30s --headless
```

Or open the Locust UI at `localhost:8089` for interactive control.

---

## Docker

```bash
docker build -t fleet-ai-inference .
docker run -p 8000:8000 -v $(pwd)/models:/app/models fleet-ai-inference
```

---

## Project structure

```
ml_inference/
├── app/
│   └── server.py          FastAPI server — all endpoints
├── core/
│   ├── model.py           model loading, singleton registry, inference
│   └── logger.py          request logging, stats aggregation
├── monitoring/
│   └── dashboard.py       Streamlit real-time dashboard
├── tests/
│   └── load_test.py       Locust load testing
├── models/                put best_model.pth and metadata.json here
├── logs/                  auto-created, stores requests.jsonl
├── Dockerfile
└── requirements.txt
```

---

## Tech stack

Python · FastAPI · PyTorch · Uvicorn · Locust · Streamlit · Docker · psutil

---

Built a production-style ML inference API serving the Fleet AI tyre defect model —
FastAPI + Uvicorn, model loads in 72ms; achieved 13–15 req/s at avg 15.87ms latency
(p99 31ms) under 20 concurrent users with 100% success rate across 375 requests;
includes real-time monitoring dashboard, request logging, and Docker containerisation.

Built by [Vansh](https://linkedin.com/in/vanshk3) — MSc Data Science, University of Bath
Part of the [Fleet AI](https://github.com/vanshk3/fleet-ai) project.