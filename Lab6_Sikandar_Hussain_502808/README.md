# Lab 6 — Model Deployment with Flask REST API

**Student:** Sikandar Hussain  
**Roll No:** 502808

---

## Overview

This lab deploys two pre-trained scikit-learn pipelines as a lightweight Flask REST API:

| Endpoint | Pipeline | Task |
|---|---|---|
| `POST /predict` | `titanic_pipeline.joblib` | Binary classification (survived / did not survive) |
| `POST /predict_price` | `housing_pipeline.joblib` | Regression (California median house value) |
| `GET /health` | — | Health / readiness check |

Both models are loaded **once at startup** — no retraining occurs during serving.

---

## Project Structure

```
Lab6_Sikandar_Hussain_502808/
├── app.py                    # Flask REST API
├── test_client.py            # Automated test script
├── titanic_pipeline.joblib   # Saved classification pipeline
├── housing_pipeline.joblib   # Saved regression pipeline
└── README.md                 # This file
```

---

## Setup & Run

```bash
# Install dependencies (from project root)
pip install flask joblib scikit-learn pandas numpy requests

# Start the API server
cd Lab6_Sikandar_Hussain_502808
python app.py
```

The server starts on **http://localhost:5000**.

---

## API Reference

### `GET /health`

```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "OK",
  "titanic_model_loaded": true,
  "housing_model_loaded": true
}
```

### `POST /predict` — Titanic Survival

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"pclass":3,"sex":"male","age":22,"sibsp":1,"parch":0,"fare":7.25,"embarked":"S"}'
```

Response:
```json
{"prediction": 0}
```

Optional parameters:
- `"threshold": 0.8` — custom classification threshold (default 0.5)
- `"return_proba": true` — include class probabilities
- `"X-API-Key"` header — optional API key validation

### `POST /predict_price` — Housing Price

```bash
curl -X POST http://localhost:5000/predict_price \
  -H "Content-Type: application/json" \
  -d '{"MedInc":8.3252,"HouseAge":41.0,"AveRooms":6.984,"AveBedrms":1.024,"Population":322.0,"AveOccup":2.556,"Latitude":37.88,"Longitude":-122.23}'
```

Response:
```json
{"predicted_price": 4.1518}
```

---

## Error Handling

| Scenario | Status Code |
|---|---|
| Missing required fields | 400 |
| Invalid JSON body | 400 |
| Numeric field receives string | 400 |
| Invalid API key | 401 |
| Internal prediction error | 500 |

Extra / unexpected fields in the JSON body are silently ignored.

---

## Testing

With the server running in one terminal:

```bash
python test_client.py
```

The test client covers:
1. Health check
2. Valid classification input
3. Missing feature → 400
4. Wrong data type → 400
5. Invalid JSON → 400
6. Extra fields (ignored)
7. Probability output (advanced)
8. Custom threshold (advanced)
9. Valid regression input
10. Missing housing field → 400
11. Wrong housing type → 400
12. Invalid API key → 401

---

## Robustness Features (Part 4)

- **Input length constraint** — only recognised fields are forwarded to the model; unexpected keys are dropped.
- **Type safety** — numeric fields are validated before prediction; strings in numeric slots return 400.
- **Logging** — every request logs the input received and the prediction returned to stdout.

---

## Analytical Reflection (Part 5)

1. **Why must preprocessing be serialized with the model?**  
   The model expects data in the exact transformed representation it was trained on (imputed, scaled, encoded). If preprocessing is not serialized together, we must manually reproduce every step identically, which is error-prone and brittle.

2. **What would happen if we saved only model weights?**  
   We would lose the fitted scaler means/stds, imputer fill values, and encoder category mappings. Predictions on raw input would be meaningless because the feature space would differ from what the model learned.

3. **Why load the model at startup instead of inside the endpoint?**  
   Loading from disk is slow (file I/O + deserialization). Doing it per-request adds latency and wastes memory by creating duplicate objects. Loading once keeps the model cached in memory for instant reuse.

4. **Is `/predict` idempotent here?**  
   Yes — the same input always produces the same output and causes no side effects (no database writes, no state mutation). It is a pure function behind an HTTP interface.

5. **What breaks if 100 users send requests simultaneously?**  
   Flask's default development server is single-threaded, so requests are queued and latency spikes. Under heavy load, timeouts and dropped connections can occur. The global model object is read-only so there is no data-race, but throughput is the bottleneck.

6. **What would you change to make this production-ready?**  
   - Use a production WSGI server (Gunicorn / uWSGI) with multiple workers.  
   - Add structured logging, request-ID tracing, and monitoring (Prometheus metrics).  
   - Containerize with Docker and deploy behind a reverse proxy (Nginx).  
   - Add authentication, rate limiting, and input size limits.  
   - Implement model versioning and A/B testing support.  
   - Add health/readiness probes for Kubernetes orchestration.
