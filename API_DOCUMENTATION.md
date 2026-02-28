# Flight Delay Prediction API Documentation

## Overview

A production-ready FastAPI application for predicting flight arrival delays using a Gradient Boosting machine learning model. The API achieves **95.27% accuracy** and provides real-time predictions with confidence scores.

---

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install manually
pip install fastapi uvicorn pydantic pandas numpy scikit-learn
```

### Running the API

```bash
# Start the API server
python flight_delay_api.py

# API will be available at:
# - Main: http://localhost:8000
# - Swagger UI: http://localhost:8000/docs
# - ReDoc: http://localhost:8000/redoc
```

### Running the Test Client

```bash
# In a separate terminal, run the test client
python test_api_client.py
```

---

## API Endpoints

### 1. **Root Endpoint**
Get basic API information and available endpoints.

```
GET /
```

**Response:**
```json
{
  "message": "Flight Delay Prediction API",
  "version": "1.0.0",
  "status": "active",
  "endpoints": {
    "predict_single": "/predict",
    "predict_batch": "/predict-batch",
    "model_info": "/model-info",
    "health": "/health",
    "docs": "/docs"
  }
}
```

---

### 2. **Health Check**
Verify API is running and model is loaded.

```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2026-03-01T12:00:00"
}
```

---

### 3. **Model Information**
Get detailed model performance metrics and features.

```
GET /model-info
```

**Response:**
```json
{
  "model_name": "Flight Delay Prediction",
  "model_type": "Gradient Boosting Classifier",
  "accuracy": "95.27%",
  "precision": "91.22%",
  "recall": "82.23%",
  "f1_score": "0.8649",
  "roc_auc": "0.9803",
  "features_count": 13,
  "features": [
    "DAY_OF_MONTH",
    "DAY_OF_WEEK",
    "OP_CARRIER_AIRLINE_ID",
    "DEST_AIRPORT_ID",
    "DISTANCE",
    "DEP_DELAY",
    "TAXI_OUT",
    "TAXI_IN",
    "IS_WEEKEND",
    "IS_MONTH_START",
    "IS_MONTH_END",
    "LONG_DISTANCE",
    "HIGH_DEP_DELAY"
  ],
  "training_date": "2026-03-01",
  "status": "Production Ready"
}
```

---

### 4. **Feature Importance**
Get feature importance scores (which features matter most).

```
GET /feature-importance
```

**Response:**
```json
{
  "features": [
    {
      "feature": "DEP_DELAY",
      "importance": 0.7838
    },
    {
      "feature": "TAXI_OUT",
      "importance": 0.1533
    },
    ...
  ],
  "top_5": [
    {
      "feature": "DEP_DELAY",
      "importance": 0.7838
    },
    ...
  ]
}
```

---

### 5. **Single Flight Prediction** ⭐ MAIN ENDPOINT
Predict delay for a single flight.

```
POST /predict
```

**Request Body:**
```json
{
  "day_of_month": 15,
  "day_of_week": 3,
  "op_carrier_airline_id": 19977,
  "dest_airport_id": 12892,
  "distance": 750.0,
  "dep_delay": 5.0,
  "taxi_out": 15.0,
  "taxi_in": 8.0
}
```

**Response:**
```json
{
  "prediction": 0,
  "probability_on_time": 0.9485,
  "probability_delayed": 0.0515,
  "confidence": 0.9485,
  "status": "ON_TIME"
}
```

**Field Descriptions:**
- `prediction`: 0 = On-time, 1 = Delayed (>15 minutes)
- `probability_on_time`: Confidence flight arrives on-time (0-1)
- `probability_delayed`: Confidence flight is delayed (0-1)
- `confidence`: Maximum probability value (0-1)
- `status`: Human-readable prediction (ON_TIME/DELAYED)

---

### 6. **Batch Predictions**
Predict delays for multiple flights (up to 1000 per request).

```
POST /predict-batch
```

**Request Body:**
```json
{
  "flights": [
    {
      "day_of_month": 10,
      "day_of_week": 2,
      "op_carrier_airline_id": 19977,
      "dest_airport_id": 12892,
      "distance": 600.0,
      "dep_delay": 5.0,
      "taxi_out": 14.0,
      "taxi_in": 7.0
    },
    {
      "day_of_month": 22,
      "day_of_week": 6,
      "op_carrier_airline_id": 20363,
      "dest_airport_id": 11433,
      "distance": 1200.0,
      "dep_delay": 30.0,
      "taxi_out": 20.0,
      "taxi_in": 10.0
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "prediction": 0,
      "probability_on_time": 0.9485,
      "probability_delayed": 0.0515,
      "confidence": 0.9485,
      "status": "ON_TIME"
    },
    {
      "prediction": 1,
      "probability_on_time": 0.1245,
      "probability_delayed": 0.8755,
      "confidence": 0.8755,
      "status": "DELAYED"
    }
  ],
  "total_flights": 2,
  "delayed_count": 1,
  "on_time_count": 1,
  "delayed_percentage": 50.0
}
```

---

### 7. **Detailed Explanation**
Get detailed explanation of a prediction including feature interpretation.

```
POST /explain
```

**Request Body:**
```json
{
  "day_of_month": 25,
  "day_of_week": 4,
  "op_carrier_airline_id": 19977,
  "dest_airport_id": 12892,
  "distance": 1500.0,
  "dep_delay": 50.0,
  "taxi_out": 28.0,
  "taxi_in": 11.0
}
```

**Response:**
```json
{
  "flight_data": {
    "day_of_month": 25,
    "day_of_week": 4,
    "op_carrier_airline_id": 19977,
    "dest_airport_id": 12892,
    "distance": 1500.0,
    "dep_delay": 50.0,
    "taxi_out": 28.0,
    "taxi_in": 11.0
  },
  "prediction": 1,
  "status": "DELAYED",
  "probability_on_time": 0.0234,
  "probability_delayed": 0.9766,
  "confidence": 0.9766,
  "engineered_features": {
    "IS_WEEKEND": 0,
    "IS_MONTH_START": 0,
    "IS_MONTH_END": 1,
    "LONG_DISTANCE": 1,
    "HIGH_DEP_DELAY": 1
  },
  "interpretation": "This flight is predicted to be delayed with very high confidence (97%). The flight has significant departure delay which strongly predicts arrival delay."
}
```

---

## Input Parameters Reference

### Flight Data Fields

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `day_of_month` | int | 1-31 | Day of the month |
| `day_of_week` | int | 1-7 | Day of week (1=Monday, 7=Sunday) |
| `op_carrier_airline_id` | int | Any | Airline identifier code |
| `dest_airport_id` | int | Any | Destination airport code |
| `distance` | float | 0+ | Flight distance in miles |
| `dep_delay` | float | Any | Departure delay in minutes (-ve = early) |
| `taxi_out` | float | 0+ | Ground movement time at origin (min) |
| `taxi_in` | float | 0+ | Ground movement time at destination (min) |

---

## Example Use Cases

### Use Case 1: Real-time Prediction
Predict delay for a flight departing in the next hour.

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "day_of_month": 1,
    "day_of_week": 3,
    "op_carrier_airline_id": 19977,
    "dest_airport_id": 12892,
    "distance": 750,
    "dep_delay": 8,
    "taxi_out": 15,
    "taxi_in": 8
  }'
```

### Use Case 2: Daily Operations Dashboard
Get predictions for all morning flights.

```bash
curl -X POST "http://localhost:8000/predict-batch" \
  -H "Content-Type: application/json" \
  -d '{
    "flights": [
      {...},
      {...},
      ...
    ]
  }'
```

### Use Case 3: Passenger Communication
Generate explanation for customer notification.

```bash
curl -X POST "http://localhost:8000/explain" \
  -H "Content-Type: application/json" \
  -d '{...}'
```

---

## Model Performance

### Overall Metrics
- **Accuracy**: 95.27% (Correct predictions out of all predictions)
- **Precision**: 91.22% (Predicted delays that were actually delayed)
- **Recall**: 82.23% (Actual delays that were predicted)
- **F1-Score**: 0.8649 (Balanced accuracy metric)
- **ROC-AUC**: 0.9803 (Model discrimination ability)

### Test Set Performance
- **Total Test Flights**: 30,000
- **Correctly Predicted**: 28,582 (95.27%)
- **Incorrectly Predicted**: 1,418 (4.73%)

### Feature Importance
1. **DEP_DELAY** (78.38%) - Departure delay is the dominant predictor
2. **TAXI_OUT** (15.33%) - Ground movement at origin
3. **TAXI_IN** (4.19%) - Ground movement at destination
4. **DISTANCE** (0.83%) - Flight distance
5. **OP_CARRIER_AIRLINE_ID** (0.64%) - Airline characteristics

---

## Error Handling

### Common Error Responses

**400 Bad Request** - Invalid input
```json
{
  "detail": "Prediction error: [error details]"
}
```

**400 Bad Request** - No flights in batch
```json
{
  "detail": "No flights provided"
}
```

**400 Bad Request** - Batch too large
```json
{
  "detail": "Maximum 1000 flights per request"
}
```

**500 Internal Server Error** - Server error
```json
{
  "error": "An error occurred",
  "detail": "[error details]",
  "timestamp": "2026-03-01T12:00:00"
}
```

---

## Deployment

### Production Deployment with Gunicorn

```bash
# Install gunicorn
pip install gunicorn

# Run with multiple workers
gunicorn flight_delay_api:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

### Docker Deployment

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY flight_delay_api.py .
COPY ONTIME_REPORTING.csv .
COPY *.pkl .
COPY *.json .

EXPOSE 8000

CMD ["uvicorn", "flight_delay_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build and run:**
```bash
docker build -t flight-delay-api .
docker run -p 8000:8000 flight-delay-api
```

### Cloud Deployment

#### AWS (API Gateway + Lambda)
- Package as Lambda layer
- Use API Gateway for HTTP endpoints
- Store model in S3

#### Azure (App Service + Functions)
- Deploy as Azure Web App
- Use Azure Functions for serverless deployment
- Store model in Azure Blob Storage

#### Google Cloud (Cloud Run)
```bash
gcloud run deploy flight-delay-api \
  --source . \
  --platform managed \
  --region us-central1
```

---

## Performance Considerations

### Latency
- **Single Prediction**: <1ms
- **100 Predictions**: <100ms
- **1000 Predictions**: <1s

### Throughput
- Single instance can handle 100+ requests/second
- Linear scaling with additional workers/instances

### Memory Usage
- Model: ~50MB
- API + Dependencies: ~200MB
- Total per instance: ~300MB

---

## Integration Examples

### Python Client
```python
import requests

def predict_flight_delay(flight_data):
    response = requests.post(
        "http://localhost:8000/predict",
        json=flight_data
    )
    return response.json()

result = predict_flight_delay({
    "day_of_month": 15,
    "day_of_week": 3,
    "op_carrier_airline_id": 19977,
    "dest_airport_id": 12892,
    "distance": 750,
    "dep_delay": 5,
    "taxi_out": 15,
    "taxi_in": 8
})

print(f"Status: {result['status']}")
print(f"Confidence: {result['confidence']:.1%}")
```

### JavaScript Client
```javascript
async function predictFlightDelay(flightData) {
  const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(flightData)
  });
  return response.json();
}

const result = await predictFlightDelay({
  day_of_month: 15,
  day_of_week: 3,
  op_carrier_airline_id: 19977,
  dest_airport_id: 12892,
  distance: 750,
  dep_delay: 5,
  taxi_out: 15,
  taxi_in: 8
});

console.log(`Status: ${result.status}`);
console.log(`Confidence: ${(result.confidence * 100).toFixed(1)}%`);
```

---

## Support & Troubleshooting

### API Won't Start
1. Check port 8000 is available
2. Verify all dependencies installed
3. Check ONTIME_REPORTING.csv exists

### Model Not Loading
1. Ensure model files exist (*.pkl, *.json)
2. Check file permissions
3. Verify scikit-learn version compatibility

### High Latency
1. Reduce batch size
2. Add more API workers
3. Check server resources

---

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Uvicorn Documentation](https://www.uvicorn.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pydantic Documentation](https://docs.pydantic.dev/)

---

**Status**: Production Ready ✓
**Version**: 1.0.0
**Last Updated**: 2026-03-01
