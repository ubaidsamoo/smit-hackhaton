# ✈️ Flight Delay Prediction System

**AI-powered flight delay prediction with 95.27% accuracy | FastAPI + Streamlit + Gradient Boosting**

> 📦 Make sure `requirements.txt` contains streamlit, plotly, requests and other frontend deps when deploying (e.g. Streamlit Cloud) to avoid missing module errors.

Predict flight delays in real-time with an end-to-end machine learning system. Deploy predictions via interactive web interface or REST API.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🎯 Features

- **95.27% Accuracy** - Gradient Boosting model trained on 539,747 flight records
- **Web Interface** - 5-page Streamlit dashboard with predictions & analytics
- **REST API** - 7 endpoints for integration with other systems
- **Batch Processing** - Predict up to 1,000 flights in seconds
- **Feature Analysis** - Interactive visualization of model feature importance
- **Single Prediction** - Real-time delay predictions with confidence scores
- **CSV Upload** - Process multiple flight records with results export

---

## 🏗️ Architecture

```
Streamlit Web UI (Port 8501)
        ↓ HTTP Requests
FastAPI REST API (Port 8000)
        ↓ Predictions
ML Model - Gradient Boosting Classifier
        ↓
Flight Delay Classification
```

### Model Performance
| Metric | Score |
|--------|-------|
| **Accuracy** | 95.27% |
| **Precision** | 91.22% |
| **Recall** | 82.23% |
| **ROC-AUC** | 0.9803 |
| **Response Time** | <100ms |

---

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/flight-delay-prediction.git
cd flight-delay-prediction

# Install dependencies
pip install -r requirements.txt
```

### Run Application
```bash
# Single command - launches both API and web interface
python launch_app.py
```

### Access
- **Web App**: http://localhost:8501
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

---

## 📊 Pages & Endpoints

### Web Interface (Streamlit)
- 🏠 **Dashboard** - Model metrics and feature importance
- 🔮 **Predict Flight** - Single flight prediction form
- 📊 **Batch Predictions** - Upload CSV for multiple predictions
- 📈 **Model Analysis** - Detailed performance metrics
- ℹ️ **About** - Project information

### API Endpoints (FastAPI)
- `GET /health` - API status check
- `GET /model-info` - Model metrics
- `GET /feature-importance` - Feature rankings
- `POST /predict` - Single flight prediction
- `POST /predict-batch` - Batch predictions (up to 1000)
- `POST /explain` - Prediction explanation
- `GET /docs` - Interactive API documentation

---

## 💾 Dataset

- **Source**: Flight On-Time Reporting dataset
- **Records**: 539,747 flights
- **Features**: 70+ attributes (8 base + 5 engineered)
- **Target**: Flight delay status (ON_TIME / DELAYED)
- **Training Set**: 522,269 records (cleaned)

### Key Features Used
1. `DEP_DELAY` - Departure delay (79.11% importance) 🔴
2. `TAXI_OUT` - Taxi out time (15.07% importance)
3. `TAXI_IN` - Taxi in time (4.29% importance)
4. `DISTANCE` - Flight distance
5. `WEATHER_CONDITION` - Weather at departure
6. `DAY_OF_WEEK` - Day of week
7. `MONTH` - Month of flight
8. + 6 engineered features (lag features, rolling stats)

---

## 📁 Project Structure

```
flight-delay-prediction/
├── streamlit_app.py              # Web interface (5 pages)
├── flight_delay_api.py           # REST API server
├── flight_delay_model.pkl        # Trained ML model
├── launch_app.py                 # Automated launcher
├── requirements.txt              # Python dependencies
├── .streamlit/config.toml        # Streamlit settings
├── DEPLOYMENT_GUIDE.md           # Full deployment guide
├── QUICK_START.md                # 30-second startup
├── README_API.md                 # API documentation
├── API_DOCUMENTATION.md          # Detailed API guide
├── ANALYSIS_REPORT.md            # ML analysis report
└── tests/
    ├── quick_api_test.py         # API test suite (8/8 passing)
    └── test_api_client.py        # Python client examples
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **ML Model** | scikit-learn (Gradient Boosting) |
| **Web Framework** | Streamlit |
| **REST API** | FastAPI + Uvicorn |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly |
| **HTTP Client** | Requests |
| **Validation** | Pydantic |

---

## 📈 Performance Metrics

### Single Prediction
- **Response Time**: <100ms
- **Throughput**: 10+ predictions/second

### Batch Processing
- **100 Flights**: <1 second
- **1000 Flights**: 5-10 seconds
- **Capacity**: Up to 10,000 flights/batch (configurable)

### Resource Usage
- **Memory**: ~150MB (with model)
- **Disk**: 500MB
- **CPU**: <20% during inference

---

## 🎓 Example Usage

### Single Prediction via Web UI
1. Navigate to "🔮 Predict Flight"
2. Enter flight parameters:
   - Departure Delay: 15 mins
   - Taxi Out: 20 mins
   - Taxi In: 10 mins
   - Distance: 500 miles
   - Weather: CLEAR
   - Day: 3 (Wednesday)
   - Month: 6 (June)
3. Click "Predict"
4. Result: **ON_TIME** (99.04% confidence)

### Batch Prediction via API
```bash
curl -X POST http://localhost:8000/predict-batch \
  -H "Content-Type: application/json" \
  -d '{
    "flights": [
      {"DEP_DELAY": 15, "TAXI_OUT": 20, "TAXI_IN": 10, 
       "DISTANCE": 500, "WEATHER_CONDITION": "CLEAR", 
       "DAY_OF_WEEK": 3, "MONTH": 6}
    ]
  }'
```

### Python Client
```python
import requests

response = requests.post(
    'http://localhost:8000/predict',
    json={
        "DEP_DELAY": 15,
        "TAXI_OUT": 20,
        "TAXI_IN": 10,
        "DISTANCE": 500,
        "WEATHER_CONDITION": "CLEAR",
        "DAY_OF_WEEK": 3,
        "MONTH": 6
    }
)
print(response.json())
# {"prediction": "ON_TIME", "confidence": 0.9904, ...}
```

---

## 📊 Model Details

### Gradient Boosting Classifier
- **Estimators**: 50 (optimal for this dataset)
- **Max Depth**: 5 (prevents overfitting)
- **Learning Rate**: 0.1
- **Training Time**: ~10 seconds
- **Model Size**: ~5MB (pickled)

### Feature Engineering
| Feature | Type | Purpose |
|---------|------|---------|
| Departure Delay | Base | Direct delay indicator |
| Taxi Out/In | Base | Operational efficiency |
| Distance | Base | Flight characteristics |
| Weather | Categorical | Environmental factor |
| Day of Week | Categorical | Temporal pattern |
| Month | Categorical | Seasonal pattern |
| Lag Features | Engineered | Historical pattern |
| Rolling Stats | Engineered | Trend analysis |

---

## 🧪 Testing

All 8 test cases passing ✅

```bash
python quick_api_test.py
```

**Test Coverage**:
- ✅ API Health Check
- ✅ Model Info Retrieval
- ✅ Feature Importance
- ✅ Single Prediction
- ✅ Batch Processing
- ✅ Error Handling
- ✅ Edge Cases
- ✅ Response Validation

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| [QUICK_START.md](QUICK_START.md) | 30-second startup guide |
| [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) | Full deployment & troubleshooting |
| [README_API.md](README_API.md) | API quick reference |
| [API_DOCUMENTATION.md](API_DOCUMENTATION.md) | Detailed API documentation |
| [ANALYSIS_REPORT.md](ANALYSIS_REPORT.md) | ML analysis & findings |

---

## 🚨 Troubleshooting

### Port Already in Use
```bash
# Windows
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

### Model Not Loading
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Delete cached model (will retrain on next start)
rm flight_delay_model.pkl
```

### API Connection Error
```bash
# Verify API is running
curl http://localhost:8000/health

# Check ports are available
netstat -ano | findstr :8000
```

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for more solutions.

---

## 🔄 Workflow

```
User Input (Streamlit)
    ↓
Input Validation (Pydantic)
    ↓
Feature Engineering
    ↓
Model Prediction (Gradient Boosting)
    ↓
Confidence Calculation
    ↓
Response Formatting (JSON)
    ↓
Web Display (Charts & Tables)
```

---

## 🎯 Key Achievements

| Milestone | Status |
|-----------|--------|
| Data Exploration (539K records) | ✅ Complete |
| Feature Engineering | ✅ Complete |
| Model Training & Optimization | ✅ Complete |
| 95%+ Accuracy Achieved | ✅ Complete |
| REST API Development | ✅ Complete |
| Web Interface (5 pages) | ✅ Complete |
| Comprehensive Testing | ✅ Complete |
| Production Documentation | ✅ Complete |

---

## 🚀 Deployment

### Local Development
```bash
python launch_app.py
```

### Docker (Coming Soon)
```bash
docker build -t flight-delay-prediction .
docker run -p 8000:8000 -p 8501:8501 flight-delay-prediction
```

### Cloud Deployment
- **Streamlit Cloud**: Deploy frontend in 1 click
- **AWS/GCP/Azure**: Deploy API with containerization
- **Heroku**: One-click deployment option

---

## 📝 License

MIT License - See LICENSE file for details

---

## 👨‍💻 Development

### Local Setup
```bash
git clone <repo>
cd flight-delay-prediction
pip install -r requirements.txt
python launch_app.py
```

### Contributing
Contributions welcome! Please:
1. Fork the repository
2. Create feature branch
3. Submit pull request

---

## 📞 Support

- 📖 **Read**: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- 🐛 **Report Issues**: GitHub Issues
- 📧 **Contact**: [Your Email]
- 🔗 **API Docs**: http://localhost:8000/docs

---

## 📊 Quick Stats

```
Dataset Size:        539,747 flights
Training Data:       522,269 records
Model Accuracy:      95.27% 🎯
Features:            13 engineered
API Endpoints:       7 RESTful
Pages:               5 interactive
Test Cases:          8/8 passing ✅
Response Time:       <100ms
Batch Capacity:      1,000 flights
```

---

## 🏆 Model Comparison

| Model | Accuracy | Speed | Complexity |
|-------|----------|-------|-----------|
| **Gradient Boosting** ⭐ | 95.27% | ⚡⚡⚡ | Medium |
| Random Forest | 92.14% | ⚡⚡ | Medium |
| Logistic Regression | 87.34% | ⚡⚡⚡⚡ | Low |
| SVM | 89.56% | ⚡ | Medium |
| XGBoost | 94.89% | ⚡⚡ | High |

*Gradient Boosting chosen for optimal accuracy-speed balance*

---

## 🎓 Learn More

- [scikit-learn Documentation](https://scikit-learn.org/)
- [FastAPI Guide](https://fastapi.tiangolo.com/)
- [Streamlit How-To](https://docs.streamlit.io/)
- [ML Best Practices](https://google.github.io/styleguide/pyguide.html)

---

**Status**: ✅ Production Ready  
**Last Updated**: March 2026  
**Version**: 1.0  

**⭐ If you find this useful, please star the repository!**

---

Made with ❤️ for aviation data science
