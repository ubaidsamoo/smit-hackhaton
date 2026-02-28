# Flight Delay Prediction - Complete Deployment Guide

## 📋 Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Starting the Application](#starting-the-application)
4. [Accessing the Web Interface](#accessing-the-web-interface)
5. [Using the Application](#using-the-application)
6. [Troubleshooting](#troubleshooting)
7. [Architecture Overview](#architecture-overview)
8. [Project Files Summary](#project-files-summary)

---

## System Requirements

### Hardware
- **CPU**: Dual-core or better
- **RAM**: Minimum 2GB (4GB+ recommended)
- **Disk**: 500MB free space
- **Network**: Internet connection for initial setup

### Software
- **Python**: 3.8 or later (3.11+ recommended)
- **OS**: Windows, macOS, or Linux
- **Ports**: 8000 (API), 8501 (Streamlit) must be available

### Browser
- Chrome, Firefox, Safari, Edge (any modern browser)
- JavaScript enabled
- Cookies enabled for session management

---

## Installation

### Step 1: Verify Python Environment
```bash
# Check Python version
python --version
# Expected: Python 3.8.0 or later

# Check pip is available
pip --version
```

### Step 2: Navigate to Project Directory
```bash
cd D:\Python\Hackathon
# or your custom installation path
```

### Step 3: Verify All Required Files
Check that these files exist:
- ✓ `flight_delay_api.py` - FastAPI server
- ✓ `streamlit_app.py` - Streamlit web app
- ✓ `flight_delay_model.pkl` - Trained ML model (auto-generated if missing)
- ✓ `launch_app.py` - Automated launcher

> **Note for Streamlit Cloud deployments:** Streamlit Cloud installs dependencies from `requirements.txt`. Make sure the file includes all frontend packages (streamlit, plotly, streamlit-option-menu, requests) – otherwise the app will crash with `ModuleNotFoundError`.

### Step 4: Check Dependencies
```bash
# List installed packages
pip list | find "streamlit"
pip list | find "fastapi"
pip list | find "plotly"

# If missing, install them
pip install streamlit fastapi uvicorn plotly pandas requests scikit-learn
```

---

## Starting the Application

### ⭐ Recommended: Automated Launch (One Command)

#### Option A: Using Python Launcher
```bash
python launch_app.py
```
✅ Automatically starts both API and Streamlit
✅ Handles port binding and process management
✅ Cleaner shutdown with Ctrl+C

#### Option B: Using Batch File (Windows Only)
```bash
run_app.bat
```
✅ Double-click to start
✅ Opens separate command windows for API and Streamlit

---

### Manual Launch (For Debugging)

#### Terminal 1: Start FastAPI Server
```bash
python flight_delay_api.py
```

Expected output:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

#### Terminal 2: Start Streamlit
```bash
streamlit run streamlit_app.py
```

Expected output:
```
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
```

---

## Accessing the Web Interface

### Initial Access
1. **Open your browser**
2. **Go to**: http://localhost:8501
3. **Wait**: ~3-5 seconds for dashboard to load

### Port Verification
- **Streamlit Web App**: http://localhost:8501
- **API Server**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs (Swagger UI)
- **API Alternative Docs**: http://localhost:8000/redoc (ReDoc)

### Quick Health Check
```bash
# Test API is running
curl http://localhost:8000/health

# Expected response:
{"status": "healthy", "model_loaded": true, "api_version": "1.0.0"}
```

---

## Using the Application

### Dashboard (Home Page)
1. **Displays**: Model metrics and feature importance
2. **Shows**: Accuracy (95.27%), Precision, Recall, ROC-AUC
3. **Contains**: Interactive feature importance bar chart
4. **Updates**: Every time you load the page

### Single Flight Prediction
1. **Navigate**: Click "🔮 Predict Flight" in sidebar
2. **Fill Form**: Enter flight parameters:
   - Departure Delay (minutes): e.g., 15
   - Taxi Out Time (minutes): e.g., 20
   - Taxi In Time (minutes): e.g., 10
   - Distance (miles): e.g., 500
   - Weather Condition: Select from dropdown
   - Day of Week: Select from dropdown
   - Month: Select from dropdown
3. **Click**: "Predict" button
4. **Result**: Shows prediction (ON_TIME or DELAYED) with confidence

### Batch Predictions (Multiple Flights)
1. **Navigate**: Click "📊 Batch Predictions"
2. **Upload CSV**: Click "Browse files" and select CSV
   - Or drag-and-drop CSV file
3. **Format Required**: 
   ```csv
   DEP_DELAY,TAXI_OUT,TAXI_IN,DISTANCE,WEATHER_CONDITION,DAY_OF_WEEK,MONTH
   15,20,10,500,CLEAR,3,6
   25,25,12,750,CLOUDY,5,9
   ```
4. **View Results**: Table shows all predictions
5. **Download**: Click "Download Results CSV" to export

### Model Analysis
1. **Navigate**: Click "📈 Model Analysis"
2. **View**: Detailed performance metrics
3. **See**: Feature importance rankings (all 13 features)
4. **Review**: Model insights and behavior analysis

### About Page
1. **Navigate**: Click "ℹ️ About"
2. **Find**: Project overview and objectives
3. **See**: Technology stack used
4. **Access**: Links to API documentation

---

## Troubleshooting

### Problem: "Connection refused" Error
**Cause**: API server is not running
**Solution**:
```bash
# Start API in separate terminal
python flight_delay_api.py

# Wait for message: "Application startup complete"
```

### Problem: Port 8501 Already in Use
**Cause**: Another Streamlit instance is running
**Solution**:
```bash
# Kill existing process
# Windows:
netstat -ano | findstr :8501
taskkill /PID <PID> /F

# macOS/Linux:
lsof -i :8501
kill -9 <PID>

# Or start Streamlit on different port:
streamlit run streamlit_app.py --server.port 8502
```

### Problem: Port 8000 (API) Already in Use
**Cause**: Another API instance or service using port
**Solution**:
```bash
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# macOS/Linux:
lsof -i :8000
kill -9 <PID>

# Or edit flight_delay_api.py to use different port:
# Change: uvicorn.run(app, host="127.0.0.1", port=8000)
# To:     uvicorn.run(app, host="127.0.0.1", port=8001)
```

### Problem: "Module not found" Error
**Cause**: Required packages not installed
**Solution**:
```bash
# Reinstall all dependencies
pip install streamlit fastapi uvicorn plotly pandas requests scikit-learn pydantic

# Or use requirements file
pip install -r requirements.txt
```

### Problem: Model File Not Found (flight_delay_model.pkl)
**Cause**: Model not trained yet
**Solution**:
- First start of API will train model (~30 seconds)
- Wait for message: "Model training completed"
- No action needed - automatic

### Problem: CSV Upload Fails
**Cause**: Wrong CSV format or headers
**Solution**:
```csv
# Verify CSV has these exact column names:
DEP_DELAY,TAXI_OUT,TAXI_IN,DISTANCE,WEATHER_CONDITION,DAY_OF_WEEK,MONTH

# Check data types:
# - Numeric columns: DEP_DELAY, TAXI_OUT, TAXI_IN, DISTANCE (numbers)
# - String columns: WEATHER_CONDITION (text), DAY_OF_WEEK (0-6), MONTH (1-12)

# Example valid CSV:
15,20,10,500,CLEAR,3,6
-5,15,8,300,RAINY,1,12
```

### Problem: Predictions are Slow
**Cause**: API overloaded or network issues
**Solution**:
```bash
# Check API logs for errors
# Monitor system resources (CPU, RAM, Network)
# Try with fewer rows in batch
# Restart API: stop (Ctrl+C) and start again
```

### Problem: Blank or White Screen
**Cause**: JavaScript error or outdated browser
**Solution**:
```bash
# Try these steps in order:
1. Hard refresh: Ctrl+Shift+F5 (Windows) or Cmd+Shift+R (Mac)
2. Clear browser cache: Settings → Clear browsing data
3. Use incognito/private window
4. Try different browser
5. Check browser console: F12 → Console tab for errors
```

### Problem: Graphs Not Rendering
**Cause**: Plotly library issue or slow connection
**Solution**:
```bash
# Update Plotly
pip install --upgrade plotly

# Restart Streamlit
# Streamlit page will auto-refresh
```

---

## Architecture Overview

### Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────┐
│               STREAMLIT LAYER (Port 8501)               │
│  Dashboard | Predict | Batch | Analysis | About        │
│  • Page routing                                         │
│  • Form inputs                                          │
│  • CSV upload                                           │
│  • Visualization & charts                               │
│  • Results display                                      │
└──────────┬────────────────────────────────┬─────────────┘
           │        HTTP Requests (Requests) │
           ├────────────────────────────────┤
           │                                 │
           ▼                                 ▼
┌─────────────────────────────────────────────────────────┐
│             FASTAPI LAYER (Port 8000)                   │
│  RESTful API with 7 endpoints                           │
│  • /health - API status                                 │
│  • /model-info - Model metrics                          │
│  • /feature-importance - Feature rankings               │
│  • /predict - Single prediction                         │
│  • /predict-batch - Batch predictions                   │
│  • /explain - Prediction explanation                    │
│  • /docs - Swagger UI documentation                     │
└──────────┬────────────────────────────────┬─────────────┘
           │     Prediction Calls (Pickle)  │
           ├────────────────────────────────┤
           │                                 │
           ▼                                 ▼
┌─────────────────────────────────────────────────────────┐
│           ML MODEL LAYER (In-Memory)                    │
│  Gradient Boosting Classifier                           │
│  • 95.27% accuracy                                      │
│  • 13 engineered features                               │
│  • Feature importance rankings                          │
│  • Probability predictions & confidence                 │
│  • Model explanation (SHAP or feature contribution)     │
└─────────────────────────────────────────────────────────┘
```

### Data Flow Example: Single Prediction
1. **User Action**: Enters flight details in Streamlit form
2. **Streamlit**: Validates inputs locally
3. **HTTP Request**: Sends POST to `http://localhost:8000/predict`
4. **FastAPI**: Receives request, validates with Pydantic models
5. **ML Model**: Processes features through Gradient Boosting
6. **Response**: Returns JSON with prediction and confidence
7. **Streamlit**: Displays result with visualization

---

## Project Files Summary

### Core Application Files
| File | Purpose | Status |
|------|---------|--------|
| `streamlit_app.py` | Web UI with 5 pages | ✅ Complete |
| `flight_delay_api.py` | REST API backend | ✅ Complete |
| `flight_delay_model.pkl` | Trained ML model | ✅ Auto-generated |
| `launch_app.py` | Automated launcher | ✅ Complete |
| `run_app.bat` | Windows batch launcher | ✅ Complete |

### Configuration Files
| File | Purpose | Status |
|------|---------|--------|
| `.streamlit/config.toml` | Streamlit settings | ✅ Complete |
| `requirements.txt` | Python dependencies | ✅ Complete |

### Documentation Files
| File | Purpose | Status |
|------|---------|--------|
| `README_STREAMLIT.md` | Streamlit guide | ✅ Complete |
| `README_API.md` | API quick start | ✅ Complete |
| `API_DOCUMENTATION.md` | Detailed API docs | ✅ Complete |
| `ANALYSIS_REPORT.md` | ML analysis report | ✅ Complete |

### Testing Files
| File | Purpose | Status |
|------|---------|--------|
| `quick_api_test.py` | API test suite (8 tests) | ✅ All Passed |
| `test_api_client.py` | Python client examples | ✅ Complete |
| `test_api_examples.bat` | cURL examples | ✅ Complete |

### Data Files
| File | Purpose | Status |
|------|---------|--------|
| `ONTIME_REPORTING.csv` | Raw flight data | ✅ 539,747 records |

---

## Performance Characteristics

### Response Times
| Operation | Time | Notes |
|-----------|------|-------|
| Single Prediction | <100ms | Real-time |
| Batch (100 flights) | <1 second | Fast |
| Batch (1000 flights) | 5-10 seconds | Maximum |
| Page Load | 3-5 seconds | Streamlit + API |
| Dashboard Refresh | <1 second | Cached data |

### Resource Usage
| Resource | Usage | Notes |
|----------|-------|-------|
| Memory | ~150MB | With model loaded |
| Disk | ~500MB | Full installation |
| CPU | <20% | Normal operation |
| Network | <1 Mbps | API calls |

---

## Customization

### Change API Port
Edit `flight_delay_api.py` (bottom line):
```python
# Change from:
uvicorn.run(app, host="127.0.0.1", port=8000)
# To:
uvicorn.run(app, host="127.0.0.1", port=8001)
```

### Change Streamlit Port
```bash
streamlit run streamlit_app.py --server.port 8502
```

### Change Theme Colors
Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#FF6B6B"      # Change this hex color
backgroundColor = "#1E1E1E"   # And this
secondaryBackgroundColor = "#262626"
textColor = "#FFFFFF"
```

### Add Custom CSS
Edit `streamlit_app.py`, add to `st.markdown(...)`:
```python
st.markdown("""
<style>
    /* Your custom CSS here */
    .css-myclass { color: blue; }
</style>
""", unsafe_allow_html=True)
```

---

## Next Steps

After successful deployment:

### ✅ Completed
- [x] ML model training and evaluation
- [x] FastAPI REST server creation
- [x] Streamlit web interface design
- [x] API testing and validation
- [x] Documentation

### ⏳ Potential Enhancements
1. **Cloud Deployment**
   - Deploy to AWS, Google Cloud, or Azure
   - Use Docker containers
   - Set up CI/CD pipeline

2. **Advanced Features**
   - Historical prediction tracking
   - Model retraining endpoint
   - User authentication
   - Real-time data integration
   - Email notifications

3. **Performance Optimization**
   - Model quantization
   - Redis caching layer
   - Database integration
   - Load balancing

4. **Monitoring & Analytics**
   - Prediction logging
   - Model drift detection
   - Performance dashboards
   - Error tracking

---

## Support & Help

### Getting Help
1. **Check Troubleshooting** section above
2. **Review logs**:
   - API logs: Check terminal running `flight_delay_api.py`
   - Streamlit logs: Check terminal running Streamlit
3. **API Documentation**: http://localhost:8000/docs
4. **Check file timestamps**: Verify recent changes

### Common Commands Reference
```bash
# Check if ports are available
netstat -ano | findstr :8000
netstat -ano | findstr :8501

# Kill processes by port
taskkill /PID <PID> /F

# Check Python version
python --version

# List installed packages
pip list

# Update a package
pip install --upgrade streamlit
```

---

## Status Summary

| Component | Status | Version |
|-----------|--------|---------|
| **ML Model** | ✅ Ready | V1.0 (95.27% acc.) |
| **API Server** | ✅ Ready | V1.0.0 |
| **Web Interface** | ✅ Ready | V1.0 (5 pages) |
| **Documentation** | ✅ Complete | Complete |
| **Testing** | ✅ Passed | 8/8 tests |
| **Deployment** | ✅ Ready | Ready to use |

**Application Status**: 🟢 FULLY OPERATIONAL

---

**Last Updated**: January 2025  
**Installation Path**: D:\Python\Hackathon  
**Maintained By**: Flight Delay Prediction Team
