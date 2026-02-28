# Flight Delay Prediction - Streamlit Deployment Guide

## Overview
This Streamlit application provides a comprehensive web interface for the Flight Delay Prediction system. The app connects to the FastAPI backend to deliver real-time predictions and analysis capabilities.

## Architecture
```
┌─────────────────────────────────────────────────────────┐
│          STREAMLIT WEB APPLICATION                      │
│                  (Port 8501)                            │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Dashboard │ Predict │ Batch │ Analysis │ About         │
│                                                          │
└──────────────────┬──────────────────────────────────────┘
                   │
                   │ HTTP Requests
                   ↓
┌─────────────────────────────────────────────────────────┐
│          FASTAPI REST API                               │
│                  (Port 8000)                            │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  /predict │ /batch │ /info │ /importance │ /explain    │
│                                                          │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────┐
│     MACHINE LEARNING MODEL                              │
│  Gradient Boosting Classifier (95.27% Accuracy)        │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### Option 1: Automated Startup (Recommended)
```bash
# Double-click run_app.bat to start both API and Streamlit
run_app.bat
```

### Option 2: Manual Startup

#### Start the API Server
```bash
# Terminal 1 - Start FastAPI
python flight_delay_api.py
# API will be available at: http://localhost:8000
```

#### Start Streamlit
```bash
# Terminal 2 - Start Streamlit
streamlit run streamlit_app.py
# Streamlit will be available at: http://localhost:8501
```

## Pages & Features

### 1. 🏠 Dashboard
- **Model Metrics**: Displays accuracy, precision, recall, ROC-AUC score
- **Feature Importance Chart**: Interactive bar chart showing top features
- **Quick Stats**: Total flights analyzed, model version, training date
- **Performance Overview**: Visual representation of model performance

### 2. 🔮 Predict Flight
- **Single Flight Prediction**: Input form for flight details
- **Required Fields**:
  - Departure Delay (minutes)
  - Taxi Out Time (minutes)
  - Taxi In Time (minutes)
  - Distance (miles)
  - Weather Condition
  - Day of Week
  - Month
- **Output**: Prediction (ON_TIME or DELAYED) with confidence percentage
- **Visualization**: Probability distribution chart

### 3. 📊 Batch Predictions
- **CSV Upload**: Upload multiple flights for prediction
- **CSV Format**: Requires columns matching single prediction fields
- **Results Display**: Table with predictions and confidence scores
- **Export**: Download results as CSV file
- **Batch Limits**: Up to 1000 flights per request

### 4. 📈 Model Analysis
- **Detailed Metrics**: Complete model performance statistics
- **Confusion Matrix**: Visual representation of predictions
- **Feature Rankings**: Top 13 features by importance
- **Performance Insights**: Key findings and model behavior analysis
- **Calibration Curve**: Probability calibration visualization

### 5. ℹ️ About
- **Project Overview**: Description and objectives
- **Technology Stack**: Tools and libraries used
- **API Documentation**: Links to OpenAPI docs
- **Contact & Support**: Help resources

## API Endpoints Used by Streamlit

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Check API status |
| `/model-info` | GET | Get model metrics |
| `/feature-importance` | GET | Get feature rankings |
| `/predict` | POST | Single prediction |
| `/predict-batch` | POST | Batch predictions |

## Configuration

### Streamlit Settings (.streamlit/config.toml)
```toml
# Server Port
port = 8501

# Theme
primaryColor = "#FF6B6B"
backgroundColor = "#1E1E1E"

# Max Upload Size (MB)
maxUploadSize = 500

# Auto-reload on save
runOnSave = true
```

### API Configuration
- **Base URL**: http://127.0.0.1:8000
- **Timeout**: 30 seconds
- **Batch Size Limit**: 1000 flights

## Requirements

### Python Packages
```
streamlit==1.28.0
plotly==5.17.0
pandas==2.0.3
requests==2.31.0
streamlit-option-menu==0.3.7
pydantic==2.4.0
scikit-learn==1.3.2
```

### System Requirements
- Python 3.8+
- Windows/Mac/Linux
- 500MB RAM minimum
- Internet connection (for initial setup)

## Troubleshooting

### Issue: "Connection refused" to API
**Solution**: Ensure API is running
```bash
python flight_delay_api.py
```

### Issue: Streamlit page blank
**Solution**: Check browser console for errors, restart Streamlit
```bash
streamlit run streamlit_app.py --logger.level=debug
```

### Issue: CSV upload fails
**Solution**: Verify CSV format matches required columns
```
Required columns: DEP_DELAY, TAXI_OUT, TAXI_IN, DISTANCE, WEATHER_CONDITION, DAY_OF_WEEK, MONTH
```

### Issue: Slow predictions
**Solution**: Ensure API is not overloaded
```bash
# Check API logs in the API terminal
# Maximum 1000 flights per batch request
```

## Performance Metrics

| Metric | Value |
|--------|-------|
| Single Prediction Time | <100ms |
| Batch Processing (1000 flights) | ~5-10 seconds |
| Memory Usage | ~150MB (with model loaded) |
| API Response Time | <50ms average |
| Streamlit Load Time | ~3 seconds |

## Development Notes

### Caching Strategy
- Model info cached for 5 minutes
- Feature importance cached for 5 minutes
- Predictions not cached (real-time)

### Error Handling
- API timeouts: 30 seconds
- Connection errors: Graceful fallback with error message
- Invalid input: Validation on client and server side

### Data Flow
1. User enters/uploads flight data in Streamlit
2. Streamlit validates input locally
3. Streamlit sends POST request to FastAPI
4. API processes request and returns prediction
5. Streamlit displays result with visualization

## Deployment Checklist

- [ ] Python environment configured with all dependencies
- [ ] FastAPI server running and tested (8/8 tests passed)
- [ ] Streamlit installation verified
- [ ] API and Streamlit ports available (8000, 8501)
- [ ] Test single prediction in Streamlit
- [ ] Test batch prediction with CSV upload
- [ ] Verify all 5 pages render correctly
- [ ] Check visualizations load properly
- [ ] Test error handling with invalid inputs
- [ ] Document any customizations made

## Advanced Features

### CSV Batch Template
Create a CSV file with these columns:
```csv
DEP_DELAY,TAXI_OUT,TAXI_IN,DISTANCE,WEATHER_CONDITION,DAY_OF_WEEK,MONTH
15,20,10,500,CLEAR,3,6
25,25,12,750,CLOUDY,5,9
-5,15,8,300,RAINY,1,12
```

### Custom Styling
The app uses custom CSS with:
- Dark theme (dark gray background)
- Red accent color (#FF6B6B)
- Responsive layout
- Custom fonts for headers

## Support & Documentation

| Resource | Location |
|----------|----------|
| API Docs | http://localhost:8000/docs |
| API Schema | http://localhost:8000/openapi.json |
| This Guide | README_STREAMLIT.md |
| ML Analysis | ANALYSIS_REPORT.md |
| API Guide | API_DOCUMENTATION.md |

## Next Steps

1. ✅ Verify API is running
2. ✅ Start Streamlit application
3. ✅ Test all 5 pages
4. ✅ Perform sample predictions
5. ⏳ Consider cloud deployment (Streamlit Cloud, AWS, etc.)
6. ⏳ Add real-time data integration
7. ⏳ Implement user feedback collection

## License & Credits

Flight Delay Prediction System - Educational Project

**Components**:
- ML Model: scikit-learn Gradient Boosting Classifier
- API Framework: FastAPI with Uvicorn
- Web Framework: Streamlit
- Data Source: Flight delay dataset with 539,747 records

---

**Last Updated**: January 2025
**Status**: Ready for deployment ✅
