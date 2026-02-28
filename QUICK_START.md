# ⚡ Quick Start Guide - Flight Delay Prediction System

## 🚀 Start in 30 Seconds

> **Note:** if deploying on Streamlit Cloud or another host, update `requirements.txt` to include:
> ```text
> streamlit plotly streamlit-option-menu requests
> ```
> Missing these will lead to `ModuleNotFoundError` when the app imports them.

### Option 1: Single Command (Easiest)
```bash
python launch_app.py
```
Wait for URLs to appear in terminal → Open browser to `http://localhost:8501`

### Option 2: Batch File (Windows)
Double-click `run_app.bat` → Wait 10 seconds → Browser opens automatically

### Option 3: Manual (If needed for debugging)
**Terminal 1:**
```bash
python flight_delay_api.py
```

**Terminal 2:**
```bash
streamlit run streamlit_app.py
```

---

## 📍 What to Expect

✅ **API Server**: http://localhost:8000
- Health check: http://localhost:8000/health
- Documentation: http://localhost:8000/docs

✅ **Web Application**: http://localhost:8501
- Dashboard with metrics
- Single flight predictions
- Batch CSV predictions
- Model analysis

---

## 🎯 First Steps

### 1. Try Dashboard
- Page loads automatically
- Shows model accuracy: **95.27%**
- Shows feature importance

### 2. Make a Prediction
- Click "🔮 Predict Flight"
- Fill in example values:
  - Departure Delay: `15`
  - Taxi Out: `20`
  - Taxi In: `10`
  - Distance: `500`
  - Weather: `CLEAR`
  - Day: `3`
  - Month: `6`
- Click "Predict"
- See result: `ON_TIME` (99% confidence)

### 3. Try Batch Prediction
- Click "📊 Batch Predictions"
- Create CSV file with header:
  ```
  DEP_DELAY,TAXI_OUT,TAXI_IN,DISTANCE,WEATHER_CONDITION,DAY_OF_WEEK,MONTH
  15,20,10,500,CLEAR,3,6
  20,25,12,750,CLOUDY,5,9
  ```
- Upload CSV
- See predictions table
- Download results

---

## 🔧 Troubleshooting

### Not Working?
1. **Check ports**:
   ```bash
   # Open new terminal and run:
   curl http://localhost:8000/health
   ```
   Should show: `{"status": "healthy", "model_loaded": true}`

2. **Kill stuck processes**:
   ```bash
   # Windows:
   netstat -ano | findstr :8501
   taskkill /PID <PID> /F
   ```

3. **Reinstall dependencies**:
   ```bash
   pip install streamlit fastapi uvicorn plotly pandas requests
   ```

---

## 📂 File Structure
```
D:\Python\Hackathon\
├── streamlit_app.py          ← Web interface (run this)
├── flight_delay_api.py       ← API server (auto-starts)
├── launch_app.py             ← Automated launcher ⭐
├── run_app.bat               ← Batch launcher (Windows)
├── flight_delay_model.pkl    ← ML model (auto-generated)
├── .streamlit/config.toml    ← Streamlit settings
├── DEPLOYMENT_GUIDE.md       ← Full guide (this file)
├── README_STREAMLIT.md       ← Streamlit docs
├── README_API.md             ← API docs
└── ANALYSIS_REPORT.md        ← ML analysis
```

---

## 📊 Model Information

- **Type**: Gradient Boosting Classifier
- **Accuracy**: 95.27%
- **Precision**: 91.22%
- **Recall**: 82.23%
- **Features**: 13 engineered features
- **Training Data**: 539,747 flights

---

## 🌐 Access Points

| Service | URL | Purpose |
|---------|-----|---------|
| **Streamlit UI** | http://localhost:8501 | 🎯 Main interface |
| **API Server** | http://localhost:8000 | Backend service |
| **API Docs** | http://localhost:8000/docs | Interactive API |
| **Health Check** | http://localhost:8000/health | Status check |

---

## ⌨️ Keyboard Shortcuts

| Action | Keys |
|--------|------|
| Stop Application | `Ctrl + C` (in terminal) |
| Hard Refresh Browser | `Ctrl + Shift + F5` |
| Open DevTools | `F12` |
| Clear Browser Cache | `Ctrl + Shift + Del` |

---

## 💡 Tips

- **Slow startup?** First load trains the ML model (~30 sec)
- **Port in use?** Change port in config file or use different port
- **CSV errors?** Make sure exact column names match: `DEP_DELAY,TAXI_OUT,TAXI_IN,DISTANCE,WEATHER_CONDITION,DAY_OF_WEEK,MONTH`
- **Want more data?** API supports up to 1000 flights per batch

---

## ✅ Verification Checklist

- [ ] Python installed (`python --version`)
- [ ] Dependencies installed (`pip list | find "streamlit"`)
- [ ] Ports 8000 and 8501 available
- [ ] API starts without errors
- [ ] Streamlit app loads at http://localhost:8501
- [ ] Dashboard shows metrics
- [ ] Single prediction works
- [ ] Batch prediction works

---

## 🆘 Still Having Issues?

1. **Read**: `DEPLOYMENT_GUIDE.md` (full troubleshooting)
2. **Check**: Terminal output for error messages
3. **Verify**: All files present in project directory
4. **Try**: Reinstalling packages: `pip install -r requirements.txt`
5. **Restart**: Kill all terminals and start fresh

---

## 📞 Quick Reference

```bash
# Full deployment guide
cat DEPLOYMENT_GUIDE.md

# API documentation
http://localhost:8000/docs

# View API health
curl http://localhost:8000/health

# Stop all servers
# Press Ctrl+C in the terminal running launch_app.py
```

---

**Status**: ✅ Ready to Use  
**Latest Update**: January 2025  
**Version**: 1.0  

🎉 **You're all set! Start with `python launch_app.py`**
