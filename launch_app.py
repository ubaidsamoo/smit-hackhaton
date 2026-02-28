#!/usr/bin/env python3
"""
Flight Delay Prediction - App Launcher
Launches both API and Streamlit servers with proper error handling
"""

import subprocess
import sys
import time
import requests
import os
from pathlib import Path

def check_api_health(max_retries=10, timeout=2):
    """Check if API is running and healthy"""
    for attempt in range(max_retries):
        try:
            response = requests.get("http://127.0.0.1:8000/health", timeout=timeout)
            if response.status_code == 200:
                data = response.json()
                print(f"✓ API is healthy: {data}")
                return True
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⏳ Waiting for API... (attempt {attempt + 1}/{max_retries})")
                time.sleep(1)
    return False

def start_api():
    """Start FastAPI server"""
    print("\n" + "="*70)
    print("STARTING FASTAPI SERVER")
    print("="*70 + "\n")
    
    try:
        # Start API in subprocess
        api_process = subprocess.Popen(
            [sys.executable, "flight_delay_api.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print("✓ API process started (PID: {})".format(api_process.pid))
        
        # Wait for API to be healthy
        print("⏳ Waiting for API to initialize...")
        time.sleep(5)
        
        if check_api_health():
            print("✓ API is ready!\n")
            return api_process
        else:
            print("✗ API health check failed")
            api_process.terminate()
            return None
            
    except Exception as e:
        print(f"✗ Error starting API: {e}")
        return None

def start_streamlit():
    """Start Streamlit"""
    print("\n" + "="*70)
    print("STARTING STREAMLIT APPLICATION")
    print("="*70 + "\n")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--logger.level=error"
        ])
    except Exception as e:
        print(f"✗ Error starting Streamlit: {e}")
        return False
    return True

def main():
    """Main launcher"""
    print("\n" + "="*70)
    print("FLIGHT DELAY PREDICTION - APP LAUNCHER")
    print("="*70)
    print("\n📊 Multi-tier Application Startup")
    print("-" * 70)
    print("Components:")
    print("  1. Machine Learning Model (Gradient Boosting, 95.27% accuracy)")
    print("  2. FastAPI REST Server (Port 8000)")
    print("  3. Streamlit Web Interface (Port 8501)")
    print("-" * 70 + "\n")
    
    # Check if streamlit_app.py exists
    if not Path("streamlit_app.py").exists():
        print("✗ streamlit_app.py not found in current directory")
        return False
    
    # Check if flight_delay_api.py exists
    if not Path("flight_delay_api.py").exists():
        print("✗ flight_delay_api.py not found in current directory")
        return False
    
    # Check if model file exists
    if not Path("flight_delay_model.pkl").exists():
        print("⚠ Model file not found. API will train model on startup (may take ~30 seconds)")
    
    # Start API
    api_process = start_api()
    
    if api_process is None:
        print("✗ Failed to start API. Aborting.")
        return False
    
    # Start Streamlit
    try:
        print("\n🌐 Streamlit URLs:")
        print("   • Local: http://localhost:8501")
        print("   • Network: http://<your-ip>:8501")
        print("\n📡 API URLs:")
        print("   • Health: http://localhost:8000/health")
        print("   • Docs: http://localhost:8000/docs")
        print("   • ReDoc: http://localhost:8000/redoc")
        print("\n" + "-"*70)
        print("Press Ctrl+C to stop all services\n")
        
        start_streamlit()
        
    except KeyboardInterrupt:
        print("\n\n⏹ Shutting down services...")
    finally:
        if api_process:
            print("Terminating API...")
            api_process.terminate()
            try:
                api_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                api_process.kill()
        print("✓ All services stopped")

if __name__ == "__main__":
    main()
