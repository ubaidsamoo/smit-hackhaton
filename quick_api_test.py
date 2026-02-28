"""
Quick API Test - Demonstrates API functionality
Run this after starting the API server (python flight_delay_api.py)
"""

import time
import requests
import json

BASE_URL = "http://127.0.0.1:8000"

print("\n" + "="*80)
print("FLIGHT DELAY PREDICTION API - QUICK TESTS")
print("="*80)

# Test 1: Root endpoint
print("\n[TEST 1] Root Endpoint")
print("-"*80)
try:
    response = requests.get(f"{BASE_URL}/")
    data = response.json()
    print("✓ API is running!")
    print(f"  Version: {data['version']}")
    print(f"  Status: {data['status']}")
except Exception as e:
    print(f"✗ Error: {e}")
    print("  Make sure API is running: python flight_delay_api.py")
    exit(1)

# Test 2: Health Check
print("\n[TEST 2] Health Check")
print("-"*80)
try:
    response = requests.get(f"{BASE_URL}/health")
    data = response.json()
    print(f"✓ API Status: {data['status']}")
    print(f"✓ Model Loaded: {data['model_loaded']}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 3: Model Info
print("\n[TEST 3] Model Information")
print("-"*80)
try:
    response = requests.get(f"{BASE_URL}/model-info")
    data = response.json()
    print(f"✓ Model: {data['model_name']}")
    print(f"✓ Type: {data['model_type']}")
    print(f"✓ Accuracy: {data['accuracy']}")
    print(f"✓ Precision: {data['precision']}")
    print(f"✓ Recall: {data['recall']}")
    print(f"✓ F1-Score: {data['f1_score']}")
    print(f"✓ ROC-AUC: {data['roc_auc']}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 4: Feature Importance
print("\n[TEST 4] Top 5 Important Features")
print("-"*80)
try:
    response = requests.get(f"{BASE_URL}/feature-importance")
    data = response.json()
    print("Features ranked by importance:")
    for i, feature in enumerate(data['top_5'], 1):
        print(f"  {i}. {feature['feature']}: {feature['importance']:.2%}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 5: Single Prediction - Low Risk Flight
print("\n[TEST 5] Single Prediction - Low Risk Flight")
print("-"*80)
flight = {
    "day_of_month": 10,
    "day_of_week": 2,
    "op_carrier_airline_id": 19977,
    "dest_airport_id": 12892,
    "distance": 600.0,
    "dep_delay": 2.0,
    "taxi_out": 12.0,
    "taxi_in": 6.0
}
try:
    response = requests.post(f"{BASE_URL}/predict", json=flight)
    data = response.json()
    print(f"✓ Prediction: {data['status']}")
    print(f"✓ Probability On-Time: {data['probability_on_time']:.2%}")
    print(f"✓ Probability Delayed: {data['probability_delayed']:.2%}")
    print(f"✓ Confidence: {data['confidence']:.2%}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 6: Single Prediction - High Risk Flight
print("\n[TEST 6] Single Prediction - High Risk Flight")
print("-"*80)
flight = {
    "day_of_month": 20,
    "day_of_week": 5,
    "op_carrier_airline_id": 20363,
    "dest_airport_id": 11433,
    "distance": 1200.0,
    "dep_delay": 45.0,
    "taxi_out": 25.0,
    "taxi_in": 12.0
}
try:
    response = requests.post(f"{BASE_URL}/predict", json=flight)
    data = response.json()
    print(f"✓ Prediction: {data['status']}")
    print(f"✓ Probability On-Time: {data['probability_on_time']:.2%}")
    print(f"✓ Probability Delayed: {data['probability_delayed']:.2%}")
    print(f"✓ Confidence: {data['confidence']:.2%}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 7: Batch Predictions
print("\n[TEST 7] Batch Predictions (3 Flights)")
print("-"*80)
flights = {
    "flights": [
        {
            "day_of_month": 5,
            "day_of_week": 1,
            "op_carrier_airline_id": 19977,
            "dest_airport_id": 12892,
            "distance": 500.0,
            "dep_delay": 3.0,
            "taxi_out": 13.0,
            "taxi_in": 7.0
        },
        {
            "day_of_month": 15,
            "day_of_week": 3,
            "op_carrier_airline_id": 20363,
            "dest_airport_id": 11433,
            "distance": 1000.0,
            "dep_delay": 25.0,
            "taxi_out": 18.0,
            "taxi_in": 9.0
        },
        {
            "day_of_month": 25,
            "day_of_week": 6,
            "op_carrier_airline_id": 20366,
            "dest_airport_id": 10397,
            "distance": 800.0,
            "dep_delay": 50.0,
            "taxi_out": 28.0,
            "taxi_in": 11.0
        }
    ]
}
try:
    response = requests.post(f"{BASE_URL}/predict-batch", json=flights)
    data = response.json()
    print(f"✓ Total Flights: {data['total_flights']}")
    print(f"✓ On-Time: {data['on_time_count']}")
    print(f"✓ Delayed: {data['delayed_count']}")
    print(f"✓ Delay Rate: {data['delayed_percentage']:.1f}%")
    print(f"\nPredictions:")
    for i, pred in enumerate(data['predictions'], 1):
        print(f"  Flight {i}: {pred['status']} (Confidence: {pred['confidence']:.1%})")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 8: Detailed Explanation
print("\n[TEST 8] Prediction Explanation")
print("-"*80)
flight = {
    "day_of_month": 22,
    "day_of_week": 4,
    "op_carrier_airline_id": 19977,
    "dest_airport_id": 12892,
    "distance": 1500.0,
    "dep_delay": 55.0,
    "taxi_out": 30.0,
    "taxi_in": 12.0
}
try:
    response = requests.post(f"{BASE_URL}/explain", json=flight)
    data = response.json()
    print(f"✓ Status: {data['status']}")
    print(f"✓ Probability Delayed: {data['probability_delayed']:.2%}")
    print(f"✓ Confidence: {data['confidence']:.2%}")
    print(f"\n  Interpretation:")
    for line in data['interpretation'].split('. '):
        if line:
            print(f"  {line}.")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "="*80)
print("ALL TESTS COMPLETED SUCCESSFULLY! ✓")
print("="*80)
print("\nAPI is ready for production use.")
print("Documentation: http://localhost:8000/docs")
print("="*80 + "\n")
