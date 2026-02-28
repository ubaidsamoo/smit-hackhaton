"""
Flight Delay Prediction API - Test Client
Demonstrates how to use the API endpoints
"""

import requests
import json
from typing import List, Dict

# API Base URL
BASE_URL = "http://127.0.0.1:8000"

class FlightDelayAPIClient:
    """Client for interacting with Flight Delay Prediction API"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> Dict:
        """Check API health status"""
        response = self.session.get(f"{self.base_url}/health")
        return response.json()
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        response = self.session.get(f"{self.base_url}/model-info")
        return response.json()
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance"""
        response = self.session.get(f"{self.base_url}/feature-importance")
        return response.json()
    
    def predict_single(self, flight_data: Dict) -> Dict:
        """Make a single flight delay prediction"""
        response = self.session.post(
            f"{self.base_url}/predict",
            json=flight_data
        )
        return response.json()
    
    def predict_batch(self, flights_data: List[Dict]) -> Dict:
        """Make batch flight delay predictions"""
        response = self.session.post(
            f"{self.base_url}/predict-batch",
            json={"flights": flights_data}
        )
        return response.json()
    
    def explain_prediction(self, flight_data: Dict) -> Dict:
        """Get detailed explanation of a prediction"""
        response = self.session.post(
            f"{self.base_url}/explain",
            json=flight_data
        )
        return response.json()

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def run_examples():
    """Run example API calls"""
    client = FlightDelayAPIClient()
    
    print("\n" + "="*80)
    print("FLIGHT DELAY PREDICTION API - TEST CLIENT")
    print("="*80)
    
    # Example 1: Health Check
    print("\n[1] HEALTH CHECK")
    print("-"*80)
    try:
        health = client.health_check()
        print(f"✓ API Status: {health['status']}")
        print(f"✓ Model Loaded: {health['model_loaded']}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Example 2: Model Info
    print("\n[2] MODEL INFORMATION")
    print("-"*80)
    try:
        info = client.get_model_info()
        print(f"Model Name: {info['model_name']}")
        print(f"Model Type: {info['model_type']}")
        print(f"Accuracy: {info['accuracy']}")
        print(f"Precision: {info['precision']}")
        print(f"Recall: {info['recall']}")
        print(f"F1-Score: {info['f1_score']}")
        print(f"ROC-AUC: {info['roc_auc']}")
        print(f"Features: {len(info['features'])}")
        print(f"Status: {info['status']}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Example 3: Feature Importance
    print("\n[3] FEATURE IMPORTANCE (Top 5)")
    print("-"*80)
    try:
        importance = client.get_feature_importance()
        for i, feat in enumerate(importance['top_5'], 1):
            print(f"{i}. {feat['feature']}: {feat['importance']:.4f}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Example 4: Single Prediction
    print("\n[4] SINGLE FLIGHT PREDICTION")
    print("-"*80)
    
    # Scenario 1: Flight with minimal delay
    flight1 = {
        "day_of_month": 15,
        "day_of_week": 3,
        "op_carrier_airline_id": 19977,
        "dest_airport_id": 12892,
        "distance": 750.0,
        "dep_delay": 2.0,  # Small departure delay
        "taxi_out": 12.0,
        "taxi_in": 6.0
    }
    
    try:
        pred1 = client.predict_single(flight1)
        print(f"\nFlight 1 (Minimal Delay):")
        print(f"  Prediction: {pred1['status']}")
        print(f"  Probability On-Time: {pred1['probability_on_time']:.2%}")
        print(f"  Probability Delayed: {pred1['probability_delayed']:.2%}")
        print(f"  Confidence: {pred1['confidence']:.2%}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Scenario 2: Flight with significant delay
    flight2 = {
        "day_of_month": 20,
        "day_of_week": 5,
        "op_carrier_airline_id": 20363,
        "dest_airport_id": 12892,
        "distance": 500.0,
        "dep_delay": 45.0,  # Large departure delay
        "taxi_out": 25.0,
        "taxi_in": 12.0
    }
    
    try:
        pred2 = client.predict_single(flight2)
        print(f"\nFlight 2 (Significant Delay):")
        print(f"  Prediction: {pred2['status']}")
        print(f"  Probability On-Time: {pred2['probability_on_time']:.2%}")
        print(f"  Probability Delayed: {pred2['probability_delayed']:.2%}")
        print(f"  Confidence: {pred2['confidence']:.2%}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Example 5: Batch Prediction
    print("\n[5] BATCH FLIGHT PREDICTIONS")
    print("-"*80)
    
    flights = [
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
        },
        {
            "day_of_month": 5,
            "day_of_week": 1,
            "op_carrier_airline_id": 20366,
            "dest_airport_id": 10397,
            "distance": 800.0,
            "dep_delay": 15.0,
            "taxi_out": 16.0,
            "taxi_in": 8.0
        }
    ]
    
    try:
        batch_result = client.predict_batch(flights)
        print(f"\nTotal Flights: {batch_result['total_flights']}")
        print(f"On-Time Flights: {batch_result['on_time_count']}")
        print(f"Delayed Flights: {batch_result['delayed_count']}")
        print(f"Delay Rate: {batch_result['delayed_percentage']:.1f}%")
        
        print(f"\nDetailed Predictions:")
        for i, pred in enumerate(batch_result['predictions'], 1):
            print(f"  Flight {i}: {pred['status']} (Confidence: {pred['confidence']:.1%})")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Example 6: Detailed Explanation
    print("\n[6] DETAILED PREDICTION EXPLANATION")
    print("-"*80)
    
    flight = {
        "day_of_month": 25,
        "day_of_week": 4,
        "op_carrier_airline_id": 19977,
        "dest_airport_id": 12892,
        "distance": 1500.0,
        "dep_delay": 50.0,
        "taxi_out": 28.0,
        "taxi_in": 11.0
    }
    
    try:
        explanation = client.explain_prediction(flight)
        print(f"\nPrediction Status: {explanation['status']}")
        print(f"Probability Delayed: {explanation['probability_delayed']:.2%}")
        print(f"Confidence: {explanation['confidence']:.2%}")
        print(f"\nInterpretation:")
        print(f"{explanation['interpretation']}")
        print(f"\nEngineered Features:")
        for feat, value in explanation['engineered_features'].items():
            print(f"  {feat}: {value}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "="*80)
    print("TEST EXAMPLES COMPLETED")
    print("="*80 + "\n")

if __name__ == "__main__":
    print("\n⏳ Waiting for API to be available...")
    print("Make sure the API is running: python flight_delay_api.py")
    
    # Try to connect to API
    import time
    max_retries = 10
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            client = FlightDelayAPIClient()
            client.health_check()
            print("✓ API is running!\n")
            break
        except:
            retry_count += 1
            if retry_count < max_retries:
                print(f"⏳ Attempt {retry_count}/{max_retries}: Waiting for API...")
                time.sleep(2)
            else:
                print("✗ Could not connect to API")
                print("Please ensure the API is running: python flight_delay_api.py")
                exit(1)
    
    # Run examples
    run_examples()
