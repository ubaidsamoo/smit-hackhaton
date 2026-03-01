"""
Flight Delay Prediction - Streamlit Web Application
Interactive dashboard for flight delay predictions
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import time

# Page configuration
st.set_page_config(
    page_title="Flight Delay Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    :root{
        --accent:#1f77b4;
        --muted:#6b7280;
        --bg:#f7fafc;
        --card:#ffffff;
        --success:#2ecc71;
        --danger:#e74c3c;
        --glass: rgba(255,255,255,0.6);
    }
    html, body, [class*="css"]  {
        font-family: 'Inter', system-ui, sans-serif !important;
    }
    .app-hero{
        background: linear-gradient(90deg, rgba(31,119,180,0.95), rgba(73,138,204,0.95));
        color: white;
        padding: 1.2rem 1.6rem;
        border-radius: 12px;
        box-shadow: 0 8px 30px rgba(31,119,180,0.12);
        margin-bottom: 1rem;
    }
    .app-hero h1{margin:0; font-size:1.6rem; font-weight:800}
    .app-hero p{margin:0.35rem 0 0; color: rgba(255,255,255,0.92)}

    .card {
        background: var(--card);
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 6px 18px rgba(15,23,42,0.06);
        border: 1px solid rgba(15,23,42,0.03);
    }

    .metric-inline{
        display:flex; gap:1rem; align-items:center; justify-content:flex-start;
    }
    .metric-value{font-weight:700; font-size:1.1rem; color:var(--accent)}

    .prediction-on-time{ color: var(--success); font-weight:700}
    .prediction-delayed{ color: var(--danger); font-weight:700}

    /* Tighter sidebar look */
    .stSidebar .css-1d391kg { padding-top: 1rem; }
    .stButton>button { border-radius: 8px; padding: 0.55rem 0.9rem; }

    footer {visibility: visible}
    </style>
""", unsafe_allow_html=True)


def render_hero(title: str, subtitle: str = ""):
    """Render a compact hero/header with a gradient background and subtitle."""
    st.markdown(f"""
    <div class="app-hero">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://127.0.0.1:8000"

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

with st.sidebar:
    st.title("✈️ Flight Delay Predictor")
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🔮 Predict Flight", "📊 Batch Predictions", "📈 Model Analysis", "ℹ️ About"]
    )
    
    st.markdown("---")
    
    # API Status
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if response.status_code == 200:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Error")
    except:
        st.error("❌ API Offline")
    
    st.markdown("---")
    st.caption("Flight Delay Prediction API v1.0.0")
    st.caption("Powered by Gradient Boosting Classifier")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data
def get_model_info():
    """Fetch model information from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/model-info", timeout=5)
        return response.json()
    except Exception as e:
        st.error(f"Error fetching model info: {e}")
        return None

@st.cache_data
def get_feature_importance():
    """Fetch feature importance from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/feature-importance", timeout=5)
        return response.json()
    except Exception as e:
        st.error(f"Error fetching feature importance: {e}")
        return None

def predict_flight(flight_data):
    """Make a single flight prediction"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=flight_data,
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def predict_batch(flights_data):
    """Make batch predictions"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict-batch",
            json={"flights": flights_data},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Batch prediction error: {e}")
        return None

def get_prediction_explanation(flight_data):
    """Get detailed explanation of a prediction"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/explain",
            json=flight_data,
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        return None

# ============================================================================
# PAGE 1: DASHBOARD
# ============================================================================

if page == "🏠 Dashboard":
    render_hero("✈️ Flight Delay Prediction Dashboard", "Interactive insights and real-time predictions")
    
    # Get model info
    model_info = get_model_info()
    
    if model_info:
        # Key Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Accuracy", model_info['accuracy'])
        with col2:
            st.metric("Precision", model_info['precision'])
        with col3:
            st.metric("Recall", model_info['recall'])
        with col4:
            st.metric("F1-Score", model_info['f1_score'])
        with col5:
            st.metric("ROC-AUC", model_info['roc_auc'])
        
        st.markdown("---")
        
        # Model Details
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Model Information")
            st.write(f"**Model Name:** {model_info['model_name']}")
            st.write(f"**Model Type:** {model_info['model_type']}")
            st.write(f"**Training Date:** {model_info['training_date']}")
            st.write(f"**Status:** {model_info['status']}")
            st.write(f"**Features:** {model_info['features_count']}")
        
        with col2:
            st.subheader("🎯 What This Means")
            st.write("✅ **95% Accuracy** - Correctly predicts 95 out of 100 flights")
            st.write("✅ **91% Precision** - Low false alarm rate")
            st.write("✅ **82% Recall** - Catches most actual delays")
            st.write("✅ **0.98 ROC-AUC** - Excellent discrimination ability")
        
        st.markdown("---")
        
        # Feature Importance
        importance_data = get_feature_importance()
        if importance_data:
            st.subheader("⭐ Top 5 Most Important Features")
            
            df_importance = pd.DataFrame(importance_data['top_5'])
            
            fig = px.bar(
                df_importance,
                x='importance',
                y='feature',
                orientation='h',
                title='Feature Importance Ranking',
                labels={'importance': 'Importance Score', 'feature': 'Feature'},
                color='importance',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
            **Key Insight:** Departure delay (DEP_DELAY) is by far the most important 
            feature (79%), meaning flights that depart late almost always arrive late.
            """)

# ============================================================================
# PAGE 2: SINGLE FLIGHT PREDICTION
# ============================================================================

elif page == "🔮 Predict Flight":
    render_hero("✈️ Flight Delay Predictor", "Enter details below to predict arrival delay probability")
    
    st.write("Enter flight details to predict arrival delay probability.")
    st.markdown("---")
    
    # Create input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("✈️ Flight Details")
        day_of_month = st.slider("Day of Month", 1, 31, 15)
        day_of_week = st.selectbox(
            "Day of Week",
            options=[1, 2, 3, 4, 5, 6, 7],
            format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x-1]
        )
        op_carrier_airline_id = st.number_input("Airline ID", min_value=10000, max_value=30000, value=19977)
        dest_airport_id = st.number_input("Destination Airport ID", min_value=10000, max_value=20000, value=12892)
    
    with col2:
        st.subheader("📊 Flight Metrics")
        distance = st.number_input("Distance (miles)", min_value=0.0, max_value=3000.0, value=750.0)
        dep_delay = st.number_input("Departure Delay (minutes)", min_value=-30.0, max_value=180.0, value=5.0)
        taxi_out = st.number_input("Taxi Out Time (minutes)", min_value=0.0, max_value=60.0, value=15.0)
        taxi_in = st.number_input("Taxi In Time (minutes)", min_value=0.0, max_value=50.0, value=8.0)
    
    st.markdown("---")
    
    # Predict Button
    if st.button("🔮 Predict Delay", use_container_width=True):
        flight_data = {
            "day_of_month": int(day_of_month),
            "day_of_week": int(day_of_week),
            "op_carrier_airline_id": int(op_carrier_airline_id),
            "dest_airport_id": int(dest_airport_id),
            "distance": float(distance),
            "dep_delay": float(dep_delay),
            "taxi_out": float(taxi_out),
            "taxi_in": float(taxi_in)
        }
        
        with st.spinner("Making prediction..."):
            prediction = predict_flight(flight_data)
        
        if prediction:
            st.success("✅ Prediction Complete!")
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction['status'] == 'ON_TIME':
                    st.markdown(f"""
                    <div class="card">
                        <div class="prediction-on-time">✅ ON TIME</div>
                        <p>Flight expected to arrive on time</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="card">
                        <div class="prediction-delayed">⚠️ DELAYED</div>
                        <p>Flight expected to be delayed</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.metric(
                    "Probability On-Time",
                    f"{prediction['probability_on_time']*100:.1f}%"
                )
            
            with col3:
                st.metric(
                    "Confidence Score",
                    f"{prediction['confidence']*100:.1f}%"
                )
            
            st.markdown("---")
            
            # Detailed explanation
            explanation = get_prediction_explanation(flight_data)
            if explanation:
                st.subheader("📖 Prediction Explanation")
                st.write(f"**Interpretation:** {explanation['interpretation']}")
                
                # Show engineered features
                with st.expander("🔍 Feature Details"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Input Features:**")
                        st.write(f"- Day of Month: {day_of_month}")
                        st.write(f"- Day of Week: {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][day_of_week-1]}")
                        st.write(f"- Distance: {distance} miles")
                        st.write(f"- Departure Delay: {dep_delay} min")
                    
                    with col2:
                        st.write("**Engineered Features:**")
                        for feat, value in explanation['engineered_features'].items():
                            st.write(f"- {feat}: {value}")
            
            # Probability visualization
            st.markdown("---")
            fig = go.Figure(data=[
                go.Bar(
                    x=['On-Time', 'Delayed'],
                    y=[prediction['probability_on_time'] * 100, prediction['probability_delayed'] * 100],
                    marker_color=['#2ecc71', '#e74c3c'],
                    text=[f"{prediction['probability_on_time']*100:.1f}%", f"{prediction['probability_delayed']*100:.1f}%"],
                    textposition='auto',
                )
            ])
            fig.update_layout(
                title="Prediction Probability Distribution",
                yaxis_title="Probability (%)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 3: BATCH PREDICTIONS
# ============================================================================

elif page == "📊 Batch Predictions":
    render_hero("📊 Batch Flight Predictions", "Upload a CSV with flights to get batch predictions")
    
    st.write("Upload a CSV file with multiple flights to get predictions for all of them.")
    st.markdown("---")
    
    # CSV Template
    with st.expander("📋 CSV Format Template"):
        template_data = {
            'day_of_month': [15, 20, 25],
            'day_of_week': [3, 5, 1],
            'op_carrier_airline_id': [19977, 20363, 19977],
            'dest_airport_id': [12892, 11433, 10397],
            'distance': [750.0, 1200.0, 500.0],
            'dep_delay': [5.0, 30.0, 15.0],
            'taxi_out': [15.0, 25.0, 12.0],
            'taxi_in': [8.0, 12.0, 7.0]
        }
        template_df = pd.DataFrame(template_data)
        st.write("**Required columns:**")
        st.dataframe(template_df, use_container_width=True)
        
        # Download template
        csv = template_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Template CSV",
            data=csv,
            file_name="flight_predictions_template.csv",
            mime="text/csv"
        )
    
    st.markdown("---")
    
    # File upload
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Loaded {len(df)} flights")
            
            # Preview data
            with st.expander("👀 Preview Data"):
                st.dataframe(df, use_container_width=True)
            
            # Check max flights
            if len(df) > 1000:
                st.error("❌ Maximum 1000 flights per batch")
            else:
                if st.button("🔮 Predict Batch", use_container_width=True):
                    # Convert to list of dicts
                    flights_list = df.to_dict('records')
                    
                    with st.spinner(f"Predicting {len(flights_list)} flights..."):
                        results = predict_batch(flights_list)
                    
                    if results:
                        st.success("✅ Predictions Complete!")
                        
                        # Summary statistics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Flights", results['total_flights'])
                        with col2:
                            st.metric("On-Time", results['on_time_count'])
                        with col3:
                            st.metric("Delayed", results['delayed_count'])
                        with col4:
                            st.metric("Delay Rate", f"{results['delayed_percentage']:.1f}%")
                        
                        st.markdown("---")
                        
                        # Results table
                        st.subheader("📊 Prediction Results")
                        
                        results_df = pd.DataFrame([
                            {
                                'Flight #': i+1,
                                'Status': pred['status'],
                                'Probability On-Time': f"{pred['probability_on_time']*100:.1f}%",
                                'Probability Delayed': f"{pred['probability_delayed']*100:.1f}%",
                                'Confidence': f"{pred['confidence']*100:.1f}%"
                            }
                            for i, pred in enumerate(results['predictions'])
                        ])
                        
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Download results
                        csv_results = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results CSV",
                            data=csv_results,
                            file_name="flight_predictions_results.csv",
                            mime="text/csv"
                        )
                        
                        # Summary visualization
                        st.markdown("---")
                        
                        fig = go.Figure(data=[
                            go.Pie(
                                labels=['On-Time', 'Delayed'],
                                values=[results['on_time_count'], results['delayed_count']],
                                marker=dict(colors=['#2ecc71', '#e74c3c']),
                                textinfo='label+percent'
                            )
                        ])
                        fig.update_layout(
                            title="Flight Status Distribution",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error processing file: {e}")

# ============================================================================
# PAGE 4: MODEL ANALYSIS
# ============================================================================

elif page == "📈 Model Analysis":
    render_hero("📈 Model Analysis & Insights", "Understand model performance and the features that drive predictions")
    
    model_info = get_model_info()
    importance_data = get_feature_importance()
    
    if model_info and importance_data:
        # Performance metrics
        st.subheader("📊 Model Performance Metrics")
        
        metrics = {
            'Accuracy': float(model_info['accuracy'].rstrip('%')) / 100,
            'Precision': float(model_info['precision'].rstrip('%')) / 100,
            'Recall': float(model_info['recall'].rstrip('%')) / 100,
            'F1-Score': float(model_info['f1_score']),
            'ROC-AUC': float(model_info['roc_auc'])
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Metrics bars
            fig = go.Figure(data=[
                go.Bar(
                    x=list(metrics.values()),
                    y=list(metrics.keys()),
                    orientation='h',
                    marker_color=['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#e74c3c'],
                    text=[f"{v*100:.2f}" for v in metrics.values()],
                    textposition='auto',
                )
            ])
            fig.update_layout(
                title="Performance Metrics",
                xaxis_title="Score",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("### Interpretation")
            st.write("""
            - **Accuracy (95.27%)**: Model correctly predicts 95 out of 100 flights
            - **Precision (91.22%)**: When predicting delay, it's correct 91% of the time
            - **Recall (82.23%)**: Model catches 82% of all actual delays
            - **F1-Score (0.8649)**: Balanced measure of precision and recall
            - **ROC-AUC (0.9803)**: Excellent ability to distinguish delays from on-time flights
            """)
        
        st.markdown("---")
        
        # Feature importance detailed
        st.subheader("⭐ Feature Importance Analysis")
        
        df_all_features = pd.DataFrame(importance_data['features'])
        
        fig = px.bar(
            df_all_features,
            x='importance',
            y='feature',
            orientation='h',
            title='All Features Ranked by Importance',
            labels={'importance': 'Importance Score', 'feature': 'Feature'},
            color='importance',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("### Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Tier 1 Features (> 50%):**")
            for feat in df_all_features[df_all_features['importance'] > 0.5].to_dict('records'):
                st.write(f"- {feat['feature']}: {feat['importance']*100:.1f}%")
        
        with col2:
            st.write("**Tier 2 Features (1-50%):**")
            tier2 = df_all_features[(df_all_features['importance'] >= 0.01) & (df_all_features['importance'] <= 0.5)]
            for feat in tier2.to_dict('records'):
                st.write(f"- {feat['feature']}: {feat['importance']*100:.1f}%")
        
        st.markdown("---")
        
        # Model characteristics
        st.subheader("🎯 Model Characteristics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Algorithm**")
            st.write("Gradient Boosting Classifier")
            st.write("- 50 estimators")
            st.write("- Max depth: 5")
            st.write("- Handles non-linear patterns")
        
        with col2:
            st.write("**Training Data**")
            st.write("539,747 flights")
            st.write("- 81.84% on-time")
            st.write("- 18.16% delayed")
            st.write("- 522,269 clean records")
        
        with col3:
            st.write("**Prediction Speed**")
            st.write("Single: <1ms")
            st.write("Batch (100): <100ms")
            st.write("Batch (1000): <1s")

# ============================================================================
# PAGE 5: ABOUT
# ============================================================================

elif page == "ℹ️ About":
    render_hero("ℹ️ About This Application", "Project overview, how to use the app, and documentation links")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Project Overview")
        st.write("""
        **Flight Delay Prediction System** is an end-to-end machine learning 
        solution for predicting flight arrival delays with high accuracy.
        
        **Dataset**: 539,747 flight records
        **Target**: Arrival delay > 15 minutes
        **Features**: 13 engineered features
        **Model**: Gradient Boosting Classifier
        """)
        
        st.subheader("📊 Model Performance")
        st.write("""
        - **Accuracy**: 95.27%
        - **Precision**: 91.22%
        - **Recall**: 82.23%
        - **F1-Score**: 0.8649
        - **ROC-AUC**: 0.9803
        """)
    
    with col2:
        st.subheader("✈️ Key Features")
        st.write("""
        ✅ **Real-time Predictions**
        Make immediate predictions for single flights
        
        ✅ **Batch Processing**
        Process up to 1000 flights in one go
        
        ✅ **Detailed Explanations**
        Understand why each prediction was made
        
        ✅ **Feature Analysis**
        See which factors matter most
        """)
    
    st.markdown("---")
    
    st.subheader("🔧 Technology Stack")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Backend**")
        st.write("- FastAPI")
        st.write("- Uvicorn")
        st.write("- Scikit-learn")
    
    with col2:
        st.write("**Frontend**")
        st.write("- Streamlit")
        st.write("- Plotly")
        st.write("- Pandas")
    
    with col3:
        st.write("**Data**")
        st.write("- CSV Format")
        st.write("- Pandas DataFrames")
        st.write("- Model Persistence")
    
    st.markdown("---")
    
    st.subheader("📚 How to Use")
    
    st.write("""
    1. **Single Prediction**: Go to "Predict Flight" to check a single flight
    2. **Batch Predictions**: Upload a CSV file with multiple flights
    3. **View Analysis**: Check model performance and feature importance
    4. **Download Results**: Export predictions as CSV for further analysis
    """)
    
    st.markdown("---")
    
    st.subheader("🚀 Getting Started")
    
    st.write("""
    **Make sure the API is running:**
    ```bash
    python flight_delay_api.py
    ```
    
    **Then run this Streamlit app:**
    ```bash
    streamlit run streamlit_app.py
    ```
    
    The app will open at `http://localhost:8501`
    """)
    
    st.markdown("---")
    
    st.subheader("📖 Documentation")
    
    st.write("""
    - **API Documentation**: See `API_DOCUMENTATION.md`
    - **Analysis Report**: See `ANALYSIS_REPORT.md`
    - **Setup Guide**: See `README_API.md`
    """)
    
    st.markdown("---")
    
    st.info("✨ Flight Delay Prediction System v1.0.0 - Production Ready")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center;">
    <p>Flight Delay Prediction API • FastAPI + Streamlit + Scikit-learn</p>
    <p>Built with ❤️ | March 2026</p>
</div>
""", unsafe_allow_html=True)
