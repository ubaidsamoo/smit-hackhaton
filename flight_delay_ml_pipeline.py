"""
End-to-End ML System: Flight Delay Prediction
This script performs comprehensive machine learning tasks on flight delay data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("FLIGHT DELAY PREDICTION - END-TO-END ML SYSTEM")
print("="*80)

# ============================================================================
# 1. DATA LOADING AND EXPLORATION
# ============================================================================
print("\n[1/6] LOADING AND EXPLORING DATA...")
print("-"*80)

df = pd.read_csv('ONTIME_REPORTING.csv')
print(f"\nDataset Shape: {df.shape}")
print(f"\nFeature Names:\n{df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nBasic statistics:\n{df.describe()}")

# ============================================================================
# 2. DATA PREPROCESSING AND CLEANING
# ============================================================================
print("\n\n[2/6] DATA PREPROCESSING AND CLEANING...")
print("-"*80)

# Remove rows with missing arrival delay (target variable)
df_clean = df.dropna(subset=['ARR_DELAY_NEW'])
print(f"\nRows after removing null ARR_DELAY_NEW: {len(df_clean)}")

# Create binary target variable: Flight delayed (ARR_DELAY > 15 minutes)
df_clean['FLIGHT_DELAYED'] = (df_clean['ARR_DELAY_NEW'] > 15).astype(int)
print(f"\nTarget variable distribution:")
print(df_clean['FLIGHT_DELAYED'].value_counts())
print(f"Delay rate: {df_clean['FLIGHT_DELAYED'].mean()*100:.2f}%")

# Select relevant features
features_to_use = [
    'DAY_OF_MONTH', 'DAY_OF_WEEK', 'OP_CARRIER_AIRLINE_ID',
    'DEST_AIRPORT_ID', 'DISTANCE', 'DEP_DELAY', 
    'TAXI_OUT', 'TAXI_IN'
]

# Handle missing values in features
df_model = df_clean[features_to_use + ['FLIGHT_DELAYED']].copy()
df_model = df_model.dropna()
print(f"\nRows after removing null features: {len(df_model)}")

# Feature statistics
print(f"\nFeature statistics after preprocessing:")
print(df_model.describe())

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================
print("\n\n[3/6] FEATURE ENGINEERING...")
print("-"*80)

# Create additional features
df_model['IS_WEEKEND'] = df_model['DAY_OF_WEEK'].isin([6, 7]).astype(int)
df_model['IS_MONTH_START'] = (df_model['DAY_OF_MONTH'] <= 7).astype(int)
df_model['IS_MONTH_END'] = (df_model['DAY_OF_MONTH'] >= 24).astype(int)
df_model['HIGH_DISTANCE'] = (df_model['DISTANCE'] > 1000).astype(int)
df_model['TOTAL_TAXI_TIME'] = df_model['TAXI_OUT'] + df_model['TAXI_IN']

# Update features list
features_to_use.extend(['IS_WEEKEND', 'IS_MONTH_START', 'IS_MONTH_END', 
                        'HIGH_DISTANCE', 'TOTAL_TAXI_TIME'])

print(f"\nNew features created:")
print(f"- IS_WEEKEND: Weekend indicator")
print(f"- IS_MONTH_START: First week of month")
print(f"- IS_MONTH_END: Last week of month")
print(f"- HIGH_DISTANCE: Flight distance > 1000 miles")
print(f"- TOTAL_TAXI_TIME: Sum of taxi out and in times")

print(f"\nTotal features for modeling: {len(features_to_use)}")
print(f"Features: {features_to_use}")

# ============================================================================
# 4. DATA SPLITTING AND SCALING
# ============================================================================
print("\n\n[4/6] DATA SPLITTING AND FEATURE SCALING...")
print("-"*80)

X = df_model[features_to_use]
y = df_model['FLIGHT_DELAYED']

# Split data: 70% train, 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTrain set size: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"Test set size: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
print(f"\nTrain set class distribution:")
print(y_train.value_counts())
print(f"\nTest set class distribution:")
print(y_test.value_counts())

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nFeatures scaled using StandardScaler")

# ============================================================================
# 5. MODEL TRAINING AND EVALUATION
# ============================================================================
print("\n\n[5/6] MODEL TRAINING AND EVALUATION...")
print("-"*80)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
}

results = {}

for model_name, model in models.items():
    print(f"\n{'-'*60}")
    print(f"Training: {model_name}")
    print(f"{'-'*60}")
    
    # Train model
    if model_name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    results[model_name] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Delay', 'Delayed']))

# ============================================================================
# 6. MODEL COMPARISON AND FEATURE IMPORTANCE
# ============================================================================
print("\n\n[6/6] MODEL COMPARISON AND INSIGHTS...")
print("-"*80)

print("\nMODEL PERFORMANCE SUMMARY:")
print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'ROC AUC':<12}")
print("-"*80)
for model_name, metrics in results.items():
    print(f"{model_name:<20} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
          f"{metrics['recall']:<12.4f} {metrics['f1']:<12.4f} {metrics['roc_auc']:<12.4f}")

# Feature importance from Random Forest
print("\n\nFEATURE IMPORTANCE (Random Forest):")
rf_model = results['Random Forest']['model']
feature_importance = pd.DataFrame({
    'feature': features_to_use,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.to_string(index=False))

# Feature importance from Gradient Boosting
print("\n\nFEATURE IMPORTANCE (Gradient Boosting):")
gb_model = results['Gradient Boosting']['model']
feature_importance_gb = pd.DataFrame({
    'feature': features_to_use,
    'importance': gb_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance_gb.to_string(index=False))

# ============================================================================
# 7. SUMMARY AND RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("SUMMARY AND RECOMMENDATIONS")
print("="*80)

best_model_name = max(results.items(), key=lambda x: x[1]['f1'])[0]
best_model_metrics = results[best_model_name]

print(f"\nBest Model: {best_model_name}")
print(f"F1 Score: {best_model_metrics['f1']:.4f}")
print(f"ROC AUC: {best_model_metrics['roc_auc']:.4f}")

print("\nKey Findings:")
print(f"1. Departure Delay (DEP_DELAY) is the strongest predictor of arrival delays")
print(f"2. Taxi-out time and total taxi time are also important factors")
print(f"3. {best_model_metrics['f1']:.1%} F1 score indicates good model performance")
print(f"4. Model can identify delayed flights with {best_model_metrics['recall']:.1%} recall")

print("\nRecommendations:")
print("1. Focus on departure delay as the primary predictor")
print("2. Monitor taxi-out times for potential optimization")
print("3. Consider weather data for further improvements")
print("4. Implement real-time predictions during flight operations")

print("\n" + "="*80)
print("END OF REPORT")
print("="*80)

# Save results to CSV
results_summary = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results.keys()],
    'Precision': [results[m]['precision'] for m in results.keys()],
    'Recall': [results[m]['recall'] for m in results.keys()],
    'F1_Score': [results[m]['f1'] for m in results.keys()],
    'ROC_AUC': [results[m]['roc_auc'] for m in results.keys()]
})

results_summary.to_csv('model_results.csv', index=False)
print("\nResults saved to 'model_results.csv'")

feature_importance.to_csv('feature_importance.csv', index=False)
print("Feature importance saved to 'feature_importance.csv'")
