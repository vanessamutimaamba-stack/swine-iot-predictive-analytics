"""
Swine IoT Analytics: Predictive Health & Water Consumption
===================================================
An end-to-end predictive IoT analytics pipeline for commercial swine operations.
Detects subclinical illness and infrastructure failures through water consumption analysis.

Phases:
1. Data Simulation & Feature Engineering
2. Statistical Anomaly Detection (Rolling Z-Score)
3. Machine Learning Anomaly Detection (Isolation Forest)
4. Time Series Forecasting (Facebook Prophet)
5. Multivariate Anomaly Detection
6. Financial Impact & ROI Analysis
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from prophet import Prophet


# ============================================================================
# PHASE 1: DATA SIMULATION & FEATURE ENGINEERING
# ============================================================================

def simulate_barn_data(days=120, seed=42):
    """
    Simulate a standard 120-day grow-finish cycle for a 1,000-head swine barn.
    
    Accounts for:
    - Biological Growth: Natural scaling of water intake as pigs grow
    - Daily Variance: Natural fluctuations in drinking behavior
    - Mechanical Anomalies: Sudden spike (burst water line)
    - Biological Anomalies: Progressive drop (subclinical disease)
    - Hardware Anomalies: Complete sensor failure
    """
    np.random.seed(seed)
    dates = pd.date_range(start='2026-01-01', periods=days, freq='D')
    
    # Baseline consumption: ~2L/day (weaner) to ~9L/day (finisher)
    time_index = np.arange(days)
    base_consumption = 2 + 7 * (time_index / days) ** 1.5
    
    # Add natural daily variance
    daily_noise = np.random.normal(loc=0, scale=0.6, size=days)
    water_intake = np.maximum(base_consumption + daily_noise, 0.5)
    
    # Inject anomalies
    water_intake[35] += 12.0  # Water line leak (Mechanical)
    
    disease_start = 75
    for i in range(6):
        water_intake[disease_start + i] *= (0.8 - (i * 0.05))  # Disease progression
    
    water_intake[105] = 0.0  # Sensor failure (Hardware)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Barn_ID': 'Barn_A',
        'Avg_Daily_Intake_Liters': np.round(water_intake, 2)
    })
    
    return df


def engineer_features(df):
    """Add environmental and domain-specific features."""
    df['Date'] = pd.to_datetime(df['Date'])
    
    np.random.seed(42)
    num_days = len(df)
    
    # Simulate temperature and humidity
    base_temp = np.linspace(10, 25, num_days)
    df['Temperature_C'] = base_temp + np.random.normal(loc=0, scale=2.5, size=num_days)
    df['Humidity_Percent'] = np.random.uniform(low=40, high=80, size=num_days)
    
    # Calculate THI (Temperature-Humidity Index)
    df['THI_Heat_Stress'] = (0.8 * df['Temperature_C']) + \
                            ((df['Humidity_Percent'] / 100) * (df['Temperature_C'] - 14.4)) + 46.4
    
    df['Is_Heat_Stressed'] = (df['THI_Heat_Stress'] > 72).astype(int)
    
    # Time-based features
    df['Day_of_Week'] = df['Date'].dt.day_name()
    df['Is_Weekend'] = df['Date'].dt.dayofweek.isin([5, 6]).astype(int)
    
    return df


# ============================================================================
# PHASE 2: STATISTICAL ANOMALY DETECTION (ROLLING Z-SCORE)
# ============================================================================

def rolling_zscore_detection(df, window_size=7, z_threshold=2.5):
    """Detect anomalies using rolling 7-day Z-score."""
    df['Rolling_Mean'] = df['Avg_Daily_Intake_Liters'].rolling(window=window_size).mean().shift(1)
    df['Rolling_Std'] = df['Avg_Daily_Intake_Liters'].rolling(window=window_size).std().shift(1)
    df['Z_Score'] = (df['Avg_Daily_Intake_Liters'] - df['Rolling_Mean']) / df['Rolling_Std']
    df['Anomaly_Flag'] = df['Z_Score'].abs() > z_threshold
    
    return df


# ============================================================================
# PHASE 3: MACHINE LEARNING ANOMALY DETECTION (ISOLATION FOREST)
# ============================================================================

def ml_anomaly_detection(df):
    """Detect anomalies using Isolation Forest."""
    model_df = df.dropna(subset=['Rolling_Mean', 'Avg_Daily_Intake_Liters']).copy()
    X = model_df[['Avg_Daily_Intake_Liters', 'Rolling_Mean']]
    
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_forest.fit(X)
    
    model_df['ML_Anomaly'] = iso_forest.predict(X)
    model_df['ML_Anomaly_Flag'] = model_df['ML_Anomaly'] == -1
    
    return model_df


# ============================================================================
# PHASE 4: TIME SERIES FORECASTING (FACEBOOK PROPHET)
# ============================================================================

def prophet_forecasting(df, periods=7):
    """Generate time series forecast using Facebook Prophet."""
    prophet_df = df[['Date', 'Avg_Daily_Intake_Liters']].rename(
        columns={'Date': 'ds', 'Avg_Daily_Intake_Liters': 'y'}
    )
    
    m = Prophet(yearly_seasonality=False, daily_seasonality=False, changepoint_prior_scale=0.05)
    m.fit(prophet_df)
    
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    
    return m, forecast


# ============================================================================
# PHASE 5: MULTIVARIATE ANOMALY DETECTION
# ============================================================================

def multivariate_ml_detection(df):
    """Detect anomalies using multiple features with feature scaling."""
    features = [
        'Avg_Daily_Intake_Liters', 
        'Rolling_Mean', 
        'Temperature_C', 
        'Humidity_Percent', 
        'THI_Heat_Stress', 
        'Is_Weekend'
    ]
    
    multi_model_df = df.dropna(subset=features).copy()
    X_multi = multi_model_df[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_multi)
    
    iso_forest_multi = IsolationForest(contamination=0.05, random_state=42)
    iso_forest_multi.fit(X_scaled)
    
    multi_model_df['Multi_ML_Anomaly'] = iso_forest_multi.predict(X_scaled)
    multi_model_df['Multi_ML_Anomaly_Flag'] = multi_model_df['Multi_ML_Anomaly'] == -1
    
    return multi_model_df


# ============================================================================
# PHASE 6: FINANCIAL IMPACT & ROI ANALYSIS
# ============================================================================

def calculate_roi_metrics():
    """Calculate business impact and ROI of predictive system."""
    # Cost parameters (industry estimates)
    COST_PER_LITER_WATER_WASTE = 0.05
    MARKET_VALUE_PER_PIG = 180.00
    TREATMENT_COST_PER_PIG = 8.50
    BARN_CAPACITY = 1000
    
    # Detection scenarios
    days_lost_traditional = 4
    mortality_rate_traditional = 0.03
    
    days_lost_ml = 1
    mortality_rate_ml = 0.005
    
    # Calculate financial impact
    leak_volume_above_baseline = 14.37 - 2.93
    water_leak_cost = leak_volume_above_baseline * BARN_CAPACITY * COST_PER_LITER_WATER_WASTE
    
    traditional_disease_cost = (BARN_CAPACITY * mortality_rate_traditional * MARKET_VALUE_PER_PIG) + \
                              (BARN_CAPACITY * TREATMENT_COST_PER_PIG)
    
    ml_disease_cost = (BARN_CAPACITY * mortality_rate_ml * MARKET_VALUE_PER_PIG) + \
                     (BARN_CAPACITY * TREATMENT_COST_PER_PIG)
    
    total_cost_traditional = water_leak_cost + traditional_disease_cost
    total_cost_ml = water_leak_cost + ml_disease_cost
    total_savings = total_cost_traditional - total_cost_ml
    
    system_cost = 1500.00
    net_profit_increase = total_savings - system_cost
    roi_percentage = (net_profit_increase / system_cost) * 100
    
    return {
        'water_leak_cost': water_leak_cost,
        'traditional_cost': total_cost_traditional,
        'ml_cost': total_cost_ml,
        'savings': total_savings,
        'system_cost': system_cost,
        'net_profit': net_profit_increase,
        'roi': roi_percentage
    }


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_raw_data(df):
    """Visualize raw simulated water consumption data."""
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Avg_Daily_Intake_Liters'], marker='o', linestyle='-', 
             color='#1f77b4', markersize=4)
    plt.title('Simulated Barn Water Consumption (120-Day Cycle)', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Average Daily Intake per Pig (Liters)', fontsize=12)
    
    plt.axvline(x=df['Date'].iloc[35], color='red', linestyle='--', alpha=0.5, 
                label='Leak (Mechanical)')
    plt.axvline(x=df['Date'].iloc[75], color='orange', linestyle='--', alpha=0.5, 
                label='Health Drop (Biological)')
    plt.axvline(x=df['Date'].iloc[105], color='gray', linestyle='--', alpha=0.5, 
                label='Sensor Drop (Hardware)')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('phase1_raw_data.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: phase1_raw_data.png")
    plt.close()


def plot_zscore_detection(df):
    """Visualize rolling Z-score anomaly detection."""
    plt.figure(figsize=(14, 7))
    
    plt.plot(df['Date'], df['Avg_Daily_Intake_Liters'], label='Actual Intake', 
             color='#1f77b4', marker='o', markersize=4)
    plt.plot(df['Date'], df['Rolling_Mean'], label='7-Day Expected Baseline', 
             color='green', linestyle='--')
    
    plt.fill_between(df['Date'], 
                     df['Rolling_Mean'] - (2.5 * df['Rolling_Std']),
                     df['Rolling_Mean'] + (2.5 * df['Rolling_Std']),
                     color='green', alpha=0.15, label='Expected Range (±2.5 Z)')
    
    anomalies = df[df['Anomaly_Flag'] == True]
    plt.scatter(anomalies['Date'], anomalies['Avg_Daily_Intake_Liters'], 
                color='red', s=100, zorder=5, label='System Alert Triggered')
    
    plt.title('Predictive Herd Health: Rolling Z-Score Anomaly Detection', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Average Daily Intake per Pig (Liters)', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('phase2_zscore_detection.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: phase2_zscore_detection.png")
    plt.close()


def plot_ml_detection(model_df):
    """Visualize Isolation Forest anomaly detection."""
    plt.figure(figsize=(14, 7))
    
    plt.plot(model_df['Date'], model_df['Avg_Daily_Intake_Liters'], 
             label='Actual Intake', color='#1f77b4', marker='o', markersize=4)
    plt.plot(model_df['Date'], model_df['Rolling_Mean'], 
             label='7-Day Expected Baseline', color='green', linestyle='--', alpha=0.6)
    
    ml_anomalies = model_df[model_df['ML_Anomaly_Flag']]
    plt.scatter(ml_anomalies['Date'], ml_anomalies['Avg_Daily_Intake_Liters'], 
                color='purple', s=100, zorder=5, label='Isolation Forest Alert')
    
    plt.title('Advanced Analytics: Isolation Forest Anomaly Detection', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Average Daily Intake per Pig (Liters)', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('phase3_ml_detection.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: phase3_ml_detection.png")
    plt.close()


def plot_prophet_forecast(m, forecast):
    """Visualize Prophet time series forecast."""
    fig = m.plot(forecast, figsize=(12, 6))
    plt.title('Time Series Forecasting: Predicting Future Water Consumption with Prophet', 
              fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Average Daily Intake per Pig (Liters)', fontsize=12)
    ax = fig.gca()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('phase4_prophet_forecast.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: phase4_prophet_forecast.png")
    plt.close()


def plot_multivariate_detection(multi_model_df):
    """Visualize multivariate anomaly detection."""
    plt.figure(figsize=(14, 7))
    
    plt.plot(multi_model_df['Date'], multi_model_df['Avg_Daily_Intake_Liters'], 
             label='Actual Intake', color='#1f77b4', marker='o', markersize=4)
    plt.plot(multi_model_df['Date'], multi_model_df['Rolling_Mean'], 
             label='7-Day Expected Baseline', color='green', linestyle='--', alpha=0.6)
    
    multi_anomalies = multi_model_df[multi_model_df['Multi_ML_Anomaly_Flag']]
    plt.scatter(multi_anomalies['Date'], multi_anomalies['Avg_Daily_Intake_Liters'], 
                color='red', s=100, zorder=5, label='Multivariate Anomaly Alert')
    
    plt.title('Advanced Multivariate Anomaly Detection (Intake + Environment)', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Average Daily Intake per Pig (Liters)', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('phase5_multivariate_detection.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: phase5_multivariate_detection.png")
    plt.close()


# ============================================================================
# REPORTING FUNCTIONS
# ============================================================================

def print_roi_report(metrics):
    """Print formatted ROI and financial impact report."""
    print("\n" + "="*70)
    print("     AGRIBUSINESS FINANCIAL IMPACT & ROI REPORT                   ")
    print("="*70)
    print(f"Simulated Barn Size:          1,000 Pigs")
    print(f"Cost of Water Leak (1 Day):   ${metrics['water_leak_cost']:,.2f}")
    print("-"*70)
    print(f"Traditional Management Cost:  ${metrics['traditional_cost']:,.2f} (Delayed intervention)")
    print(f"Predictive IoT Barn Cost:     ${metrics['ml_cost']:,.2f} (Immediate intervention)")
    print("-"*70)
    print(f"Gross Financial Savings:      ${metrics['savings']:,.2f}")
    print(f"Annual IoT System Investment: ${metrics['system_cost']:,.2f}")
    print(f"Net Profit Saved per Barn:    ${metrics['net_profit']:,.2f}")
    print(f"PROJECTED SYSTEM ROI:         {metrics['roi']:.1f}%")
    print("="*70 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Execute the complete swine IoT analytics pipeline."""
    
    print("\n" + "="*70)
    print("    SWINE IoT ANALYTICS: PREDICTIVE HEALTH & WATER CONSUMPTION      ")
    print("="*70)
    
    # Create output directory
    os.makedirs('data', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    # Phase 1: Data Simulation & Feature Engineering
    print("\n[Phase 1] Simulating barn data and engineering features...")
    df = simulate_barn_data(days=120)
    df = engineer_features(df)
    print("✓ Data simulation complete")
    
    # Phase 2: Statistical Anomaly Detection
    print("\n[Phase 2] Applying rolling Z-score anomaly detection...")
    df = rolling_zscore_detection(df)
    anomalies_zscore = df[df['Anomaly_Flag'] == True]
    print(f"✓ Detected {len(anomalies_zscore)} anomalies via Z-score")
    
    # Phase 3: ML Anomaly Detection
    print("\n[Phase 3] Applying Isolation Forest ML detection...")
    model_df = ml_anomaly_detection(df)
    anomalies_ml = model_df[model_df['ML_Anomaly_Flag']]
    print(f"✓ Detected {len(anomalies_ml)} anomalies via Isolation Forest")
    
    # Phase 4: Time Series Forecasting
    print("\n[Phase 4] Generating time series forecast with Prophet...")
    m, forecast = prophet_forecasting(df, periods=7)
    print("✓ Forecast generated for next 7 days")
    
    # Phase 5: Multivariate Detection
    print("\n[Phase 5] Applying multivariate anomaly detection...")
    multi_model_df = multivariate_ml_detection(df)
    anomalies_multi = multi_model_df[multi_model_df['Multi_ML_Anomaly_Flag']]
    print(f"✓ Detected {len(anomalies_multi)} anomalies via multivariate model")
    
    # Phase 6: ROI Analysis
    print("\n[Phase 6] Calculating financial impact and ROI...")
    roi_metrics = calculate_roi_metrics()
    print_roi_report(roi_metrics)
    
    # Generate Visualizations
    print("\n[Visualization] Generating charts...")
    plot_raw_data(df)
    plot_zscore_detection(df)
    plot_ml_detection(model_df)
    plot_prophet_forecast(m, forecast)
    plot_multivariate_detection(multi_model_df)
    
    # Export Results
    print("\n[Export] Saving results to CSV...")
    df.to_csv('data/processed_barn_alerts.csv', index=False)
    model_df.to_csv('data/ml_anomalies.csv', index=False)
    forecast.to_csv('data/forecast.csv', index=False)
    multi_model_df.to_csv('data/multivariate_anomalies.csv', index=False)
    print("✓ Results exported to data/ directory")
    
    print("\n" + "="*70)
    print("    Pipeline execution complete!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
    streamlit
pandas
numpy
matplotlib
scikit-learn
prophet
