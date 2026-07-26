"""
Swine IoT Analytics: Predictive Health & Water Consumption
===================================================
Streamlit Web App for predictive IoT analytics pipeline
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
import warnings

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Swine IoT Analytics",
    page_icon="🐷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# PHASE 1: DATA SIMULATION & FEATURE ENGINEERING
# ============================================================================

def simulate_barn_data(days=120, seed=42):
    """
    Simulate a standard 120-day grow-finish cycle for a 1,000-head swine barn.
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
# PHASE 2: STATISTICAL ANOMALY DETECTION
# ============================================================================

def rolling_zscore_detection(df, window_size=7, z_threshold=2.5):
    """Detect anomalies using rolling 7-day Z-score."""
    df['Rolling_Mean'] = df['Avg_Daily_Intake_Liters'].rolling(window=window_size).mean().shift(1)
    df['Rolling_Std'] = df['Avg_Daily_Intake_Liters'].rolling(window=window_size).std().shift(1)
    df['Z_Score'] = (df['Avg_Daily_Intake_Liters'] - df['Rolling_Mean']) / df['Rolling_Std']
    df['Anomaly_Flag'] = df['Z_Score'].abs() > z_threshold
    
    return df


# ============================================================================
# PHASE 3: MACHINE LEARNING ANOMALY DETECTION
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
# PHASE 4: TIME SERIES FORECASTING
# ============================================================================

def prophet_forecasting(df, periods=7):
    """Generate time series forecast using Facebook Prophet."""
    prophet_df = df[['Date', 'Avg_Daily_Intake_Liters']].rename(
        columns={'Date': 'ds', 'Avg_Daily_Intake_Liters': 'y'}
    )
    
    with st.spinner('Training Prophet model...'):
        m = Prophet(yearly_seasonality=False, daily_seasonality=False, 
                   changepoint_prior_scale=0.05)
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
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['Date'], df['Avg_Daily_Intake_Liters'], marker='o', linestyle='-', 
            color='#1f77b4', markersize=4)
    ax.set_title('Simulated Barn Water Consumption (120-Day Cycle)', fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Average Daily Intake per Pig (Liters)', fontsize=12)
    
    ax.axvline(x=df['Date'].iloc[35], color='red', linestyle='--', alpha=0.5, 
               label='Leak (Mechanical)')
    ax.axvline(x=df['Date'].iloc[75], color='orange', linestyle='--', alpha=0.5, 
               label='Health Drop (Biological)')
    ax.axvline(x=df['Date'].iloc[105], color='gray', linestyle='--', alpha=0.5, 
               label='Sensor Drop (Hardware)')
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_zscore_detection(df):
    """Visualize rolling Z-score anomaly detection."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(df['Date'], df['Avg_Daily_Intake_Liters'], label='Actual Intake', 
            color='#1f77b4', marker='o', markersize=4)
    ax.plot(df['Date'], df['Rolling_Mean'], label='7-Day Expected Baseline', 
            color='green', linestyle='--')
    
    ax.fill_between(df['Date'], 
                    df['Rolling_Mean'] - (2.5 * df['Rolling_Std']),
                    df['Rolling_Mean'] + (2.5 * df['Rolling_Std']),
                    color='green', alpha=0.15, label='Expected Range (±2.5 Z)')
    
    anomalies = df[df['Anomaly_Flag'] == True]
    ax.scatter(anomalies['Date'], anomalies['Avg_Daily_Intake_Liters'], 
               color='red', s=100, zorder=5, label='System Alert Triggered')
    
    ax.set_title('Predictive Herd Health: Rolling Z-Score Anomaly Detection', fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Average Daily Intake per Pig (Liters)', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_ml_detection(model_df):
    """Visualize Isolation Forest anomaly detection."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(model_df['Date'], model_df['Avg_Daily_Intake_Liters'], 
            label='Actual Intake', color='#1f77b4', marker='o', markersize=4)
    ax.plot(model_df['Date'], model_df['Rolling_Mean'], 
            label='7-Day Expected Baseline', color='green', linestyle='--', alpha=0.6)
    
    ml_anomalies = model_df[model_df['ML_Anomaly_Flag']]
    ax.scatter(ml_anomalies['Date'], ml_anomalies['Avg_Daily_Intake_Liters'], 
               color='purple', s=100, zorder=5, label='Isolation Forest Alert')
    
    ax.set_title('Advanced Analytics: Isolation Forest Anomaly Detection', fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Average Daily Intake per Pig (Liters)', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


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
    return fig


def plot_multivariate_detection(multi_model_df):
    """Visualize multivariate anomaly detection."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(multi_model_df['Date'], multi_model_df['Avg_Daily_Intake_Liters'], 
            label='Actual Intake', color='#1f77b4', marker='o', markersize=4)
    ax.plot(multi_model_df['Date'], multi_model_df['Rolling_Mean'], 
            label='7-Day Expected Baseline', color='green', linestyle='--', alpha=0.6)
    
    multi_anomalies = multi_model_df[multi_model_df['Multi_ML_Anomaly_Flag']]
    ax.scatter(multi_anomalies['Date'], multi_anomalies['Avg_Daily_Intake_Liters'], 
               color='red', s=100, zorder=5, label='Multivariate Anomaly Alert')
    
    ax.set_title('Advanced Multivariate Anomaly Detection (Intake + Environment)', fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Average Daily Intake per Pig (Liters)', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ============================================================================
# STREAMLIT APP LAYOUT
# ============================================================================

def main():
    # Header
    st.markdown("""
    # 🐷 Swine IoT Predictive Analytics
    **Advanced Health & Water Consumption Monitoring System**
    """)
    
    st.markdown("""
    An end-to-end predictive IoT analytics pipeline for commercial swine operations.
    Detects subclinical illness and infrastructure failures through water consumption analysis.
    """)
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    simulation_days = st.sidebar.slider("Days to Simulate", 30, 365, 120)
    z_threshold = st.sidebar.slider("Z-Score Threshold", 1.0, 4.0, 2.5)
    forecast_days = st.sidebar.slider("Forecast Days", 3, 30, 7)
    
    run_analysis = st.sidebar.button("▶️ Run Full Analysis", use_container_width=True)
    
    if run_analysis:
        st.balloons()
        
        # Phase 1
        st.header("📊 Phase 1: Data Simulation & Feature Engineering")
        with st.spinner('Simulating barn data...'):
            df = simulate_barn_data(days=simulation_days)
            df = engineer_features(df)
            st.success('✓ Data simulation complete')
            st.write(f"Generated {len(df)} days of data for a 1,000-head barn")
        
        # Phase 2
        st.header("📈 Phase 2: Statistical Anomaly Detection (Z-Score)")
        with st.spinner('Applying rolling Z-score detection...'):
            df = rolling_zscore_detection(df, z_threshold=z_threshold)
            anomalies_zscore = df[df['Anomaly_Flag'] == True]
            st.success(f'✓ Detected {len(anomalies_zscore)} anomalies via Z-score')
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Z-Score Anomalies", len(anomalies_zscore))
        with col2:
            st.metric("Detection Rate", f"{(len(anomalies_zscore) / len(df) * 100):.1f}%")
        
        fig = plot_zscore_detection(df)
        st.pyplot(fig)
        
        # Phase 3
        st.header("🤖 Phase 3: Machine Learning Anomaly Detection (Isolation Forest)")
        with st.spinner('Applying Isolation Forest...'):
            model_df = ml_anomaly_detection(df)
            anomalies_ml = model_df[model_df['ML_Anomaly_Flag']]
            st.success(f'✓ Detected {len(anomalies_ml)} anomalies via ML')
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("ML Anomalies", len(anomalies_ml))
        with col2:
            st.metric("Detection Rate", f"{(len(anomalies_ml) / len(model_df) * 100):.1f}%")
        
        fig = plot_ml_detection(model_df)
        st.pyplot(fig)
        
        # Phase 4
        st.header("🔮 Phase 4: Time Series Forecasting (Prophet)")
        with st.spinner('Training Prophet model...'):
            m, forecast = prophet_forecasting(df, periods=forecast_days)
            st.success('✓ Forecast generated')
        
        fig = plot_prophet_forecast(m, forecast)
        st.pyplot(fig)
        
        # Phase 5
        st.header("🔬 Phase 5: Multivariate Anomaly Detection")
        with st.spinner('Applying multivariate analysis...'):
            multi_model_df = multivariate_ml_detection(df)
            anomalies_multi = multi_model_df[multi_model_df['Multi_ML_Anomaly_Flag']]
            st.success(f'✓ Detected {len(anomalies_multi)} multivariate anomalies')
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Multivariate Anomalies", len(anomalies_multi))
        with col2:
            st.metric("Detection Rate", f"{(len(anomalies_multi) / len(multi_model_df) * 100):.1f}%")
        
        fig = plot_multivariate_detection(multi_model_df)
        st.pyplot(fig)
        
        # Phase 6
        st.header("💰 Phase 6: Financial Impact & ROI Analysis")
        with st.spinner('Calculating ROI metrics...'):
            roi_metrics = calculate_roi_metrics()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Gross Savings", f"${roi_metrics['savings']:,.0f}")
        with col2:
            st.metric("System Cost", f"${roi_metrics['system_cost']:,.0f}")
        with col3:
            st.metric("Net Profit", f"${roi_metrics['net_profit']:,.0f}")
        with col4:
            st.metric("ROI", f"{roi_metrics['roi']:.1f}%", delta="Annual")
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Traditional Management")
            st.write(f"**Cost:** ${roi_metrics['traditional_cost']:,.2f}")
            st.caption("Delayed disease detection (4+ days)")
        
        with col2:
            st.markdown("### Predictive IoT System")
            st.write(f"**Cost:** ${roi_metrics['ml_cost']:,.2f}")
            st.caption("Immediate intervention (1 day)")
        
        # Data Export
        st.header("📥 Download Results")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📊 Z-Score Data",
                data=csv,
                file_name="zscore_data.csv",
                mime="text/csv"
            )
        
        with col2:
            csv = model_df.to_csv(index=False)
            st.download_button(
                label="🤖 ML Anomalies",
                data=csv,
                file_name="ml_anomalies.csv",
                mime="text/csv"
            )
        
        with col3:
            csv = forecast.to_csv(index=False)
            st.download_button(
                label="🔮 Forecast Data",
                data=csv,
                file_name="forecast.csv",
                mime="text/csv"
            )
        
        with col4:
            csv = multi_model_df.to_csv(index=False)
            st.download_button(
                label="🔬 Multivariate Data",
                data=csv,
                file_name="multivariate_data.csv",
                mime="text/csv"
            )
    
    else:
        st.info("👈 Click **Run Full Analysis** to start the pipeline")
        
        st.markdown("""
        ### 📋 Pipeline Phases
        1. **Data Simulation** - Generate realistic barn water consumption data
        2. **Z-Score Detection** - Statistical anomaly detection
        3. **Isolation Forest** - Machine learning anomaly detection
        4. **Prophet Forecast** - Time series prediction
        5. **Multivariate Detection** - Multi-feature anomaly analysis
        6. **ROI Analysis** - Financial impact calculation
        """)


if __name__ == '__main__':
    main()
