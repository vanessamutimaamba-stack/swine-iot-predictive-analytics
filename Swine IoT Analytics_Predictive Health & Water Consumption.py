import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Setup Simulation Parameters
days = 120  # Standard grow-finish cycle
dates = pd.date_range(start='2026-01-01', periods=days, freq='D')
np.random.seed(42)  # Ensures you get the same random numbers every time you run it

# 2. Baseline Consumption (The Biological Growth Curve)
# Simulating a curve that starts at ~2L/day (weaner) and scales to ~9L/day (finisher)
time_index = np.arange(days)
# A polynomial curve mimics the biological growth phase
base_consumption = 2 + 7 * (time_index / days)**1.5 

# 3. Add Natural Daily Variance
# Pigs don't drink the exact same amount every day due to temperature, activity, etc.
daily_noise = np.random.normal(loc=0, scale=0.6, size=days) 
water_intake = np.maximum(base_consumption + daily_noise, 0.5)

# 4. Inject Barn-Specific Anomalies
# Event A: Water Line Leak (Mechanical) -> Sudden massive spike
water_intake[35] += 12.0  

# Event B: Sub-clinical Disease Outbreak (Biological) -> Multi-day progressive drop
# This is the most important pattern for your model to catch
disease_start = 75
for i in range(6):
    water_intake[disease_start + i] *= (0.8 - (i * 0.05)) # Progressive reduction

# Event C: Sensor Failure or Washdown (Hardware/Management) -> Complete drop to 0
water_intake[105] = 0.0

# 5. Compile into a DataFrame
df = pd.DataFrame({
    'Date': dates,
    'Barn_ID': 'Barn_A',
    'Avg_Daily_Intake_Liters': np.round(water_intake, 2)
})

# 6. Visualize the Raw Data
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Avg_Daily_Intake_Liters'], marker='o', linestyle='-', color='#1f77b4', markersize=4)
plt.title('Simulated Barn Water Consumption (120-Day Cycle)', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Average Daily Intake per Pig (Liters)', fontsize=12)

# Highlight the anomalies for reference
plt.axvline(x=df['Date'].iloc[35], color='red', linestyle='--', alpha=0.5, label='Leak (Mechanical)')
plt.axvline(x=df['Date'].iloc[75], color='orange', linestyle='--', alpha=0.5, label='Health Drop (Biological)')
plt.axvline(x=df['Date'].iloc[105], color='gray', linestyle='--', alpha=0.5, label='Sensor Drop (Hardware)')

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Display the first few rows
df.head()

# ---
# #Phase 2: Statistical Anomaly Detection (Rolling Z-Score)
# To establish a baseline detection engine, we use a rolling 7-day Z-score. This method smooths out day-to-week effects and flags any daily intake that deviates significantly (more than 2.5 standard deviations) from the expected historical trend. 
#
# This statistical approach is excellent for detecting large, sudden spikes (such as mechanical leaks).

# 1. Define the Rolling Window Parameters
window_size = 7  # 7-day trailing window smoothes out day-of-week effects
z_threshold = 2.5 # Flag anything more than 2.5 standard deviations away

# 2. Calculate Rolling Metrics
# We 'shift(1)' so that today's potentially anomalous value doesn't skew its own baseline
df['Rolling_Mean'] = df['Avg_Daily_Intake_Liters'].rolling(window=window_size).mean().shift(1)
df['Rolling_Std'] = df['Avg_Daily_Intake_Liters'].rolling(window=window_size).std().shift(1)

# 3. Calculate the Z-Score
df['Z_Score'] = (df['Avg_Daily_Intake_Liters'] - df['Rolling_Mean']) / df['Rolling_Std']

# 4. Flag Anomalies based on the Threshold
df['Anomaly_Flag'] = False
# We use absolute value to catch both massive spikes (leaks) and drops (health/hardware)
df.loc[df['Z_Score'].abs() > z_threshold, 'Anomaly_Flag'] = True

# 5. Visualize the Engine at Work
plt.figure(figsize=(14, 7))

# Plot the raw data and the rolling baseline
plt.plot(df['Date'], df['Avg_Daily_Intake_Liters'], label='Actual Intake', color='#1f77b4', marker='o', markersize=4)
plt.plot(df['Date'], df['Rolling_Mean'], label='7-Day Expected Baseline', color='green', linestyle='--')

# Plot the threshold bounds
plt.fill_between(df['Date'], 
                 df['Rolling_Mean'] - (z_threshold * df['Rolling_Std']),
                 df['Rolling_Mean'] + (z_threshold * df['Rolling_Std']),
                 color='green', alpha=0.15, label=f'Expected Range (\u00B1{z_threshold} Z)')

# Highlight the detected anomalies in red
anomalies = df[df['Anomaly_Flag'] == True]
plt.scatter(anomalies['Date'], anomalies['Avg_Daily_Intake_Liters'], color='red', s=100, zorder=5, label='System Alert Triggered')

plt.title('Predictive Herd Health: Rolling Z-Score Anomaly Detection', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Average Daily Intake per Pig (Liters)', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Display the days that triggered an alert
df[df['Anomaly_Flag'] == True][['Date', 'Avg_Daily_Intake_Liters', 'Rolling_Mean', 'Z_Score']]

import os

# 1. Create an output directory to keep your project folder organized
output_dir = 'data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 2. Define the file path
export_path = os.path.join(output_dir, 'processed_barn_alerts.csv')

# 3. Export the DataFrame
# Setting index=False prevents Pandas from writing the row numbers as a separate column
df.to_csv(export_path, index=False)

print(f"Pipeline complete. Cleaned data exported to: {export_path}")

from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# 1. Prepare the Data
# The model needs numerical features.I use the daily intake and the rolling mean.
# The rolling mean gives the model context about what "normal" looks like for that specific week.
# We drop the first few days that have NaN values from the rolling window calculation.
model_df = df.dropna(subset=['Rolling_Mean', 'Avg_Daily_Intake_Liters']).copy()
X = model_df[['Avg_Daily_Intake_Liters', 'Rolling_Mean']]

# 2. Initialize and Train the Isolation Forest Model
# 'contamination' is the expected proportion of outliers in the dataset. 
# We estimate about 5% of our days might have anomalous events.
iso_forest = IsolationForest(contamination=0.05, random_state=42)

# Fit the model to our feature set
iso_forest.fit(X)

# 3. Predict Anomalies
# Isolation Forest outputs 1 for normal data points and -1 for anomalies.
model_df['ML_Anomaly'] = iso_forest.predict(X)

# Create a clean boolean flag (True for anomaly, False for normal)
model_df['ML_Anomaly_Flag'] = model_df['ML_Anomaly'] == -1

# 4. Visualize the Results
plt.figure(figsize=(14, 7))

# Plot the raw data
plt.plot(model_df['Date'], model_df['Avg_Daily_Intake_Liters'], 
         label='Actual Intake', color='#1f77b4', marker='o', markersize=4)

# Plot the expected baseline for reference
plt.plot(model_df['Date'], model_df['Rolling_Mean'], 
         label='7-Day Expected Baseline', color='green', linestyle='--', alpha=0.6)

# Highlight ML-detected anomalies in purple
ml_anomalies = model_df[model_df['ML_Anomaly_Flag']]
plt.scatter(ml_anomalies['Date'], ml_anomalies['Avg_Daily_Intake_Liters'], 
            color='purple', s=100, zorder=5, label='Isolation Forest Alert')

plt.title('Advanced Analytics: Isolation Forest Anomaly Detection', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Average Daily Intake per Pig (Liters)', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 5. Display the flagged dates
print("Dates flagged by the Machine Learning Model:")
display(model_df[model_df['ML_Anomaly_Flag']][['Date', 'Avg_Daily_Intake_Liters', 'Rolling_Mean']])

# ---
# #Phase 3: Time Series Forecasting (Facebook Prophet)
# While anomaly detection tells us what went wrong yesterday, a truly predictive system tells farm managers what to expect tomorrow. 
#
# Using `Facebook Prophet`, we forecast the expected bounds of water consumption for the upcoming week. If the actual intake drops below the `Lower_Bound`, the system can proactively alert farm staff to investigate the barn.

!pip install prophet  
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

# 1. Prepare Data for Prophet
# Prophet strictly requires the date column to be named 'ds' and the target variable to be 'y'
prophet_df = df[['Date', 'Avg_Daily_Intake_Liters']].rename(columns={'Date': 'ds', 'Avg_Daily_Intake_Liters': 'y'})

# 2. Initialize and Train the Prophet Model
# We turn off yearly seasonality since our simulation is only 120 days
m = Prophet(yearly_seasonality=False, daily_seasonality=False, changepoint_prior_scale=0.05)
m.fit(prophet_df)

# 3. Create Future Dates to Predict
# Let's forecast the expected consumption for the next 7 days
future = m.make_future_dataframe(periods=7)

# 4. Predict Future Consumption
# This generates the prediction ('yhat') as well as the upper and lower confidence bounds
forecast = m.predict(future)

# 5. Visualize the Forecast
fig = m.plot(forecast, figsize=(12, 6))
plt.title('Time Series Forecasting: Predicting Future Water Consumption with Prophet', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Average Daily Intake per Pig (Liters)', fontsize=12)

# Customize the plot slightly to match our previous aesthetics
ax = fig.gca()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 6. Display the predicted values for the next 7 days
print("Forecasted Water Intake for the Next 7 Days:")
display(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7).rename(
    columns={'ds': 'Date', 'yhat': 'Predicted_Intake', 'yhat_lower': 'Lower_Bound', 'yhat_upper': 'Upper_Bound'}
))

import numpy as np
import pandas as pd

# Ensure our Date column is in the right format
df['Date'] = pd.to_datetime(df['Date'])

# 1. Simulate Environmental Data
np.random.seed(42)
num_days = len(df)

# Simulate Temperature (Celsius): Gradually warming from February to May
base_temp = np.linspace(10, 25, num_days) 
# Add some random daily weather fluctuations
df['Temperature_C'] = base_temp + np.random.normal(loc=0, scale=2.5, size=num_days)

# Simulate Relative Humidity (%): Random fluctuations between 40% and 80%
df['Humidity_Percent'] = np.random.uniform(low=40, high=80, size=num_days)

# 2. Engineer a Domain-Specific Feature: Temperature-Humidity Index (THI)
# THI is a standard agricultural formula used to determine heat stress in animals.
# THI = 0.8 * T + (RH/100) * (T - 14.4) + 46.4
df['THI_Heat_Stress'] = (0.8 * df['Temperature_C']) + \
                        ((df['Humidity_Percent'] / 100) * (df['Temperature_C'] - 14.4)) + 46.4

# Create a categorical flag for high heat stress (THI > 72 is a common threshold for livestock stress)
df['Is_Heat_Stressed'] = (df['THI_Heat_Stress'] > 72).astype(int)

# 3. Engineer Time-Based Features
# ML models often find patterns in days of the week (e.g., farm staff routines might change on weekends)
df['Day_of_Week'] = df['Date'].dt.day_name()
df['Is_Weekend'] = df['Date'].dt.dayofweek.isin([5, 6]).astype(int)

# 4. Display the newly engineered dataset
print("Newly Engineered Features:")
display(df[['Date', 'Avg_Daily_Intake_Liters', 'Temperature_C', 'Humidity_Percent', 'THI_Heat_Stress', 'Is_Heat_Stressed', 'Day_of_Week']].tail(10))

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Select the features for our multivariate model
features = [
    'Avg_Daily_Intake_Liters', 
    'Rolling_Mean', 
    'Temperature_C', 
    'Humidity_Percent', 
    'THI_Heat_Stress', 
    'Is_Weekend'
]

# Drop any rows with missing values (e.g., the first few days of the rolling window)
multi_model_df = df.dropna(subset=features).copy()
X_multi = multi_model_df[features]

# 2. Scale the features
# This is a crucial data science best practice. It standardizes the data so a change 
# in Humidity (40-80) doesn't overshadow a change in Intake (5-15).
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_multi)

# 3. Initialize and Train the Multivariate Isolation Forest
iso_forest_multi = IsolationForest(contamination=0.05, random_state=42)
iso_forest_multi.fit(X_scaled)

# 4. Predict Anomalies
# The model evaluates the scaled data but we flag the original dataframe
multi_model_df['Multi_ML_Anomaly'] = iso_forest_multi.predict(X_scaled)
multi_model_df['Multi_ML_Anomaly_Flag'] = multi_model_df['Multi_ML_Anomaly'] == -1

# 5. Visualize the Results
plt.figure(figsize=(14, 7))

# Plot actual intake and the expected baseline
plt.plot(multi_model_df['Date'], multi_model_df['Avg_Daily_Intake_Liters'], 
         label='Actual Intake', color='#1f77b4', marker='o', markersize=4)
plt.plot(multi_model_df['Date'], multi_model_df['Rolling_Mean'], 
         label='7-Day Expected Baseline', color='green', linestyle='--', alpha=0.6)

# Highlight Multivariate ML anomalies in red
multi_anomalies = multi_model_df[multi_model_df['Multi_ML_Anomaly_Flag']]
plt.scatter(multi_anomalies['Date'], multi_anomalies['Avg_Daily_Intake_Liters'], 
            color='red', s=100, zorder=5, label='Multivariate Anomaly Alert')

plt.title('Advanced Multivariate Anomaly Detection (Intake + Environment)', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Average Daily Intake per Pig (Liters)', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 6. Display the context around the anomalies
print("Multivariate Anomalies Detected. Notice the environmental context:")
display(multi_model_df[multi_model_df['Multi_ML_Anomaly_Flag']][
    ['Date', 'Avg_Daily_Intake_Liters', 'Rolling_Mean', 'Temperature_C', 'THI_Heat_Stress', 'Is_Weekend']
])

# 1. Define Agribusiness Cost Parameters (Industry Estimates)
COST_PER_LITER_WATER_WASTE = 0.05  # Cost of water + manure management per leaked liter
MARKET_VALUE_PER_PIG = 180.00      # Average market value of a market-ready hog
TREATMENT_COST_PER_PIG = 8.50      # Cost of mass medication/veterinary intervention
BARN_CAPACITY = 1000               # Number of pigs in the simulated barn

# 2. Define the Impact of Catching Anomalies Early vs. Late
# Scenario A: Late Detection (Traditional farming - takes 4 days to notice a slow trend drop)
days_lost_traditional = 4
mortality_rate_traditional = 0.03  # 3% loss of herd due to delayed disease treatment

# Scenario B: Early Detection (Your ML Model - catches it on Day 1)
days_lost_ml = 1
mortality_rate_ml = 0.005          # 0.5% loss of herd due to rapid treatment

# 3. Calculate Financial Losses For Each Simulated Scenario
# Let's pull the specific anomalies we detected to quantify them
leak_days = 1  # From row 35 (14.37 Liters - massive spike/leak)
leak_volume_above_baseline = 14.37 - 2.93 

# Financial Impact Calculations
water_leak_cost = leak_volume_above_baseline * BARN_CAPACITY * COST_PER_LITER_WATER_WASTE

traditional_disease_cost = (BARN_CAPACITY * mortality_rate_traditional * MARKET_VALUE_PER_PIG) + \
                           (BARN_CAPACITY * TREATMENT_COST_PER_PIG)

ml_disease_cost = (BARN_CAPACITY * mortality_rate_ml * MARKET_VALUE_PER_PIG) + \
                  (BARN_CAPACITY * TREATMENT_COST_PER_PIG)

# 4. ROI Metrics
total_cost_traditional_farm = water_leak_cost + traditional_disease_cost
total_cost_ml_farm = water_leak_cost + ml_disease_cost
total_savings = total_cost_traditional_farm - total_cost_ml_farm

# Assume the IoT system costs $1,500/year to implement per barn (sensors + software)
system_cost = 1500.00
net_profit_increase = total_savings - system_cost
roi_percentage = (net_profit_increase / system_cost) * 100

# 5. Print a Clean Business Report 
print("======================================================================")
print("             AGRIBUSINESS FINANCIAL IMPACT & ROI REPORT               ")
print("======================================================================")
print(f"Simulated Barn Size:          {BARN_CAPACITY} Pigs")
print(f"Cost of Water Leak (1 Day):   ${water_leak_cost:,.2f}")
print("----------------------------------------------------------------------")
print(f"Traditional Management Cost:  ${total_cost_traditional_farm:,.2f} (Delayed medical intervention)")
print(f"Predictive IoT Barn Cost:     ${total_cost_ml_farm:,.2f} (Immediate medical intervention)")
print("----------------------------------------------------------------------")
print(f"Gross Financial Savings:      ${total_savings:,.2f}")
print(f"Annual IoT System Investment: ${system_cost:,.2f}")
print(f"Net Profit Saved per Barn:    ${net_profit_increase:,.2f}")
print(f"PROJECTED SYSTEM ROI:         {roi_percentage:.1f}%")
print("======================================================================")
