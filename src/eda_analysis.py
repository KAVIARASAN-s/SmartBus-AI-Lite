import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set paths
data_path = r"data\processed\travel_observations.csv"
report_dir = r"reports"

# Create reports folder if it doesn't exist
os.makedirs(report_dir, exist_ok=True)

print("Starting Exploratory Data Analysis (EDA)...")

# Load the cleaned dataset
df = pd.read_csv(data_path)

# Set the style for all charts
sns.set_theme(style="whitegrid")

# 1. Univariate Analysis: Travel Time Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['actual_travel_time_minutes'], bins=30, kde=True, color='blue')
plt.title('Distribution of Travel Journey Durations')
plt.xlabel('Travel Time (Minutes)')
plt.ylabel('Frequency')
plt.savefig(os.path.join(report_dir, '1_travel_time_distribution.png'))
plt.close()
print("Saved: 1_travel_time_distribution.png")

# 2. Bivariate Analysis: Distance vs Travel Time
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='distance_km', y='actual_travel_time_minutes', alpha=0.5, hue='is_peak_hour')
plt.title('Distance vs Travel Time (Colored by Peak Hour)')
plt.xlabel('Distance (km)')
plt.ylabel('Travel Time (Minutes)')
plt.savefig(os.path.join(report_dir, '2_distance_vs_time.png'))
plt.close()
print("Saved: 2_distance_vs_time.png")

# 3. Categorical Analysis: Average Travel Time by Traffic Level
plt.figure(figsize=(8, 5))
traffic_avg = df.groupby('traffic_level')['actual_travel_time_minutes'].mean().reset_index()
# Sorting to show Low -> Moderate -> High -> Severe (if severe exists)
sns.barplot(data=traffic_avg, x='traffic_level', y='actual_travel_time_minutes', order=['low', 'moderate', 'high', 'severe'])
plt.title('Average Travel Time by Traffic Level')
plt.xlabel('Traffic Level')
plt.ylabel('Average Travel Time (Minutes)')
plt.savefig(os.path.join(report_dir, '3_traffic_impact.png'))
plt.close()
print("Saved: 3_traffic_impact.png")

# 4. Boxplot: Weather Impact on Travel Time
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='weather', y='actual_travel_time_minutes')
plt.title('Impact of Weather on Travel Time')
plt.xlabel('Weather Condition')
plt.ylabel('Travel Time (Minutes)')
plt.savefig(os.path.join(report_dir, '4_weather_impact.png'))
plt.close()
print("Saved: 4_weather_impact.png")

print("\nEDA completed! Check the 'reports' folder for the generated charts.")