import pandas as pd
import os
from sklearn.model_selection import train_test_split

# File paths
data_path = r"data\processed\travel_observations.csv"
processed_dir = r"data\processed"

print("Starting Feature Engineering...\n")

# 1. Load the cleaned data
df = pd.read_csv(data_path)

# 2. Select Features (Inputs) and Target (Output)
# Observation ID, Route ID tharamatta ML ku thevai illa. 
# So, important features-a mattum select panrom.
selected_cols = [
    'distance_km', 'hour', 'is_weekend', 'is_peak_hour', 
    'traffic_level', 'weather', 'actual_travel_time_minutes'
]
df = df[selected_cols]

# 3. Handle Categorical Variables (One-Hot Encoding)
# 'traffic_level' (Low, Medium, High) and 'weather' (Clear, Rain) text ah irukku.
# Itha pd.get_dummies() vechu 0s and 1s ah maatha porom.
df_encoded = pd.get_dummies(df, columns=['traffic_level', 'weather'], drop_first=True)

# Convert boolean (True/False) values to Integers (1/0) for Machine Learning
for col in df_encoded.columns:
    if df_encoded[col].dtype == bool:
        df_encoded[col] = df_encoded[col].astype(int)

print("Final Features for ML Model:")
for col in df_encoded.columns:
    print(f"- {col}")

# 4. Split Features (X) and Target (y)
# X -> Input (Distance, Traffic, Weather, etc.)
# y -> Output / Target Variable (actual_travel_time_minutes)
X = df_encoded.drop('actual_travel_time_minutes', axis=1)
y = df_encoded['actual_travel_time_minutes']

# 5. Train-Test Split
# Data Leakage-a thadukka, namma kitta irukka 20,000 data-va:
# 80% (16,000 rows) - Model train panna (padikka)
# 20% (4,000 rows) - Model test panna (exam ezhutha) pirikkurom.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining Data (80%): {X_train.shape[0]} rows")
print(f"Testing Data (20%): {X_test.shape[0]} rows")

# 6. Save the Engineered Datasets
X_train.to_csv(os.path.join(processed_dir, 'X_train.csv'), index=False)
X_test.to_csv(os.path.join(processed_dir, 'X_test.csv'), index=False)
y_train.to_csv(os.path.join(processed_dir, 'y_train.csv'), index=False)
y_test.to_csv(os.path.join(processed_dir, 'y_test.csv'), index=False)

print("\nFeature Engineering Complete! X_train, X_test, y_train, y_test files are saved successfully.")