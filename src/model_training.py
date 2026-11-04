import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Paths
processed_dir = r"data\processed"
models_dir = r"models"
os.makedirs(models_dir, exist_ok=True)

print("Starting Machine Learning Model Training...\n")

# 1. Load the prepared data
print("Loading data...")
X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
X_test = pd.read_csv(os.path.join(processed_dir, 'X_test.csv'))
y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv')).values.ravel() # Convert to 1D array
y_test = pd.read_csv(os.path.join(processed_dir, 'y_test.csv')).values.ravel()

# Function to evaluate and print metrics
def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"--- {name} ---")
    print(f"MAE (Mean Absolute Error): {mae:.2f} minutes")
    print(f"RMSE (Root Mean Squared Error): {rmse:.2f} minutes")
    print(f"R2 Score: {r2:.4f}\n")
    return mae, rmse, r2

# 2. Baseline Model (Median Prediction)
# Namma training data-oda median time-a ellathukkum answer ah vechu paakurom.
median_time = np.median(y_train)
baseline_predictions = np.full_like(y_test, fill_value=median_time)
evaluate_model("1. Baseline Model (Median)", y_test, baseline_predictions)

# 3. Linear Regression Model
print("Training Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
evaluate_model("2. Linear Regression", y_test, lr_preds)

# 4. Random Forest Regressor Model
# Namma laptop RAM-ku yetha mathiri lightweight ah 100 trees and max depth 10 use panrom
print("Training Random Forest Regressor...")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
evaluate_model("3. Random Forest Regressor", y_test, rf_preds)

# 5. Save the Best Model
# Random forest kandippa better ah irukkum, so atha namma save panrom.
model_path = os.path.join(models_dir, 'travel_time_model.joblib')
joblib.dump(rf_model, model_path)
print(f"Best model (Random Forest) successfully saved at: {model_path}")