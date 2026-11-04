import pandas as pd
import os

# Namma data folder path
data_dir = r"data"

print("Starting Dataset Inspection for SmartBus AI Lite...\n")

# Namma create panna CSV files list
files_to_check = [
    "bus_routes.csv", 
    "bus_stops.csv", 
    "route_stops.csv", 
    "bus_timings.csv", 
    "trips.csv", 
    "travel_observations.csv", 
    "traffic_data.csv"
]

for file_name in files_to_check:
    file_path = os.path.join(data_dir, file_name)
    
    if os.path.exists(file_path):
        print(f"========== Analyzing: {file_name} ==========")
        df = pd.read_csv(file_path)
        
        # 1. Total Rows and Columns
        print(f"Shape (Size): {df.shape[0]} Rows, {df.shape[1]} Columns")
        
        # 2. Entha edathulayachum data miss aagi irukka?
        missing_count = df.isnull().sum().sum()
        if missing_count == 0:
            print("Missing Values: No missing values. Data is clean!")
        else:
            print(f"Missing Values: Found {missing_count} missing data points.")
            
        print("-" * 40 + "\n")
    else:
        print(f"[ERROR] {file_name} kidaikkavillai!\n")