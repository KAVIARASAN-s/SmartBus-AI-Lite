import pandas as pd
import os

raw_dir = r"data"
processed_dir = r"data\processed"

# Processed folder illana puthusa create pannidum
os.makedirs(processed_dir, exist_ok=True)

files_to_clean = [
    "bus_routes.csv", "bus_stops.csv", "route_stops.csv", 
    "bus_timings.csv", "trips.csv", "travel_observations.csv", "traffic_data.csv"
]

print("Starting Data Cleaning Process...\n")

for file_name in files_to_clean:
    raw_path = os.path.join(raw_dir, file_name)
    
    if os.path.exists(raw_path):
        df = pd.read_csv(raw_path)
        
        # 1. Duplicates check panrom
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            print(f"{file_name}: Found {duplicate_count} duplicate rows. Removing them...")
            df = df.drop_duplicates()
        else:
            print(f"{file_name}: No duplicates found. Data is clean.")
            
        # 2. Clean aana data-va 'processed' folder la save panrom
        processed_path = os.path.join(processed_dir, file_name)
        df.to_csv(processed_path, index=False)

print("\nData Cleaning Complete! All files are securely saved in 'data/processed' folder.")