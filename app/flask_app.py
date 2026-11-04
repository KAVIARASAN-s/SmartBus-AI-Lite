from flask import Flask, request, jsonify, render_template
import pandas as pd
import networkx as nx
import joblib
import os
from datetime import datetime
import numpy as np

app = Flask(__name__, template_folder='templates')

# Load Data & Model at startup
print("Loading data and models for the Backend API...")
stops_df = pd.read_csv(r"data\processed\bus_stops.csv")
route_stops_df = pd.read_csv(r"data\processed\route_stops.csv")
rf_model = joblib.load(r"models\travel_time_model.joblib")
expected_features = pd.read_csv(r"data\processed\X_train.csv", nrows=0).columns.tolist()

# Build Graph
G = nx.DiGraph()
for _, row in stops_df.iterrows():
    G.add_node(row['stop_id'], name=row['stop_name'])
route_stops_df = route_stops_df.sort_values(by=['route_id', 'stop_sequence'])
for route_id, group in route_stops_df.groupby('route_id'):
    stops_list = group['stop_id'].tolist()
    distances = group['distance_from_origin_km'].tolist()
    for i in range(len(stops_list) - 1):
        dist_a = distances[i] if pd.notnull(distances[i]) else 0
        dist_b = distances[i+1] if pd.notnull(distances[i+1]) else 0
        G.add_edge(stops_list[i], stops_list[i+1], route=route_id, weight=max(0.1, abs(dist_b - dist_a)))

stop_dict = dict(zip(stops_df['stop_name'], stops_df['stop_id']))

# Serve the HTML Frontend
@app.route('/')
def home():
    stop_names = list(stop_dict.keys())
    return render_template('index.html', stops=stop_names)

# API Endpoint to process journey
@app.route('/api/plan_journey', methods=['POST'])
def plan_journey():
    data = request.json
    from_stop_name = data.get('from_stop')
    to_stop_name = data.get('to_stop')
    
    if from_stop_name == to_stop_name:
        return jsonify({"error": "Origin and Destination cannot be the same."})
        
    start_id = stop_dict.get(from_stop_name)
    end_id = stop_dict.get(to_stop_name)
    
    try:
        path = nx.shortest_path(G, source=start_id, target=end_id, weight='weight')
        path_distance = nx.shortest_path_length(G, source=start_id, target=end_id, weight='weight')
        
        current_hour = datetime.now().hour
        is_weekend = 1 if datetime.now().weekday() >= 5 else 0
        is_peak = 1 if (7 <= current_hour <= 10) or (16 <= current_hour <= 20) else 0
        
        input_data = {feature: 0 for feature in expected_features}
        input_data['distance_km'] = path_distance
        input_data['hour'] = current_hour
        input_data['is_weekend'] = is_weekend
        input_data['is_peak_hour'] = is_peak
        if 'traffic_level_moderate' in input_data: input_data['traffic_level_moderate'] = 1
        elif 'traffic_level_Medium' in input_data: input_data['traffic_level_Medium'] = 1
            
        input_df = pd.DataFrame([input_data])
        pred_time = rf_model.predict(input_df)[0]
        
        steps = []
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            steps.append({
                "bus": G[u][v]['route'],
                "stop_name": G.nodes[u]['name']
            })
        final_stop = G.nodes[path[-1]]['name']
        
        return jsonify({
            "success": True,
            "estimated_time_mins": round(pred_time),
            "distance_km": round(path_distance, 1),
            "traffic_condition": "Heavy" if is_peak else "Moderate",
            "steps": steps,
            "final_stop": final_stop
        })
        
    except nx.NetworkXNoPath:
        return jsonify({"error": "No valid public transit route exists between these locations."})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)