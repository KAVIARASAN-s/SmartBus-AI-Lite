import streamlit as st
import pandas as pd
import networkx as nx
import joblib
import os
from datetime import datetime
import numpy as np

# 1. Premium Page Configuration
st.set_page_config(page_title="SmartBus AI Coimbatore", page_icon="🚌", layout="wide", initial_sidebar_state="expanded")

# 2. Advanced CSS Injection for Premium UI
st.markdown("""
    <style>
    /* Hide default Streamlit menu, header, and footer for a clean look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Professional Button Styling */
    div.stButton > button:first-child {
        background-color: #1E3A8A;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        background-color: #2563EB;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    
    /* Custom Styling for Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
        color: #10B981;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1.1rem;
        font-weight: 600;
        color: #4B5563;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Cache Data & Models
@st.cache_data
def load_data():
    stops_df = pd.read_csv(r"data\processed\bus_stops.csv")
    route_stops_df = pd.read_csv(r"data\processed\route_stops.csv")
    return stops_df, route_stops_df

@st.cache_resource
def build_graph(stops_df, route_stops_df):
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
            edge_dist = max(0.1, abs(dist_b - dist_a))
            G.add_edge(stops_list[i], stops_list[i+1], route=route_id, weight=edge_dist)
    return G

@st.cache_resource
def load_ml_model():
    return joblib.load(r"models\travel_time_model.joblib")

@st.cache_data
def get_model_features():
    return pd.read_csv(r"data\processed\X_train.csv", nrows=0).columns.tolist()

# Load everything
stops_df, route_stops_df = load_data()
G = build_graph(stops_df, route_stops_df)
rf_model = load_ml_model()
expected_features = get_model_features()

# 4. Premium Sidebar Navigation
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3204/3204996.png", width=80)
st.sidebar.title("SmartBus AI")
st.sidebar.markdown("<p style='color:#64748B; font-size:0.9rem;'>Enterprise Transit Planner v1.0<br>📍 Coimbatore City</p>", unsafe_allow_html=True)
st.sidebar.divider()
page = st.sidebar.radio("Main Menu", ["🧭 Journey Planner", "📊 Analytics Dashboard"])

# 5. Page 1: Home / Journey Planner
if page == "🧭 Journey Planner":
    st.markdown("<h1 style='color:#1E3A8A;'>Intelligent Route Planner</h1>", unsafe_allow_html=True)
    st.markdown("Find optimized transit routes and AI-predicted journey durations instantly.")
    st.write("") # Spacer
    
    stop_dict = dict(zip(stops_df['stop_name'], stops_df['stop_id']))
    stop_names = list(stop_dict.keys())
    
    # Clean Input Layout
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            from_stop = st.selectbox("🟢 Origin (Boarding Point):", stop_names)
        with col2:
            to_stop = st.selectbox("🔴 Destination (Drop Point):", stop_names, index=len(stop_names)-1)
            
    st.write("") # Spacer
    
    if st.button("Generate Optimized Route & AI Prediction"):
        if from_stop == to_stop:
            st.warning("⚠️ Origin and Destination cannot be the same. Please select different stops.")
        else:
            start_id = stop_dict[from_stop]
            end_id = stop_dict[to_stop]
            
            try:
                # NetworkX Routing
                path = nx.shortest_path(G, source=start_id, target=end_id, weight='weight')
                path_distance = nx.shortest_path_length(G, source=start_id, target=end_id, weight='weight')
                
                # ML Prediction
                current_hour = datetime.now().hour
                is_weekend = 1 if datetime.now().weekday() >= 5 else 0
                is_peak = 1 if (7 <= current_hour <= 10) or (16 <= current_hour <= 20) else 0
                
                input_data = {feature: 0 for feature in expected_features}
                input_data['distance_km'] = path_distance
                input_data['hour'] = current_hour
                input_data['is_weekend'] = is_weekend
                input_data['is_peak_hour'] = is_peak
                
                if 'traffic_level_moderate' in input_data:
                    input_data['traffic_level_moderate'] = 1
                elif 'traffic_level_Medium' in input_data:
                    input_data['traffic_level_Medium'] = 1
                    
                input_df = pd.DataFrame([input_data])
                pred_time = rf_model.predict(input_df)[0]
                
                st.divider()
                
                # Premium Results Display using Metrics
                st.markdown("<h3 style='color:#1E3A8A;'>Journey Overview</h3>", unsafe_allow_html=True)
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric(label="AI Estimated Time", value=f"{pred_time:.0f} mins")
                with metric_col2:
                    st.metric(label="Total Distance", value=f"{path_distance:.1f} km")
                with metric_col3:
                    st.metric(label="Traffic Condition", value="Moderate 🟡" if not is_peak else "Heavy 🔴")
                
                # Interactive Expander for Journey Steps
                st.write("")
                with st.expander("📍 View Detailed Route Instructions", expanded=True):
                    for i in range(len(path)-1):
                        u = path[i]
                        v = path[i+1]
                        route_used = G[u][v]['route']
                        st.markdown(f"**Step {i+1}:** Board **Bus {route_used}** at `{G.nodes[u]['name']}` 🚌")
                    st.markdown(f"**Arrival:** Alight at `{G.nodes[path[-1]]['name']}` ✅")
                    
                st.caption(f"Real-time prediction context: Hour={current_hour}:00, PeakHour={'Yes' if is_peak else 'No'}")
                    
            except nx.NetworkXNoPath:
                st.error("❌ No valid public transit route exists between these specific locations.")
            except Exception as e:
                st.error(f"Prediction System Error: {e}")

# 6. Page 2: Data Analytics Dashboard
elif page == "📊 Analytics Dashboard":
    st.markdown("<h1 style='color:#1E3A8A;'>Transit Analytics Engine</h1>", unsafe_allow_html=True)
    st.markdown("Visual intelligence and historical data distributions for Coimbatore transit.")
    st.divider()
    
    report_path = r"reports"
    col1, col2 = st.columns(2)
    
    with col1:
        if os.path.exists(os.path.join(report_path, '1_travel_time_distribution.png')):
            st.image(os.path.join(report_path, '1_travel_time_distribution.png'), use_container_width=True, caption="Journey Time Distribution")
        if os.path.exists(os.path.join(report_path, '3_traffic_impact.png')):
            st.image(os.path.join(report_path, '3_traffic_impact.png'), use_container_width=True, caption="Traffic Congestion Impact")
            
    with col2:
        if os.path.exists(os.path.join(report_path, '2_distance_vs_time.png')):
            st.image(os.path.join(report_path, '2_distance_vs_time.png'), use_container_width=True, caption="Distance vs Duration Correlation")
        if os.path.exists(os.path.join(report_path, '4_weather_impact.png')):
            st.image(os.path.join(report_path, '4_weather_impact.png'), use_container_width=True, caption="Weather Condition Impact")