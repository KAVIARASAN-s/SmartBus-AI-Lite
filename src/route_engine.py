import pandas as pd
import networkx as nx
import os

data_dir = r"data\processed"

print("Building Route Engine with NetworkX...\n")

# 1. Load the Data
# Namma clean panni vechurukka stops and route sequences data-va load panrom
stops_df = pd.read_csv(os.path.join(data_dir, 'bus_stops.csv'))
route_stops_df = pd.read_csv(os.path.join(data_dir, 'route_stops.csv'))

# 2. Create a Directed Graph
# Directed Graph na "A to B" poga mudiyum, aana one-way na "B to A" poga mudiyathu
G = nx.DiGraph()

# 3. Add Nodes (Bus Stops)
for _, row in stops_df.iterrows():
    G.add_node(row['stop_id'], name=row['stop_name'])

# 4. Add Edges (Bus Routes between stops)
# Route ID and sequence vechu data-va sort panrom, appo thaan correct order kidaikum
route_stops_df = route_stops_df.sort_values(by=['route_id', 'stop_sequence'])

for route_id, group in route_stops_df.groupby('route_id'):
    stops_list = group['stop_id'].tolist()
    distances = group['distance_from_origin_km'].tolist()
    
    # Oru stop-kum adutha stop-kum edge (connection) create panrom
    for i in range(len(stops_list) - 1):
        stop_a = stops_list[i]
        stop_b = stops_list[i+1]
        
        # Calculate distance (weight) between two consecutive stops
        dist_a = distances[i] if pd.notnull(distances[i]) else 0
        dist_b = distances[i+1] if pd.notnull(distances[i+1]) else 0
        edge_dist = abs(dist_b - dist_a)
        
        # Prevent 0 distance edges
        if edge_dist == 0: 
            edge_dist = 1.0 
            
        # Add connection to the graph
        G.add_edge(stop_a, stop_b, route=route_id, weight=edge_dist)

print(f"Graph built successfully! Total Stops (Nodes): {G.number_of_nodes()} | Total Connections (Edges): {G.number_of_edges()}")

# 5. Core Route Finder Function
def find_best_route(start_stop_id, end_stop_id):
    try:
        # Dijkstra's Algorithm finds the shortest path based on distance (weight)
        path = nx.shortest_path(G, source=start_stop_id, target=end_stop_id, weight='weight')
        
        start_name = G.nodes[start_stop_id]['name']
        end_name = G.nodes[end_stop_id]['name']
        print(f"\n✅ Recommended Route from '{start_name}' to '{end_name}':")
        
        for i in range(len(path)-1):
            u = path[i]
            v = path[i+1]
            route_used = G[u][v]['route']
            stop_name = G.nodes[u]['name']
            print(f"  -> Board Bus {route_used} at {stop_name}")
            
        print(f"  -> Get down at {G.nodes[path[-1]]['name']}")
        
    except nx.NetworkXNoPath:
        print(f"\n❌ Sorry, no direct or transfer route found between these stops.")
    except nx.NodeNotFound:
        print(f"\n❌ Invalid stop ID. Please check the inputs.")

# 6. Test the Engine
# S001 is typically Gandhipuram and S010 is another major stop in our synthetic data
print("\nTesting Route Recommendation...")
find_best_route('S001', 'S010')

print("\nRoute Engine is ready and optimized for 8GB RAM!")