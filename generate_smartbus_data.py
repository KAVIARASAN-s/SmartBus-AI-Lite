"""
SmartBus AI - Coimbatore Synthetic Dataset Generator
"""
from pathlib import Path
import math, random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
rng = np.random.default_rng(SEED)
OUT = Path("data")
OUT.mkdir(exist_ok=True)

CENTER_LAT, CENTER_LON = 11.0168, 76.9558
N_STOPS, N_ROUTES, N_TRIPS_PER_ROUTE = 250, 100, 50
N_TRAVEL_OBSERVATIONS, N_TRAFFIC = 20_000, 10_000

# Synthetic stop locations
lat = CENTER_LAT + rng.uniform(-0.075, 0.075, N_STOPS)
lon = CENTER_LON + rng.uniform(-0.085, 0.085, N_STOPS)
landmarks = ["Gandhipuram","Ukkadam","Town Hall","Singanallur","Peelamedu",
"RS Puram","Saibaba Colony","Vadavalli","Race Course","Airport","Hope College",
"Ganapathy","Ramanathapuram","Kovaipudur","Podanur","Kuniyamuthur",
"Saravanampatti","Thudiyalur","Perur","Ondipudur"]

stops = pd.DataFrame({
 "stop_id":[f"S{i+1:03d}" for i in range(N_STOPS)],
 "stop_name":[landmarks[i] if i<len(landmarks) else f"Coimbatore Stop {i+1:03d}" for i in range(N_STOPS)],
 "latitude":np.round(lat,6),"longitude":np.round(lon,6),
 "area_name":[random.choice(["Central","North","South","East","West","Outer"]) for _ in range(N_STOPS)],
 "is_major_stop":[i<len(landmarks) for i in range(N_STOPS)],
 "data_type":"synthetic"
})

hub_indices=[0,1,2,3,4,5,6,8,9,10,11,12]
routes_rows, route_stop_rows, paths = [], [], {}
prefixes=["1","2","3","5","7","10","12","16","20","24","31","45","60","70"]

for r in range(N_ROUTES):
 origin=int(rng.choice(hub_indices)); n=int(rng.integers(8,19))
 path=[origin]+list(map(int,rng.choice([i for i in range(N_STOPS) if i!=origin],n-1,replace=False)))
 if r%3==0 and len(path)>4: path[-1]=int(rng.choice(hub_indices))
 route_id=f"R{r+1:03d}"; paths[route_id]=path
 number=f"{random.choice(prefixes)}{random.choice(['','A','B','C'])}"
 routes_rows.append({"route_id":route_id,"bus_number":number,
  "route_name":f"{stops.iloc[path[0]].stop_name} - {stops.iloc[path[-1]].stop_name}",
  "origin_stop_id":f"S{path[0]+1:03d}","destination_stop_id":f"S{path[-1]+1:03d}",
  "origin":stops.iloc[path[0]].stop_name,"destination":stops.iloc[path[-1]].stop_name,
  "service_type":random.choice(["ordinary","express","limited_stop"]),
  "route_distance_km":round(float(rng.uniform(5,28)),2),"data_type":"synthetic"})
 for seq,idx in enumerate(path,1):
  route_stop_rows.append({"route_id":route_id,"stop_id":f"S{idx+1:03d}",
   "stop_sequence":seq,"direction":"outbound","distance_from_origin_km":round((seq-1)*float(rng.uniform(.45,1.4)),2),
   "data_type":"synthetic"})
routes=pd.DataFrame(routes_rows); route_stops=pd.DataFrame(route_stop_rows)

trip_rows=[]; timing_rows=[]; trip_no=0; days=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
for route_id,path in paths.items():
 for t in range(N_TRIPS_PER_ROUTE):
  trip_no+=1; trip_id=f"T{trip_no:05d}"; day=days[t%7]
  start=int(rng.integers(330,1261))
  trip_rows.append({"trip_id":trip_id,"route_id":route_id,"service_id":f"SVC_{day.upper()}",
   "service_day":day,"start_time":f"{start//60:02d}:{start%60:02d}:00","data_type":"synthetic"})
  current=start
  for seq,idx in enumerate(path,1):
   if seq>1: current+=int(rng.integers(3,10))
   arr=current; dep=arr+int(rng.integers(0,3))
   timing_rows.append({"trip_id":trip_id,"route_id":route_id,"stop_id":f"S{idx+1:03d}",
    "stop_sequence":seq,"arrival_time":f"{arr//60:02d}:{arr%60:02d}:00",
    "departure_time":f"{dep//60:02d}:{dep%60:02d}:00","data_type":"synthetic"})
trips=pd.DataFrame(trip_rows); timings=pd.DataFrame(timing_rows)

traffic_rows=[]; levels=["low","moderate","high","severe"]
for i in range(N_TRAFFIC):
 ts=pd.Timestamp("2026-01-01")+pd.Timedelta(minutes=int(rng.integers(0,365*24*60)))
 hour=ts.hour; weekday=ts.dayofweek<5; peak=(7<=hour<=10) or (16<=hour<=20)
 congestion=float(np.clip(rng.normal(.15+(.28 if peak else 0)+(.10 if weekday else 0),.16),0,1))
 level="low" if congestion<.25 else "moderate" if congestion<.50 else "high" if congestion<.75 else "severe"
 traffic_rows.append({"traffic_id":f"TRF{i+1:05d}","observed_at":ts.strftime("%Y-%m-%d %H:%M:%S"),
  "hour":hour,"day_of_week":ts.day_name(),"is_weekend":not weekday,"is_peak_hour":peak,
  "congestion_index":round(congestion,3),"traffic_level":level,
  "weather":random.choices(["clear","cloudy","rain"],[.68,.22,.10])[0],"data_type":"synthetic"})
traffic=pd.DataFrame(traffic_rows)

edges=[]
for rid,path in paths.items():
 for a,b in zip(path[:-1],path[1:]):
  km=math.sqrt(((lat[b]-lat[a])*111)**2+((lon[b]-lon[a])*102)**2)*1.25
  edges.append((rid,a,b,max(.5,min(km,12.0))))
obs_rows=[]
for i in range(N_TRAVEL_OBSERVATIONS):
 rid,a,b,dist=edges[int(rng.integers(0,len(edges)))]
 ts=pd.Timestamp("2026-01-01")+pd.Timedelta(minutes=int(rng.integers(0,365*24*60)))
 hour=ts.hour; weekday=ts.dayofweek<5; peak=(7<=hour<=10) or (16<=hour<=20)
 traffic_level=random.choices(levels,[.12,.33,.38,.17] if peak else [.35,.40,.20,.05])[0]
 tf={"low":1.0,"moderate":1.15,"high":1.38,"severe":1.70}[traffic_level]
 weather=random.choices(["clear","cloudy","rain"],[.68,.22,.10])[0]
 wf={"clear":1.0,"cloudy":1.03,"rain":1.18}[weather]
 delay=float(rng.gamma(1.5,.45)); speed=max(12,float(rng.normal(23 if peak else 29,3.5)))
 actual=max(2.0,dist/speed*60*tf*wf+delay+float(rng.normal(0,1.2)))
 obs_rows.append({"observation_id":f"OBS{i+1:06d}","route_id":rid,
  "from_stop_id":f"S{a+1:03d}","to_stop_id":f"S{b+1:03d}",
  "observed_at":ts.strftime("%Y-%m-%d %H:%M:%S"),"day_of_week":ts.day_name(),
  "hour":hour,"is_weekend":not weekday,"is_peak_hour":peak,"distance_km":round(dist,2),
  "traffic_level":traffic_level,"weather":weather,"stop_delay_minutes":round(delay,2),
  "actual_travel_time_minutes":round(actual,2),"data_type":"synthetic"})
obs=pd.DataFrame(obs_rows)

tables={"bus_routes.csv":routes,"bus_stops.csv":stops,"route_stops.csv":route_stops,
 "bus_timings.csv":timings,"travel_observations.csv":obs,"traffic_data.csv":traffic}

for name,df in tables.items(): df.to_csv(OUT/name,index=False)
trips.to_csv(OUT/"trips.csv",index=False)
print("Generated synthetic SmartBus dataset in ./data")