# 🚌 SmartBus AI Lite - Coimbatore

## 📌 Project Overview
**SmartBus AI Lite** is an AI-powered public transit route recommendation and travel-time prediction web application built for Coimbatore City. It helps commuters find the most optimized bus routes between stops and uses Machine Learning to predict the estimated journey duration based on real-time factors like traffic, peak hours, and distance.

---

## 🎯 Core Features
1. **Intelligent Route Recommendation:** 
   - Uses Graph Theory to find the shortest path between any two bus stops.
   - Automatically handles direct routes and transfer connections.
2. **AI Travel Time Prediction:** 
   - Uses a trained Machine Learning model to estimate travel time instead of relying on static timetables.
   - Factors in peak hours, weekends, and dynamic traffic conditions.
3. **Premium Mobile-First UI:** 
   - A highly responsive, full-stack web interface styled like a modern mobile application (e.g., TNBUS / Uber).
   - Features a custom visual timeline for journey steps.

---

## 🏗️ System Architecture & Tech Stack

### 1. Frontend (User Interface)
* **HTML5 & CSS3:** Custom-built glassmorphism and mobile-app-like UI.
* **JavaScript (Fetch API):** For seamless, real-time communication with the backend without page reloads.

### 2. Backend (API & Routing)
* **Python (Flask):** Serves the web application and creates REST API endpoints.
* **NetworkX:** Models the city's bus network as a Directed Graph and uses **Dijkstra's Algorithm** for shortest-path routing.

### 3. Machine Learning (AI Engine)
* **Pandas & NumPy:** For data cleaning and feature engineering (One-Hot Encoding).
* **Scikit-Learn:** **Random Forest Regressor** model trained to predict travel times.
* **Joblib:** Used to serialize (save) and deserialize (load) the trained ML model.

---

## 🚀 How to Run the Project Locally

**Step 1: Clone the repository**
```bash
git clone https://github.com/your-username/SmartBus-AI-Lite.git
cd SmartBus-AI-Lite
```

**Step 2: Set up a Virtual Environment**
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\Activate.ps1
# On Mac/Linux:
source venv/bin/activate
```

**Step 3: Install Dependencies**
```bash
pip install pandas numpy scikit-learn networkx flask joblib
```

**Step 4: Run the Application**
```bash
python app/flask_app.py
```
Open your browser and navigate to: `http://127.0.0.1:5000`

---

## ⚠️ Disclaimer
This is an educational portfolio project. The dataset used for routes, stops, and travel observations is **synthetic (dummy data)** generated for demonstration purposes. It does not represent the official or real-time public transit data of Coimbatore.