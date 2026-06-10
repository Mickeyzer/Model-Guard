# ModelGuard

ModelGuard is a dynamic Machine Learning defense middleware designed to protect your ML models from extraction attacks, data poisoning, and API abuse. Instead of relying on static rate limits, ModelGuard actively fingerprints user behavior and employs a tiered defense strategy.

## Features

- **Behavioral Fingerprinting:** Uses an XGBoost classifier to monitor incoming request patterns (entropy, variance, query velocity, and boundary proximity) to identify out-of-distribution attacks like grid-sweeps.
- **Dynamic Tiered Defense:**
  - `T0 ALLOW`: Normal traffic passes through seamlessly.
  - `T1 / T2 PERTURB`: Injects dynamic noise into probabilities to corrupt an attacker's dataset.
  - `T3 FAKE`: Cuts off access to the real model and returns entirely synthetic predictions.
  - `T4 BLOCK`: Hard network-level drop for sustained attacks.
- **Real-Time Dashboard:** A sleek, live Streamlit dashboard providing immediate insights into traffic volume, threat distribution, and recent defense actions.

## Architecture

ModelGuard consists of three main components:
1. **Protected Model:** A dummy ML service (FastAPI) simulating a sensitive endpoint (e.g., an Iris classification model).
2. **Middleware:** A FastAPI layer that intercepts traffic, computes behavioral features, assigns threat scores, and applies defense actions in real-time.
3. **Dashboard:** A Streamlit web application that visualizes the SQLite database logs.

## Setup Guide

### Prerequisites
- Docker and Docker Compose installed on your system.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Mickeyzer/Model-Guard.git
   cd Model-Guard
   ```

2. **Start the services:**
   Run the following command to build and start the protected model, middleware, and dashboard:
   ```bash
   docker-compose up -d --build
   ```

3. **Access the application:**
   - **Target API:** `http://localhost:8000/predict`
   - **Live Dashboard:** `http://localhost:8501`

### Running the Attack Simulation

To see ModelGuard in action, run the live attack simulation script. It mimics a standard user before gradually escalating into a full reconnaissance and extraction grid-sweep attack.

```bash
# Ensure you have the requests/httpx library installed locally
python demo/simulate_attack.py
```

Watch the dashboard at `http://localhost:8501` as the simulation progresses and observe the threat tiers escalating from `ALLOW` to `BLOCK`!
