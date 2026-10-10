# ModelGuard

ModelGuard is a dynamic Machine Learning defense middleware designed to protect your ML models from extraction attacks, data poisoning, and API abuse. Instead of relying on static rate limits, ModelGuard actively fingerprints user behavior and employs a tiered defense strategy.

## Features

- **Behavioral Fingerprinting:** Uses an XGBoost classifier to monitor incoming request patterns (entropy, variance, query velocity, and boundary proximity) to identify out-of-distribution attacks like grid-sweeps.
- **Dynamic Tiered Defense:**
  - `T0 ALLOW`: Normal traffic passes through seamlessly.
  - `T1 / T2 PERTURB`: Injects dynamic noise into probabilities to corrupt an attacker's dataset (tier 2 also rate-limits).
  - `T3 FAKE`: Returns synthetic predictions in exactly the real model's format, so the attacker cannot tell they are fake.
  - `T4 BLOCK`: Bans the client IP with escalating durations (5 min, then 1 h, then 24 h). The ban holds even if the score drops, and blocked requests stay in the behavioural window.
- **No signals for attackers:** Threat scores and tiers are not sent to clients unless `EXPOSE_THREAT_HEADERS=1`. Behavioural features are computed from the real model's outputs, so ModelGuard's own noise and fakes never inflate a client's score.
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

To see the score and tier of each query in the script output, start the stack with `EXPOSE_THREAT_HEADERS=1 docker-compose up -d --build`. The simulation ends in a 5-minute ban for your IP, so wait it out before running it again (or set shorter `BAN_DURATIONS_SECONDS` in `docker-compose.yml`).

To attack a remote deployment instead, set `MODELGUARD_URL`, e.g. `MODELGUARD_URL=http://<server-ip>:8000/predict python demo/simulate_attack.py`.

### Running the tests

```bash
pip install -r requirements.txt
python -m pytest
```

The tests use a temporary database and an in-process copy of the protected model, so no services need to be running.

## Deploying to AWS

`deploy/aws/` contains Terraform that runs the whole stack on a single EC2 instance (t3.micro, Ubuntu 24.04):

- A security group exposes only the middleware (8000) and dashboard (8501). The protected model (8001) is unreachable from the internet, so clients cannot bypass the middleware. SSH is limited to the deployer's IP.
- IMDSv2 is enforced, and a generated ED25519 key pair is used for SSH.
- `bootstrap.sh` runs as EC2 user data on first boot: it adds swap, installs Docker, clones this repo and starts the stack with `docker compose`, with containers restarting after a reboot.

```bash
cd deploy/aws
aws login                # or any other AWS credentials
terraform init
terraform apply          # prints api_url and dashboard_url
terraform destroy        # tears everything down
```
