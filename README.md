# ModelGuard

ModelGuard is middleware that sits in front of an ML model API and defends it against model-extraction attacks (clients querying the API to copy its decision boundaries). Instead of a static rate limit, it scores each client's recent query behaviour and escalates through tiered responses.

## Features

- **Behavioural fingerprinting:** An XGBoost classifier scores each request from 7 features of the client's last 5 minutes of queries (up to 100): request rate, timing, input-space entropy and variance, prediction entropy, boundary proximity and repetition. A client is not scored until it has 5 queries of history.
- **Dynamic Tiered Defense:**
  - `T0 ALLOW`: Normal traffic passes through seamlessly.
  - `T1 / T2 PERTURB`: Injects dynamic noise into probabilities to corrupt an attacker's dataset (tier 2 also rate-limits).
  - `T3 FAKE`: Returns synthetic predictions in exactly the real model's format, so the attacker cannot tell they are fake.
  - Tiers 1-2 apply immediately; tiers 3-4 need 3 consecutive high scores, so one odd request never earns a real user fake answers or a ban.
  - `T4 BLOCK`: Bans the client IP with escalating durations (5 min, then 1 h, then 24 h). The ban holds even if the score drops, and blocked requests stay in the behavioural window.
- **No signals for attackers:** Threat scores and tiers are not sent to clients unless `EXPOSE_THREAT_HEADERS=1`. Behavioural features are computed from the real model's outputs, so ModelGuard's own noise and fakes never inflate a client's score.
- **Live dashboard:** A Streamlit view of what each client actually received over time (real, noised, fake or refused), a per-client table with peak threat score and ban status, and the raw event log. Client IPs are masked to /24.

## Evaluation

The detector is trained and evaluated on **simulated** traffic, not real users (`data/traffic_sim.py`). Each simulated session is a sequence of queries with timestamps; every query goes through the real protected model and the middleware's own feature extractor, on the same window the middleware sees. There are 5 normal-client families (casual use, steady app integration, batch scoring, "what-if" tweaking, polling) and 6 attack families (fast and slow grid sweeps, boundary bisection, uniform random queries, high-volume noisy surrogate data, slow random queries inside the real data range).

Evaluation uses fresh sessions from a different seed plus two families never seen in training (a busier 25-40 req/min app and a Latin-hypercube sweep), with the deployed thresholds and escalation rule: 40 sessions per family, about 50,000 scored queries. From `data/fingerprint_metrics.json`:

| Attack family | Sessions banned | Median queries to ban |
|---|---|---|
| Grid sweep, fast / slow (2-20 req/min) | 100% / 100% | 8 / 8 |
| Boundary bisection | 100% | 8 |
| Uniform random | 100% | 8 |
| High-volume noisy surrogate | 100% | 9 |
| Slow random inside data range | 100% | 8 |
| Latin-hypercube sweep (held out) | 100% | 8 |

| Normal family | Answers noised | Sessions ever banned |
|---|---|---|
| Casual, batch, what-if, polling | 0-3% | 0% |
| Steady app, 3-25 req/min | 5% | 5% |
| Busy app, 25-40 req/min (held out) | 29% | 5% |

Limits: these are simulated behaviours, so real traffic will differ. Busy legitimate integrations are the weak spot (about 1 in 20 sessions still hit a first, 5-minute ban, and heavy apps see noised answers on almost a third of requests). A client slower than about one query a minute leaves too little history in the 5-minute window to be scored.

To regenerate: `python data/generate_training_data.py` then `python data/train_fingerprint_model.py`.

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
