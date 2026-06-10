import os
import random
import math
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
from fastapi import HTTPException

# Load Environment Variables
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", 5))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", 60))

T1 = float(os.getenv("THREAT_TIER1_THRESHOLD", 0.30))
T2 = float(os.getenv("THREAT_TIER2_THRESHOLD", 0.50))
T3 = float(os.getenv("THREAT_TIER3_THRESHOLD", 0.70))
T4 = float(os.getenv("THREAT_TIER4_THRESHOLD", 0.90))

# In-memory Rate Limiter
_rate_limit_store: dict[str, deque] = defaultdict(deque)

def is_rate_limited(client_ip: str) -> bool:
    """
    Checks if client_ip has exceeded RATE_LIMIT_MAX_REQUESTS in the last RATE_LIMIT_WINDOW_SECONDS.
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(seconds=RATE_LIMIT_WINDOW_SECONDS)
    
    # Evict old timestamps
    while _rate_limit_store[client_ip] and _rate_limit_store[client_ip][0] < cutoff:
        _rate_limit_store[client_ip].popleft()
        
    if len(_rate_limit_store[client_ip]) >= RATE_LIMIT_MAX_REQUESTS:
        return True
    
    _rate_limit_store[client_ip].append(now)
    return False

def get_tier(threat_score: float) -> int:
    if threat_score < T1:
        return 0
    elif threat_score < T2:
        return 1
    elif threat_score < T3:
        return 2
    elif threat_score < T4:
        return 3
    else:
        return 4

def perturb_probabilities(probabilities: list[float], sigma: float) -> dict:
    """
    Add Gaussian noise N(0, sigma) to each probability.
    Clip all values to [0, 1].
    Renormalize so they sum to 1.
    Recompute prediction as argmax of new probabilities.
    """
    noisy_probs = []
    for p in probabilities:
        noise = random.gauss(0, sigma)
        clipped = max(0.0, min(1.0, p + noise))
        noisy_probs.append(clipped)
        
    total = sum(noisy_probs)
    if total == 0:
        # Fallback to uniform if noise somehow zeroed everything out
        n = len(probabilities)
        noisy_probs = [1.0 / n] * n
    else:
        noisy_probs = [p / total for p in noisy_probs]
        
    prediction = int(noisy_probs.index(max(noisy_probs)))
    
    return {
        "prediction": prediction,
        "probabilities": noisy_probs
    }

def fake_response() -> dict:
    """
    Generate a random prediction without calling the protected model.
    """
    # 3 classes in Iris
    prediction = random.randint(0, 2)
    # Generate 3 uniform randoms and softmax/normalize them
    raw = [random.uniform(0, 1) for _ in range(3)]
    total = sum(raw)
    probs = [r / total for r in raw]
    
    return {
        "prediction": prediction,
        "probabilities": probs
    }

def apply_defense(client_ip: str, tier: int, real_response: dict | None) -> tuple[dict, str]:
    """
    Returns (response_dict, action_taken_str).
    May raise HTTPException for tiers 2 and 4.
    """
    if tier == 0:
        return real_response, "ALLOW"
        
    elif tier == 1:
        mod_resp = perturb_probabilities(real_response["probabilities"], sigma=0.03)
        # Keep original model name but overwrite prediction/probs
        final_resp = {**real_response, **mod_resp}
        return final_resp, "PERTURB"
        
    elif tier == 2:
        if is_rate_limited(client_ip):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        else:
            mod_resp = perturb_probabilities(real_response["probabilities"], sigma=0.08)
            final_resp = {**real_response, **mod_resp}
            return final_resp, "PERTURB"
            
    elif tier == 3:
        fake_resp = fake_response()
        fake_resp["model"] = "modelguard-fake-v1"
        return fake_resp, "FAKE"
        
    elif tier == 4:
        raise HTTPException(status_code=429, detail="Access denied by ModelGuard")
