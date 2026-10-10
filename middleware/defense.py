import os
import random
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque

# Load Environment Variables
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", 5))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", 60))

T1 = float(os.getenv("THREAT_TIER1_THRESHOLD", 0.30))
T2 = float(os.getenv("THREAT_TIER2_THRESHOLD", 0.50))
T3 = float(os.getenv("THREAT_TIER3_THRESHOLD", 0.70))
T4 = float(os.getenv("THREAT_TIER4_THRESHOLD", 0.90))

# The protected model is a 100-tree random forest, so its probabilities are
# always multiples of 0.01. Altered responses are rounded the same way;
# otherwise their extra decimals would tell an attacker they were altered.
PROB_DECIMALS = 2

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

def _round_probs(probs: list[float]) -> list[float]:
    """Round to PROB_DECIMALS and push any rounding drift onto the largest
    class so the result still sums to exactly 1."""
    rounded = [round(p, PROB_DECIMALS) for p in probs]
    top = rounded.index(max(rounded))
    rounded[top] = round(rounded[top] + 1.0 - sum(rounded), PROB_DECIMALS)
    return rounded

def perturb_probabilities(probabilities: list[float], sigma: float) -> dict:
    """
    Add Gaussian noise N(0, sigma) to each probability.
    Clip all values to [0, 1].
    Renormalize so they sum to 1 and round like the real model's output.
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

    noisy_probs = _round_probs(noisy_probs)
    prediction = int(noisy_probs.index(max(noisy_probs)))

    return {
        "prediction": prediction,
        "probabilities": noisy_probs
    }

def fake_response(n_classes: int = 3) -> dict:
    """
    Generate a random prediction that looks like a real one: confident in
    one class (often fully, as the random forest is), with the prediction
    equal to the argmax of the probabilities.
    """
    prediction = random.randrange(n_classes)
    top = min(1.0, random.uniform(0.6, 1.15))
    weights = [random.random() for _ in range(n_classes - 1)]
    total = sum(weights) or 1.0
    rest = iter([(1.0 - top) * w / total for w in weights])
    probs = [top if i == prediction else next(rest) for i in range(n_classes)]

    return {
        "prediction": prediction,
        "probabilities": _round_probs(probs)
    }

def apply_defense(client_ip: str, tier: int, real_response: dict | None) -> tuple[dict | None, str]:
    """
    Returns (response_dict, action_taken_str). The response is None when the
    request must be refused (action "RATE_LIMIT" or "BLOCK").
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
            return None, "RATE_LIMIT"
        mod_resp = perturb_probabilities(real_response["probabilities"], sigma=0.08)
        final_resp = {**real_response, **mod_resp}
        return final_resp, "PERTURB"

    elif tier == 3:
        # Same shape and model name as a real answer, so the client cannot
        # tell it is being fed fake predictions
        n_classes = len(real_response["probabilities"]) if real_response else 3
        fake_resp = fake_response(n_classes)
        if real_response and "model" in real_response:
            fake_resp["model"] = real_response["model"]
        return fake_resp, "FAKE"

    return None, "BLOCK"
