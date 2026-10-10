import math
import os
import uuid
import traceback
import httpx
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from sqlalchemy.orm import Session

from middleware.db import (
    init_db, get_db, log_request, utcnow,
    get_last_query_timestamp, get_recent_queries,
    get_active_ban, register_ban,
    get_ip_stats, get_recent_defense_events
)
from middleware import fingerprint_engine
from middleware import defense

app = FastAPI(title="ModelGuard Middleware")

# Iris measurements are in centimetres; this bound is generous on purpose so
# out-of-distribution sweeps are still scored, while junk input is rejected.
FEATURE_MIN, FEATURE_MAX = 0.0, 100.0

class PredictRequest(BaseModel):
    features: List[float]

def validate_features(features: List[float]) -> None:
    # Checked here rather than in a pydantic validator: FastAPI's validation
    # error echoes the input back, and a NaN in it cannot be JSON-encoded,
    # which turned a bad request into a 500.
    if len(features) != 4:
        raise HTTPException(status_code=422, detail="Features list must contain exactly 4 floats.")
    # NaN/inf would crash the feature extractor for this client's whole window
    if not all(math.isfinite(x) and FEATURE_MIN <= x <= FEATURE_MAX for x in features):
        raise HTTPException(
            status_code=422,
            detail=f"Features must be finite numbers between {FEATURE_MIN} and {FEATURE_MAX}."
        )

def protected_model_url() -> str:
    return (
        f"http://{os.getenv('PROTECTED_MODEL_HOST','localhost')}"
        f":{os.getenv('PROTECTED_MODEL_PORT','8001')}/predict"
    )

def call_protected_model(features: List[float]) -> dict:
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.post(protected_model_url(), json={"features": features})
            r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Error calling protected model: {e}")
        raise HTTPException(503, "Protected model unavailable")

def response_headers(threat_score: float, tier: int, session_id: str) -> dict:
    headers = {"X-Session-ID": session_id}
    # The score tells an attacker how close they are to each tier, so it is
    # only sent when explicitly enabled (e.g. for the demo script).
    if os.getenv("EXPOSE_THREAT_HEADERS", "0") == "1":
        headers["X-Threat-Score"] = str(round(threat_score, 4))
        headers["X-Defense-Tier"] = str(tier)
    return headers

@app.on_event("startup")
def startup():
    print("Initializing Database...")
    init_db()
    model_status = "loaded" if fingerprint_engine._model is not None else "missing"
    print(f"Startup complete. Fingerprint model status: {model_status}")

@app.post("/predict")
def predict(request: Request, body: PredictRequest, db: Session = Depends(get_db)):
    validate_features(body.features)

    # Step 1
    client_ip = request.client.host if request.client else "127.0.0.1"
    session_id = request.headers.get("X-Session-ID", str(uuid.uuid4()))

    try:
        # Step 2
        last_ts = get_last_query_timestamp(db, client_ip)
        inter_query_time_ms = (utcnow() - last_ts).total_seconds() * 1000 if last_ts else 0.0

        # Step 3
        recent_queries = get_recent_queries(db, client_ip, window_seconds=60)

        # Step 4
        fingerprint_result = fingerprint_engine.classify_query(recent_queries)
        threat_score = fingerprint_result["threat_score"]

        # Step 5
        tier = defense.get_tier(threat_score)

        # Step 6: an active ban blocks whatever the current score is. Reaching
        # tier 4 while not banned adds a strike and starts a longer ban.
        ban = get_active_ban(db, client_ip)
        if ban is None and tier == 4:
            ban = register_ban(db, client_ip)

        # Step 7: tiers 0-3 query the real model. Tier 3 still answers with a
        # fake, but the real output is logged so features reflect the true
        # model's behaviour rather than ModelGuard's own fakes.
        raw_response = None
        retry_after = None
        if ban is not None:
            tier = 4
            final_response, action_taken = None, "BLOCK"
            retry_after = max(1, math.ceil((ban.banned_until - utcnow()).total_seconds()))
        else:
            raw_response = call_protected_model(body.features)
            final_response, action_taken = defense.apply_defense(client_ip, tier, raw_response)

        # Step 8: log before responding (blocked requests too), so the next
        # request's window includes this one.
        log_request(
            db, client_ip, session_id, body.features,
            raw_response.get("prediction") if raw_response else None,
            raw_response.get("probabilities") if raw_response else None,
            final_response["prediction"] if final_response else None,
            final_response["probabilities"] if final_response else None,
            inter_query_time_ms, threat_score, tier,
            action_taken, len(recent_queries)
        )

        # Step 9
        headers = response_headers(threat_score, tier, session_id)
        if final_response is None:
            if retry_after is not None:
                headers["Retry-After"] = str(retry_after)
                detail = "Access denied by ModelGuard"
            else:
                detail = "Rate limit exceeded"
            return JSONResponse(status_code=429, content={"detail": detail}, headers=headers)

        return JSONResponse(content=final_response, headers=headers)

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Middleware Error")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "fingerprint_model_loaded": fingerprint_engine._model is not None,
        "db": "connected"
    }

@app.get("/stats/{client_ip}")
def get_stats(client_ip: str, db: Session = Depends(get_db)):
    return get_ip_stats(db, client_ip)

@app.get("/admin/logs")
def get_logs(limit: int = 50, db: Session = Depends(get_db)):
    events = get_recent_defense_events(db, limit=limit)
    return [
        {
            "id": e.id,
            "timestamp": e.timestamp.isoformat(),
            "client_ip": e.client_ip,
            "threat_score": e.threat_score,
            "tier_applied": e.tier_applied,
            "action_taken": e.action_taken,
        }
        for e in events
    ]
