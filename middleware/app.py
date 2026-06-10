import os
import uuid
import traceback
from datetime import datetime, timezone
import httpx
from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from sqlalchemy.orm import Session

from middleware.db import (
    init_db, get_db, log_query, log_defense_event,
    get_last_query_timestamp, get_recent_queries,
    get_ip_stats, get_recent_defense_events
)
from middleware import fingerprint_engine
from middleware import defense

app = FastAPI(title="ModelGuard Middleware")

class PredictRequest(BaseModel):
    features: List[float]

@app.on_event("startup")
def startup():
    print("Initializing Database...")
    init_db()
    model_status = "loaded" if fingerprint_engine._model is not None else "missing"
    print(f"Startup complete. Fingerprint model status: {model_status}")

@app.post("/predict")
def predict(request: Request, body: PredictRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    if len(body.features) != 4:
        raise HTTPException(status_code=422, detail="Features list must contain exactly 4 floats.")
        
    # Step 1
    client_ip = request.client.host if request.client else "127.0.0.1"
    session_id = request.headers.get("X-Session-ID", str(uuid.uuid4()))
    
    try:
        # Step 2
        last_ts = get_last_query_timestamp(db, client_ip)
        # DB stores naive UTC — keep both sides naive for correct subtraction
        now_naive = datetime.utcnow()
        inter_query_time_ms = (now_naive - last_ts).total_seconds() * 1000 if last_ts else 0.0
        
        # Step 3
        recent_queries = get_recent_queries(db, client_ip, window_seconds=60)
        
        # Step 4
        fingerprint_result = fingerprint_engine.classify_query(recent_queries)
        threat_score = fingerprint_result["threat_score"]
        
        # Step 5
        tier = defense.get_tier(threat_score)
        
        # Step 6
        if tier <= 2:
            protected_url = (
                f"http://{os.getenv('PROTECTED_MODEL_HOST','localhost')}"
                f":{os.getenv('PROTECTED_MODEL_PORT','8001')}/predict"
            )
            try:
                with httpx.Client(timeout=5.0) as client:
                    r = client.post(protected_url, json={"features": body.features})
                    r.raise_for_status()
                raw_response = r.json()
            except httpx.RequestError as e:
                print(f"Protected model unreachable: {e}")
                raise HTTPException(503, "Protected model unavailable")
            except Exception as e:
                print(f"Error calling protected model: {e}")
                raise HTTPException(503, "Protected model unavailable")
        else:
            raw_response = None
            
        # Step 7
        try:
            final_response, action_taken = defense.apply_defense(client_ip, tier, raw_response)
        except HTTPException as e:
            if e.status_code == 429:
                # Log the blocked event, then return a JSONResponse (not raise)
                # so that X-Threat-Score and X-Defense-Tier headers are visible
                # to clients (including the simulation script).
                background_tasks.add_task(
                    log_defense_event,
                    db, client_ip, session_id, threat_score, tier,
                    "BLOCK", len(recent_queries)
                )
                return JSONResponse(
                    status_code=429,
                    content={"detail": e.detail},
                    headers={
                        "X-Threat-Score": str(round(threat_score, 4)),
                        "X-Defense-Tier": str(tier),
                        "X-Session-ID": session_id
                    }
                )
            raise

        # Step 8
        raw_pred = raw_response.get("prediction") if raw_response else None
        raw_probs = raw_response.get("probabilities") if raw_response else None
        
        background_tasks.add_task(
            log_query,
            db, client_ip, session_id, body.features,
            raw_pred, raw_probs,
            final_response["prediction"], final_response["probabilities"],
            inter_query_time_ms, threat_score, tier
        )
        
        # Step 9
        background_tasks.add_task(
            log_defense_event,
            db, client_ip, session_id, threat_score, tier,
            action_taken, len(recent_queries)
        )
        
        # Step 10
        return JSONResponse(
            content=final_response,
            headers={
                "X-Threat-Score": str(round(threat_score, 4)),
                "X-Defense-Tier": str(tier),
                "X-Session-ID": session_id
            }
        )
        
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
        "protected_model_url": f"http://{os.getenv('PROTECTED_MODEL_HOST','localhost')}:{os.getenv('PROTECTED_MODEL_PORT','8001')}/predict",
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
