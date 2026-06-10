from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import os

app = FastAPI(title="Victim Model API")

# Global variable to hold the model
model = None
MODEL_VERSION = "iris-random-forest-v1"

class PredictRequest(BaseModel):
    features: List[float]

@app.on_event("startup")
def startup_event():
    global model
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        model = joblib.load(model_path)
    else:
        print(f"WARNING: Model file not found at {model_path}. Please run train_and_save.py first.")

@app.post("/predict")
def predict(request: PredictRequest):
    # Manual validation for exactly 4 floats as requested (no Pydantic Field restrictions)
    if len(request.features) != 4:
        raise HTTPException(
            status_code=422, 
            detail="Features list must contain exactly 4 floats."
        )
    
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded."
        )

    # Scikit-learn expects a 2D array: [n_samples, n_features]
    features_2d = [request.features]
    
    # Predict is synchronous, so this route is correctly defined as `def` (runs in threadpool)
    prediction = int(model.predict(features_2d)[0])
    probabilities = model.predict_proba(features_2d)[0].tolist()

    return {
        "prediction": prediction,
        "probabilities": probabilities,
        "model": MODEL_VERSION
    }

@app.get("/health")
def health():
    status = "ok" if model is not None else "model_missing"
    return {
        "status": status,
        "model": MODEL_VERSION,
        "features_expected": 4
    }
