import os
import joblib
import numpy as np
import logging
from middleware.feature_extractor import extract_behavioral_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level model load — happens once when this module is first imported
# ---------------------------------------------------------------------------
_MODEL_PATH = os.getenv(
    "FINGERPRINT_MODEL_PATH",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "fingerprint_model.pkl",
    ),
)

_model = None

try:
    if os.path.exists(_MODEL_PATH):
        _model = joblib.load(_MODEL_PATH)
        logger.info(f"Fingerprint model loaded successfully from {_MODEL_PATH}")
    else:
        logger.warning(
            f"Fingerprint model not found at {_MODEL_PATH}. "
            "Run data/train_fingerprint_model.py to generate it. "
            "Threat scores will be 0.0 until the model is available."
        )
except Exception as e:
    logger.error(f"Failed to load fingerprint model: {e}")
    _model = None

# ---------------------------------------------------------------------------
# Feature order MUST match the column order used during training
# ---------------------------------------------------------------------------
FEATURE_ORDER = [
    "query_count_1min",
    "mean_inter_query_time_ms",
    "feature_space_entropy",
    "prediction_entropy",
    "boundary_proximity_score",
    "input_variance",
    "repetition_ratio",
]


def classify_query(recent_queries: list) -> dict:
    """
    Computes the 7 behavioural features for the query window and runs the
    XGBoost fingerprint model to produce a threat score in [0, 1].

    Returns:
        {
          "threat_score"  : float   — probability of attack (class 1)
          "features_used" : dict    — the 7 computed features (for logging)
          "model_loaded"  : bool    — False when model file was missing at startup
        }
    """
    # 1. Early-exit if model is not available
    if _model is None:
        return {"threat_score": 0.0, "features_used": {}, "model_loaded": False}

    # 2. Compute features using the shared extractor
    features_dict = extract_behavioral_features(recent_queries)

    # 3. Build numpy array in strict FEATURE_ORDER
    try:
        features_array = np.array([[features_dict[f] for f in FEATURE_ORDER]])
    except KeyError as e:
        logger.error(f"Missing feature key during fingerprint classification: {e}")
        return {"threat_score": 0.0, "features_used": features_dict, "model_loaded": True}

    # 4. Predict — always use predict_proba()[:,1], never predict()
    try:
        threat_score = float(_model.predict_proba(features_array)[0][1])
    except Exception as e:
        logger.error(f"Error during XGBoost prediction: {e}")
        threat_score = 0.0

    # 5. Return
    return {
        "threat_score": threat_score,
        "features_used": features_dict,
        "model_loaded": True,
    }
