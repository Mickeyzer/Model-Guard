"""
train_fingerprint_model.py

Loads the simulated training_data.csv, trains an XGBoost binary classifier,
evaluates it, and saves the model to data/fingerprint_model.pkl.
"""
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from xgboost import XGBClassifier

FEATURE_COLS = [
    "query_count_1min",
    "mean_inter_query_time_ms",
    "feature_space_entropy",
    "prediction_entropy",
    "boundary_proximity_score",
    "input_variance",
    "repetition_ratio",
]

def main():
    csv_path = os.path.join("data", "training_data.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Training data not found at {csv_path}. "
            "Run data/generate_training_data.py first."
        )

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  Rows: {len(df)}  |  Columns: {list(df.columns)}")

    X = df[FEATURE_COLS]
    y = df["label"]

    # --- Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")

    # --- Train ---
    print("\nTraining XGBClassifier...")
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    # --- Evaluate ---
    y_pred = model.predict(X_test)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall    = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1        = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    cm        = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Accuracy          : {accuracy:.4f}")
    print(f"  Precision (attack): {precision:.4f}")
    print(f"  Recall    (attack): {recall:.4f}")
    print(f"  F1        (attack): {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"    FN={cm[1][0]}  TP={cm[1][1]}")

    # --- Feature importances ---
    importances = sorted(
        zip(FEATURE_COLS, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )
    print(f"\n  Feature Importances (descending):")
    for feat, imp in importances:
        bar = "#" * int(imp * 40)
        print(f"    {feat:<35} {imp:.4f} {bar}")

    # --- Warning if F1 is low ---
    if f1 < 0.80:
        weak = [feat for feat, imp in importances if imp < 0.05]
        print("\n[WARNING] F1 on attack class is below 0.80!")
        print("   The following features have very low importance and may be weakening the model:")
        for feat in weak:
            print(f"     - {feat}")
        print("   Consider improving their computation in feature_extractor.py and re-generating data.")

    # --- Save ---
    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "fingerprint_model.pkl")
    joblib.dump(model, out_path)
    print(f"\n[OK] Model saved to {out_path}")


if __name__ == "__main__":
    main()
