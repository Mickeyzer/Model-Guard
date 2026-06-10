"""
generate_training_data.py  (v3)

Generates feature rows DIRECTLY from calibrated distributions.
Distributions are designed so that overall XGBoost accuracy lands
naturally in [88%, 95%] — no tree depth changes needed.

Key design decisions:
  - Type 2 (slow attacker) is heavily overlapping with normal (250 sessions).
    The model struggles to separate these → drags accuracy below 95%.
  - Types 3-5 are less extreme than naive implementations, creating
    genuine ambiguity and graduated tier coverage.
  - Total = 1000 rows (300 normal + 700 attacks across 4 subtypes).
"""
import os
import sys
import csv
import random
import statistics
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

random.seed(42)

# ---------------------------------------------------------------------------
# Tier thresholds (must match defense.py defaults)
# ---------------------------------------------------------------------------
T1 = float(os.getenv("THREAT_TIER1_THRESHOLD", 0.30))
T2 = float(os.getenv("THREAT_TIER2_THRESHOLD", 0.50))
T3 = float(os.getenv("THREAT_TIER3_THRESHOLD", 0.70))
T4 = float(os.getenv("THREAT_TIER4_THRESHOLD", 0.90))

FEATURE_COLS = [
    "query_count_1min",
    "mean_inter_query_time_ms",
    "feature_space_entropy",
    "prediction_entropy",
    "boundary_proximity_score",
    "input_variance",
    "repetition_ratio",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def clip(val, lo=0.0, hi=None):
    val = max(lo, val)
    if hi is not None:
        val = min(hi, val)
    return val

def u(lo, hi):   return random.uniform(lo, hi)
def ri(lo, hi):  return random.randint(lo, hi)


# ---------------------------------------------------------------------------
# Session generators — features sampled directly from controlled distributions
# ---------------------------------------------------------------------------

def normal_session() -> dict:
    """Type 1 — Normal user (label=0, 300 sessions)."""
    return {
        "session_type":              1,
        "query_count_1min":         float(ri(3, 18)),
        "mean_inter_query_time_ms": u(8000, 30000),
        "feature_space_entropy":    u(0.5, 2.0),
        "prediction_entropy":       clip(u(0.0, 1.2)),
        "boundary_proximity_score": u(0.0, 0.35),
        "input_variance":           u(0.2, 2.0),
        "repetition_ratio":         u(0.0, 0.06),
        "label": 0,
    }


def slow_attacker_session() -> dict:
    """
    Type 2 — Slow / early-stage attacker (label=1, 250 sessions).
    Deliberately overlaps with normal to force the model to produce
    intermediate threat scores (tiers 1-2) for ambiguous cases.
    """
    return {
        "session_type":              2,
        "query_count_1min":         float(ri(5, 30)),    # overlaps normal [3,18] even more
        "mean_inter_query_time_ms": u(7000, 25000),      # overlaps normal [8000,30000]
        "feature_space_entropy":    u(0.8, 2.0),         # overlaps normal [0.5,1.6]
        "prediction_entropy":       clip(u(0.3, 1.3)),
        "boundary_proximity_score": u(0.05, 0.35),       # overlaps normal [0.0,0.25]
        "input_variance":           u(0.5, 2.0),         # overlaps normal [0.2,1.5]
        "repetition_ratio":         u(0.0, 0.05),
        "label": 1,
    }


def naive_sweep_session() -> dict:
    """Type 3 — Naive grid sweep (label=1, 150 sessions). Less extreme than v1."""
    return {
        "session_type":              3,
        "query_count_1min":         float(ri(30, 65)),
        "mean_inter_query_time_ms": u(200, 2000),
        "feature_space_entropy":    u(1.8, 2.8),
        "prediction_entropy":       clip(u(1.1, 1.58)),
        "boundary_proximity_score": u(0.25, 0.60),
        "input_variance":           u(2.0, 4.5),
        "repetition_ratio":         u(0.0, 0.01),
        "label": 1,
    }


def boundary_prober_session() -> dict:
    """Type 4 — Boundary prober (label=1, 150 sessions). Less extreme than v1."""
    return {
        "session_type":              4,
        "query_count_1min":         float(ri(20, 45)),
        "mean_inter_query_time_ms": u(500, 3000),
        "feature_space_entropy":    u(1.4, 2.4),
        "prediction_entropy":       clip(u(0.9, 1.58)),
        "boundary_proximity_score": u(0.45, 0.85),
        "input_variance":           u(1.3, 3.2),
        "repetition_ratio":         u(0.0, 0.03),
        "label": 1,
    }


def knockoff_session() -> dict:
    """Type 5 — Knockoff high volume (label=1, 150 sessions). Less extreme than v1."""
    return {
        "session_type":              5,
        "query_count_1min":         float(ri(40, 90)),
        "mean_inter_query_time_ms": u(50, 500),
        "feature_space_entropy":    u(2.0, 3.0),
        "prediction_entropy":       clip(u(1.2, 1.58)),
        "boundary_proximity_score": u(0.20, 0.55),
        "input_variance":           u(2.5, 5.5),
        "repetition_ratio":         u(0.0, 0.015),
        "label": 1,
    }


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------

def _col_stats(rows, feat, session_type=None, label=None):
    if session_type is not None:
        vals = [r[feat] for r in rows if r["session_type"] == session_type]
    else:
        vals = [r[feat] for r in rows if r["label"] == label]
    mean = statistics.mean(vals)
    std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, std


def check_1_overlap(rows):
    """
    Compare Normal (Type 1) vs Slow Attacker (Type 2) only.
    Type 2 is the critical boundary type designed to overlap with normal.
    """
    print("\n[CHECK 1] Distribution overlap: Normal vs Slow Attacker (type 2)...")
    failed = []
    for feat in FEATURE_COLS:
        m1, s1 = _col_stats(rows, feat, session_type=1)
        m2, s2 = _col_stats(rows, feat, session_type=2)

        if m1 <= m2:
            lo_m, lo_s, hi_m, hi_s = m1, s1, m2, s2
        else:
            lo_m, lo_s, hi_m, hi_s = m2, s2, m1, s1

        overlaps = (lo_m + lo_s) > (hi_m - hi_s)
        status = "OK" if overlaps else "FAIL"
        print(f"  {feat:<35} lower+std={lo_m+lo_s:.4f}  upper-std={hi_m-hi_s:.4f}  [{status}]")
        if not overlaps:
            failed.append(feat)

    if failed:
        print(f"\n[ERROR] Check 1 FAILED for: {failed}")
        sys.exit(1)
    print("  Check 1 passed.")


def check_2_feature_importance(model):
    """All 7 features must have importance > 0."""
    print("\n[CHECK 2] Feature importances > 0...")
    zero_feats = [
        feat for feat, imp in zip(FEATURE_COLS, model.feature_importances_)
        if imp == 0.0
    ]
    for feat, imp in sorted(zip(FEATURE_COLS, model.feature_importances_),
                            key=lambda x: x[1], reverse=True):
        bar = "#" * int(imp * 40)
        print(f"  {feat:<35} {imp:.4f}  {bar}")

    if zero_feats:
        print(f"\n[ERROR] Check 2 FAILED. Zero-importance: {zero_feats}")
        sys.exit(1)
    print("  Check 2 passed.")


def check_3_accuracy(model, X_test, y_test):
    """Accuracy must be in [88%, 95%]."""
    print("\n[CHECK 3] Model accuracy in range [88%, 95%]...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  Accuracy: {acc:.4f}")

    if acc > 0.95:
        print("[ERROR] Check 3 FAILED — accuracy above 95%.")
        print("  Increase Type 2 session count or widen its distributions to create more overlap.")
        sys.exit(1)
    elif acc < 0.88:
        print("[ERROR] Check 3 FAILED — accuracy below 88%.")
        print("  Reduce Type 2 session count or tighten attack distributions.")
        sys.exit(1)
    print("  Check 3 passed.")


def check_4_tier_coverage(model, X_test_clean):
    """Tiers 1, 2, and 3 must each have >= 5% of test samples."""
    print("\n[CHECK 4] Tier coverage on clean test set (tiers 1-3 >= 5% each)...")
    scores = model.predict_proba(X_test_clean)[:, 1]
    n = len(scores)

    def get_tier(s):
        if s < T1:  return 0
        elif s < T2: return 1
        elif s < T3: return 2
        elif s < T4: return 3
        else:        return 4

    tier_counts = {i: 0 for i in range(5)}
    for s in scores:
        tier_counts[get_tier(s)] += 1

    print(f"  {'Tier':<8} {'Count':<8} {'Pct':>6}")
    for t in range(5):
        pct = tier_counts[t] / n * 100
        print(f"  {t:<8} {tier_counts[t]:<8} {pct:>5.1f}%")

    failed = [t for t in [1, 2, 3] if tier_counts[t] / n < 0.05]
    if failed:
        print(f"\n[ERROR] Check 4 FAILED. Tiers {failed} have < 5% coverage.")
        print("  Increase Type 2 session count to add more ambiguous samples.")
        sys.exit(1)
    print("  Check 4 passed.")


def check_5_no_negative_entropy(rows):
    """prediction_entropy must be >= 0 everywhere."""
    print("\n[CHECK 5] No negative prediction_entropy values...")
    bad = [i for i, r in enumerate(rows) if r["prediction_entropy"] < 0.0]
    if bad:
        print(f"[ERROR] Check 5 FAILED. {len(bad)} rows have negative prediction_entropy.")
        sys.exit(1)
    print("  Check 5 passed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rows = []

    counts = {1: 400, 2: 350, 3: 83, 4: 83, 5: 84}
    generators = {
        1: normal_session,
        2: slow_attacker_session,
        3: naive_sweep_session,
        4: boundary_prober_session,
        5: knockoff_session,
    }
    labels = {1: "Normal user", 2: "Slow attacker",
              3: "Naive sweep", 4: "Boundary prober", 5: "Knockoff"}

    print("Generating sessions...")
    for t, fn in generators.items():
        print(f"  Type {t} - {labels[t]:<20}: {counts[t]}")
        for _ in range(counts[t]):
            rows.append(fn())

    total  = len(rows)
    n_norm = sum(1 for r in rows if r["label"] == 0)
    n_atk  = sum(1 for r in rows if r["label"] == 1)

    print(f"\nClass balance:")
    print(f"  Normal  (0): {n_norm} ({n_norm/total*100:.1f}%)")
    print(f"  Attack  (1): {n_atk}  ({n_atk/total*100:.1f}%)")

    print("\nFeature stats (per class):")
    print(f"  {'Feature':<35} {'Normal mean':>12} {'Attack mean':>12}")
    for feat in FEATURE_COLS:
        m0, _ = _col_stats(rows, feat, label=0)
        m1, _ = _col_stats(rows, feat, label=1)
        print(f"  {feat:<35} {m0:>12.4f} {m1:>12.4f}")

    # ----- Check 1 & 5 (before stripping session_type) -----
    check_1_overlap(rows)
    check_5_no_negative_entropy(rows)

    # Strip session_type before training
    for r in rows:
        r.pop("session_type", None)

    # ----- Build DataFrames -----
    df = pd.DataFrame(rows)
    X  = df[FEATURE_COLS]
    y  = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Save clean copy for check 4 (must use unmodified data)
    X_test_clean = X_test.copy()

    # ----- Train inline model -----
    print("\nTraining inline XGBoost for validation...")
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    # ----- Checks 2, 3, 4 -----
    check_2_feature_importance(model)
    check_3_accuracy(model, X_test_clean, y_test)
    check_4_tier_coverage(model, X_test_clean)

    # ----- Save CSV -----
    os.makedirs("data", exist_ok=True)
    csv_path = os.path.join("data", "training_data.csv")
    columns  = FEATURE_COLS + ["label"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[OK] All 5 checks passed. Saved {total} rows to {csv_path}")
    print("     Run data/train_fingerprint_model.py to train and save the final model.")


if __name__ == "__main__":
    main()
