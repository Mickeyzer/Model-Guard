"""
train_fingerprint_model.py

Trains the XGBoost fingerprint model on data/training_data.csv and evaluates
it the way it is used: on fresh simulated sessions (different seed, plus two
families never seen in training), scoring every query and mapping scores to
the deployed tier thresholds. Saves data/fingerprint_model.pkl and
data/fingerprint_metrics.json.

Run from the repo root after generate_training_data.py.
"""
import json
import os
import sys
from multiprocessing import Pool

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import traffic_sim as sim  # noqa: E402

# Evaluate at the deployed thresholds (docker-compose.yml), not defense.py's
# defaults; they are read when middleware.defense is imported
for _i, _d in zip(range(1, 5), ("0.05", "0.20", "0.60", "0.85")):
    os.environ.setdefault(f"THREAT_TIER{_i}_THRESHOLD", _d)
from middleware import defense  # noqa: E402

THRESHOLDS = [defense.T1, defense.T2, defense.T3, defense.T4]
EVAL_SEED = 2
EVAL_PER_FAMILY = 40
EVAL_MAX_QUERIES = 200


def deployed_tiers(scores, window_sizes):
    """The middleware's policy: no score below MIN_SCORING_WINDOW queries of
    history, then the streak rule for tiers 3 and 4."""
    scores = [0.0 if n < defense.MIN_SCORING_WINDOW else s for s, n in zip(scores, window_sizes)]
    return np.array([defense.escalate(defense.get_tier(s), scores[:i]) for i, s in enumerate(scores)])


def replay_all(args):
    family, label, session = args
    session = session[:EVAL_MAX_QUERIES]
    out = sim.replay(session)
    return family, label, [n for _, n, _ in out], [feats for _, _, feats in out]


def first(tiers, level):
    hits = np.nonzero(tiers >= level)[0]
    return int(hits[0]) + 1 if len(hits) else None


def evaluate(model, sessions):
    with Pool() as pool:
        replayed = pool.map(replay_all, sessions)

    per_family, all_scores, all_labels = {}, [], []
    for family, label, sizes, feats in replayed:
        X = pd.DataFrame(feats)[sim.FEATURE_COLS]
        scores = model.predict_proba(X)[:, 1]
        tiers = deployed_tiers(scores, sizes)
        all_scores += list(scores)
        all_labels += [label] * len(scores)
        per_family.setdefault((family, label), []).append(tiers)

    report = {"thresholds": THRESHOLDS, "min_scoring_window": defense.MIN_SCORING_WINDOW,
              "harsh_tier_streak": defense.HARSH_TIER_STREAK, "row_auc": float(roc_auc_score(all_labels, all_scores)), "families": {}}
    for (family, label), runs in per_family.items():
        allq = np.concatenate(runs)
        entry = {"label": label, "sessions": len(runs), "queries": int(len(allq))}
        if label == 0:
            entry.update({
                "queries_noised_T1_T2": float(np.mean((allq >= 1) & (allq <= 2))),
                "queries_faked_or_blocked_T3plus": float(np.mean(allq >= 3)),
                "sessions_ever_T3": float(np.mean([r.max() >= 3 for r in runs])),
                "sessions_ever_banned_T4": float(np.mean([r.max() >= 4 for r in runs])),
            })
        else:
            t3 = [first(r, 3) for r in runs]
            t4 = [first(r, 4) for r in runs]
            entry.update({
                "sessions_reaching_T3": float(np.mean([x is not None for x in t3])),
                "sessions_banned_T4": float(np.mean([x is not None for x in t4])),
                "median_queries_to_T3": float(np.median([x for x in t3 if x])) if any(t3) else None,
                "median_queries_to_T4": float(np.median([x for x in t4 if x])) if any(t4) else None,
            })
        report["families"][family] = entry
    return report


def print_report(rep):
    print(f"\nRow-level ROC AUC: {rep['row_auc']:.4f}   thresholds T1-T4: {rep['thresholds']}")
    print(f"\n{'Normal family':<18}{'noised':>9}{'T3+ q':>9}{'ever T3':>9}{'banned':>9}")
    for f, e in rep["families"].items():
        if e["label"] == 0:
            print(f"{f:<18}{e['queries_noised_T1_T2']:>9.1%}{e['queries_faked_or_blocked_T3plus']:>9.1%}"
                  f"{e['sessions_ever_T3']:>9.1%}{e['sessions_ever_banned_T4']:>9.1%}")
    print(f"\n{'Attack family':<18}{'->T3':>9}{'banned':>9}{'q to T3':>9}{'q to T4':>9}")
    for f, e in rep["families"].items():
        if e["label"] == 1:
            fmt = lambda v: f"{v:>9.0f}" if v is not None else f"{'-':>9}"
            print(f"{f:<18}{e['sessions_reaching_T3']:>9.1%}{e['sessions_banned_T4']:>9.1%}"
                  f"{fmt(e['median_queries_to_T3'])}{fmt(e['median_queries_to_T4'])}")


def main():
    df = pd.read_csv(os.path.join(HERE, "training_data.csv"))
    # Every family counts equally whatever its row count, and the two classes
    # carry equal total weight
    fam_rows = df.groupby("family")["label"].transform("size")
    n_fam = df.groupby("label")["family"].transform("nunique")
    weights = 1.0 / (fam_rows * n_fam)
    weights *= len(df) / weights.sum()

    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42,
    )
    model.fit(df[sim.FEATURE_COLS], df["label"], sample_weight=weights)
    print(f"Trained on {len(df)} rows from {df['session_id'].nunique()} sessions")

    print("\nFeature importances:")
    for f, imp in sorted(zip(sim.FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {f:<28}{imp:.3f}")

    sessions = sim.make_sessions(EVAL_SEED, EVAL_PER_FAMILY, include_held_out=True)
    report = evaluate(model, sessions)
    report["held_out_families"] = list(sim.HELD_OUT)
    print_report(report)

    joblib.dump(model, os.path.join(HERE, "fingerprint_model.pkl"))
    with open(os.path.join(HERE, "fingerprint_metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\nSaved data/fingerprint_model.pkl and data/fingerprint_metrics.json")


if __name__ == "__main__":
    main()
