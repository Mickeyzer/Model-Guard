"""
generate_training_data.py

Builds data/training_data.csv by simulating client sessions (data/traffic_sim.py)
and replaying them through the middleware's real feature extractor.

Each session contributes up to 40 rows, one per sampled query, scored on the
window the middleware would have seen at that moment. Attack rows whose window
holds fewer than 3 queries are dropped: at that point an attack looks exactly
like a new normal client, so labelling it "attack" would only teach noise.

Run from the repo root: python data/generate_training_data.py
"""
import csv
import os
import random
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import traffic_sim as sim  # noqa: E402

SEED = 1
SESSIONS_PER_FAMILY = 120
ROWS_PER_SESSION = 40


def rows_for(args):
    sid, (family, label, session) = args
    rng = random.Random(SEED * 100_003 + sid)
    idx = sorted(rng.sample(range(len(session)), min(ROWS_PER_SESSION, len(session))))
    out = []
    for i, n_window, feats in sim.replay(session, idx):
        if label == 1 and n_window < 3:
            continue
        out.append({"session_id": sid, "family": family, **feats, "label": label})
    return out


def main():
    sessions = sim.make_sessions(SEED, SESSIONS_PER_FAMILY)
    print(f"Simulated {len(sessions)} sessions ({SESSIONS_PER_FAMILY} per family)")
    with Pool() as pool:
        rows = [r for chunk in pool.map(rows_for, enumerate(sessions)) for r in chunk]

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_data.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["session_id", "family"] + sim.FEATURE_COLS + ["label"])
        w.writeheader()
        w.writerows(rows)

    n_attack = sum(r["label"] for r in rows)
    print(f"Saved {len(rows)} rows to {path} ({len(rows) - n_attack} normal, {n_attack} attack)")


if __name__ == "__main__":
    main()
