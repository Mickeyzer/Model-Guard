"""
traffic_sim.py

Simulates client sessions against the protected Iris model and turns them into
behavioural feature rows exactly the way the middleware does: each query is
scored on the window of that client's earlier queries (last 5 minutes, at most
100), using middleware.feature_extractor and the real protected model's
outputs. Nothing here samples feature values directly.

A session is a list of (t_seconds, [4 floats]). Generators return
(family, label, session).
"""
import os
import random
import sys
from datetime import datetime, timedelta
from types import SimpleNamespace

import joblib
import numpy as np
from sklearn.datasets import load_iris

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from middleware.feature_extractor import extract_behavioral_features  # noqa: E402

FEATURE_COLS = [
    "query_count_1min",
    "mean_inter_query_time_ms",
    "feature_space_entropy",
    "prediction_entropy",
    "boundary_proximity_score",
    "input_variance",
    "repetition_ratio",
]

# Must match middleware.db.get_recent_queries defaults
WINDOW_SECONDS = 300
WINDOW_LIMIT = 100

IRIS = load_iris()
IRIS_X = IRIS.data
IRIS_Y = IRIS.target
IRIS_MIN = IRIS_X.min(axis=0)
IRIS_MAX = IRIS_X.max(axis=0)
_model = joblib.load(os.path.join(ROOT, "protected_model", "model.pkl"))


def _clip(x):
    return [float(min(max(v, 0.1), 10.0)) for v in x]


def iris_like(rng, noise=0.15, classes=None):
    idx = rng.randrange(len(IRIS_X))
    if classes is not None:
        idx = rng.choice([i for i in range(len(IRIS_X)) if IRIS_Y[i] in classes])
    return _clip([v + rng.gauss(0, noise) for v in IRIS_X[idx]])


def _times(rng, n, gap_fn, start=0.0):
    t, out = start, []
    for _ in range(n):
        out.append(t)
        t += gap_fn()
    return out


# ---------------------------------------------------------------------------
# Normal clients (label 0)
# ---------------------------------------------------------------------------

def casual(rng):
    """A person trying the API by hand."""
    mean_gap = rng.uniform(15, 90)
    n = rng.randint(3, 40)
    ts = _times(rng, n, lambda: 2 + rng.expovariate(1 / mean_gap))
    return [(t, iris_like(rng)) for t in ts]


def app_integration(rng, rate=None):
    """A backend scoring real measurements at a steady rate (3-25 per minute)."""
    rate = rate or rng.uniform(3, 25)
    n = int(rate * rng.uniform(5, 20))
    skew = rng.sample([0, 1, 2], rng.randint(1, 3))
    ts = _times(rng, n, lambda: rng.expovariate(rate / 60))
    return [(t, iris_like(rng, classes=skew)) for t in ts]


def heavy_app(rng):
    """Held-out family: a busier integration, 25-40 per minute."""
    return app_integration(rng, rate=rng.uniform(25, 40))


def batch_scorer(rng):
    """Bursts of real rows sent back to back, then idle. Some duplicates."""
    out, t = [], 0.0
    dup = rng.uniform(0, 0.2)
    for _ in range(rng.randint(1, 4)):
        sent = []
        for _ in range(rng.randint(10, 60)):
            x = rng.choice(sent) if sent and rng.random() < dup else iris_like(rng, noise=0.05)
            sent.append(x)
            out.append((t, x))
            t += rng.uniform(0.05, 0.5)
        t += rng.uniform(120, 900)
    return out


def what_if(rng):
    """Someone nudging one measurement at a time around a real flower."""
    base = iris_like(rng, noise=0.05)
    out, t = [], 0.0
    for _ in range(rng.randint(10, 40)):
        x = list(base)
        d = rng.randrange(4)
        x[d] += rng.uniform(-1.0, 1.0)
        if rng.random() < 0.3:
            base = _clip(x)
        out.append((t, _clip(x)))
        t += rng.uniform(3, 20)
    return out


def poller(rng):
    """The same one to three inputs re-sent every few seconds (health checks)."""
    fixed = [iris_like(rng) for _ in range(rng.randint(1, 3))]
    ts = _times(rng, rng.randint(10, 60), lambda: rng.uniform(5, 30))
    return [(t, rng.choice(fixed)) for t in ts]


# ---------------------------------------------------------------------------
# Extraction attacks (label 1)
# ---------------------------------------------------------------------------

def _grid(rng):
    dims = rng.sample(range(4), rng.randint(2, 4))
    steps = rng.randint(4, 10)
    lo, hi = rng.uniform(0.1, 2.0), rng.uniform(5.0, 9.0)
    axis = np.linspace(lo, hi, steps)
    fixed = iris_like(rng)
    pts = []
    for combo in np.array(np.meshgrid(*[axis] * len(dims))).T.reshape(-1, len(dims)):
        x = list(fixed)
        for d, v in zip(dims, combo):
            x[d] = float(v)
        pts.append(_clip(x))
    return pts


def grid_fast(rng):
    pts = _grid(rng)[:150]
    ts = _times(rng, len(pts), lambda: rng.uniform(0.05, 1.0))
    return list(zip(ts, pts))


def grid_slow(rng):
    """Same sweep at 2-20 queries a minute, the case the old model missed."""
    pts = _grid(rng)[:120]
    gap = rng.uniform(3, 30)
    ts = _times(rng, len(pts), lambda: gap * rng.uniform(0.7, 1.3))
    return list(zip(ts, pts))


def boundary_probe(rng):
    """Bisects between flowers of different classes to find decision edges."""
    out, t = [], 0.0
    gap = rng.uniform(0.2, 10)
    target = rng.randint(30, 120)
    while len(out) < target:
        a, b = rng.sample(range(len(IRIS_X)), 2)
        if IRIS_Y[a] == IRIS_Y[b]:
            continue
        lo, hi = np.array(IRIS_X[a]), np.array(IRIS_X[b])
        ca = IRIS_Y[a]
        for _ in range(rng.randint(5, 10)):
            mid = (lo + hi) / 2
            out.append((t, _clip(mid)))
            t += gap * rng.uniform(0.7, 1.3)
            if int(_model.predict([mid])[0]) == ca:
                lo = mid
            else:
                hi = mid
    return out


def random_uniform(rng):
    """Synthetic queries spread uniformly over a wide box."""
    hi = rng.uniform(6, 10)
    n = rng.randint(30, 150)
    gap = rng.uniform(0.05, 5)
    ts = _times(rng, n, lambda: gap * rng.uniform(0.5, 1.5))
    return [(t, _clip([rng.uniform(0.1, hi) for _ in range(4)])) for t in ts]


def surrogate_flood(rng):
    """Knockoff-style: plausible inputs with heavy noise, sustained high volume."""
    rate = rng.uniform(20, 200)
    noise = rng.uniform(0.3, 1.0)
    n = min(int(rate * rng.uniform(5, 15)), 400)
    ts = _times(rng, n, lambda: rng.expovariate(rate / 60))
    return [(t, iris_like(rng, noise=noise)) for t in ts]


def mimicry_slow(rng):
    """Random points inside the real flowers' range, at a human-ish 3-12 per minute."""
    rate = rng.uniform(3, 12)
    n = rng.randint(40, 120)
    ts = _times(rng, n, lambda: rng.expovariate(rate / 60))
    return [(t, _clip([rng.uniform(lo, hi) for lo, hi in zip(IRIS_MIN, IRIS_MAX)])) for t in ts]


def lhs_sweep(rng):
    """Held-out family: Latin-hypercube coverage instead of a grid, moderate pace."""
    n = rng.randint(40, 120)
    hi = rng.uniform(6, 9)
    cols = []
    for _ in range(4):
        edges = np.linspace(0.1, hi, n + 1)
        col = [rng.uniform(edges[i], edges[i + 1]) for i in range(n)]
        rng.shuffle(col)
        cols.append(col)
    gap = rng.uniform(0.5, 15)
    ts = _times(rng, n, lambda: gap * rng.uniform(0.7, 1.3))
    return [(t, _clip([cols[d][i] for d in range(4)])) for i, t in enumerate(ts)]


NORMAL = {"casual": casual, "app_integration": app_integration, "batch_scorer": batch_scorer,
          "what_if": what_if, "poller": poller}
ATTACK = {"grid_fast": grid_fast, "grid_slow": grid_slow, "boundary_probe": boundary_probe,
          "random_uniform": random_uniform, "surrogate_flood": surrogate_flood, "mimicry_slow": mimicry_slow}
HELD_OUT = {"heavy_app": (heavy_app, 0), "lhs_sweep": (lhs_sweep, 1)}


def make_sessions(seed, per_family, include_held_out=False):
    rng = random.Random(seed)
    out = []
    for name, fn in NORMAL.items():
        out += [(name, 0, fn(rng)) for _ in range(per_family)]
    for name, fn in ATTACK.items():
        out += [(name, 1, fn(rng)) for _ in range(per_family)]
    if include_held_out:
        for name, (fn, label) in HELD_OUT.items():
            out += [(name, label, fn(rng)) for _ in range(per_family)]
    return out


# ---------------------------------------------------------------------------
# Replay through the real feature extractor
# ---------------------------------------------------------------------------

def replay(session, indices=None):
    """
    Feature dict for each requested query index, computed from the window of
    earlier queries the middleware would see at that moment.
    """
    t0 = datetime(2026, 1, 1)
    xs = np.array([x for _, x in session])
    preds = _model.predict(xs)
    probs = _model.predict_proba(xs)
    rows = []
    prev_t = None
    for (t, x), p, pr in zip(session, preds, probs):
        gap = 0.0 if prev_t is None else (t - prev_t) * 1000
        rows.append(SimpleNamespace(
            timestamp=t0 + timedelta(seconds=t), features=x, inter_query_time_ms=gap,
            raw_prediction=int(p), raw_probabilities=[float(v) for v in pr]))
        prev_t = t

    out = []
    lo = 0
    for i in (range(len(session)) if indices is None else indices):
        now = rows[i].timestamp
        while rows[lo].timestamp < now - timedelta(seconds=WINDOW_SECONDS):
            lo += 1
        lo = min(lo, i)
        window = rows[max(lo, i - WINDOW_LIMIT):i]
        out.append((i, len(window), extract_behavioral_features(window, now=now)))
    return out
