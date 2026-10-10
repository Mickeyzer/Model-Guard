"""
ModelGuard Attack Simulation -- LinkedIn Demo Version
=====================================================
Simulates a realistic, gradual attack escalation across all 5 tiers so that
every defense response (ALLOW -> PERTURB -> FAKE -> BLOCK) is clearly visible.

Phase 0 -- Baseline   : Normal user, random Iris-like queries, slow pace
Phase 1 -- Recon      : Slightly faster, values drifting toward OOD
Phase 2 -- Probing    : Moderate speed, boundary-adjacent feature values
Phase 3 -- Sweep Prep : Higher speed, clearly OOD, systematic
Phase 4 -- Full Attack: Rapid grid sweep, maximum aggression
"""
import os
import sys
import time
import random
import httpx
import numpy as np

# Force UTF-8 output so the script works on Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Point at a remote deployment with e.g. MODELGUARD_URL=http://<ec2-ip>:8000/predict
URL = os.getenv("MODELGUARD_URL", "http://localhost:8000/predict")

# Scores and tiers are only visible when the middleware runs with
# EXPOSE_THREAT_HEADERS=1; otherwise only allowed vs blocked can be told apart.

TIER_LABELS = {0: "ALLOW", 1: "PERTURB", 2: "PERTURB", 3: "FAKE", 4: "BLOCK"}


def send_query(client, features, query_type, idx):
    try:
        response = client.post(URL, json={"features": features})

        threat_score_str = response.headers.get("X-Threat-Score", None)
        tier_str         = response.headers.get("X-Defense-Tier", None)

        threat_score = float(threat_score_str) if threat_score_str else None
        tier         = int(tier_str)           if tier_str         else None

        action = "UNKNOWN"
        if response.status_code == 200:
            # Fakes look exactly like real answers, so only the tier header
            # tells them apart
            action = TIER_LABELS.get(tier, "ANSWERED") if tier is not None else "ANSWERED"
        elif response.status_code == 429:
            action = "BLOCK"
            if response.headers.get("Retry-After"):
                action += f" (banned {response.headers['Retry-After']}s)"

        return {
            "index":        idx,
            "type":         query_type,
            "threat_score": threat_score,
            "tier":         tier,
            "action":       action,
        }
    except Exception as e:
        print(f"  [error] query {idx}: {e}")
        return None


def print_row(r):
    # Icon based on actual action taken (not tier), so rate-limit blocks show [X]
    icon_map = {"ALLOW": "[ ]", "PERTURB": "[~]", "FAKE": "[?]", "BLOCK": "[X]"}
    icon = icon_map.get(r["action"].split(" ")[0], "   ")
    score = f"{r['threat_score']:>6.4f}" if r["threat_score"] is not None else "hidden"
    tier = f"T{r['tier']}" if r["tier"] is not None else "T?"
    print(
        f"  {r['index']:<4} "
        f"{r['type']:<11} "
        f"{score}       "
        f"{tier}     "
        f"{icon} {r['action']}"
    )


def phase_banner(num, name, desc):
    print(f"\n{'='*62}")
    print(f"  Phase {num} -- {name}")
    print(f"  {desc}")
    print(f"{'='*62}")
    print(f"  {'#':<4} {'Type':<11} {'Threat Score':<14} {'Tier':<6} Action")
    print(f"  {'-'*56}")


def countdown(seconds, label="Waiting for behavioral window to age"):
    for i in range(seconds, 0, -1):
        print(f"  [{label}... {i}s]", end="\r", flush=True)
        time.sleep(1)
    print(" " * 60, end="\r")


def main():
    print(f"\n{'='*62}")
    print(f"  ModelGuard -- Live Attack Simulation (LinkedIn Demo)")
    print(f"  Target: {URL}")
    print(f"  (start the stack with EXPOSE_THREAT_HEADERS=1 to see scores and tiers)")
    print(f"{'='*62}\n")

    all_results = []
    idx = 1

    with httpx.Client(timeout=10.0) as client:

        # ── Phase 0: Normal baseline ──────────────────────────────────────────────
        phase_banner(0, "Baseline",
                     "Normal user -- 2 clean queries, human pace -- T0 expected")
        means = [5.8, 3.0, 3.7, 1.2]
        stds  = [0.4, 0.2, 0.8, 0.3]
        for _ in range(2):  # 2 queries: feature extractor zeroes entropy for n<3
            feats = [random.gauss(m, s) for m, s in zip(means, stds)]
            r = send_query(client, feats, "NORMAL", idx)
            if r:
                all_results.append(r)
                print_row(r)
            idx += 1
            time.sleep(random.uniform(15.0, 20.0))  # Human-pace: 15-20s between queries


        # Short pause so Phase 0 queries begin aging in the 60s window
        countdown(8, "Phase transition pause")

        # ── Phase 1: Reconnaissance ───────────────────────────────────
        phase_banner(1, "Recon",
                     "Moderate pace, values drifting slightly OOD -- Tier 0/1 expected")
        for _ in range(8):
            # Slightly outside distribution: sepal length 6.5-7.5, wider variance
            feats = [
                random.gauss(7.0, 0.4),
                random.gauss(2.5, 0.3),
                random.gauss(5.2, 0.4),
                random.gauss(1.9, 0.2),
            ]
            r = send_query(client, feats, "RECON", idx)
            if r:
                all_results.append(r)
                print_row(r)
            idx += 1
            time.sleep(random.uniform(0.9, 1.4))

        # Longer pause -- allows Phase 0 queries to age out of 60s window
        countdown(15, "Phase transition pause")

        # ── Phase 2: Boundary probing ─────────────────────────────────
        phase_banner(2, "Probe",
                     "Faster pace, boundary-adjacent values -- Tier 1/2 expected")
        for _ in range(10):
            # Values near decision boundaries, moderate variance
            feats = [
                random.uniform(3.5, 5.0),
                random.uniform(2.0, 2.8),
                random.uniform(3.8, 5.5),
                random.uniform(1.0, 1.8),
            ]
            r = send_query(client, feats, "PROBE", idx)
            if r:
                all_results.append(r)
                print_row(r)
            idx += 1
            time.sleep(random.uniform(0.45, 0.75))

        countdown(8, "Phase transition pause")

        # ── Phase 3: High-speed sweep prep ───────────────────────────
        phase_banner(3, "Sweep Prep",
                     "High velocity + OOD features -- Tier 2/3 expected")
        f1_vals = np.linspace(2.0, 6.5, 5)
        f2_vals = np.linspace(1.5, 5.5, 3)
        grid = [(f1, f2, 4.0, 1.5) for f1 in f1_vals for f2 in f2_vals]
        for feats in grid[:6]:
            r = send_query(client, list(feats), "SWEEP", idx)
            if r:
                all_results.append(r)
                print_row(r)
            idx += 1
            time.sleep(random.uniform(0.18, 0.30))

        countdown(3, "Launching full attack")

        # ── Phase 4: Full grid sweep ──────────────────────────────────
        phase_banner(4, "Full Attack",
                     "Maximum aggression -- grid sweep -- Tier 4 expected")
        f1_vals = np.linspace(0.5, 7.5, 8)
        f2_vals = np.linspace(0.5, 7.5, 8)
        full_grid = [(f1, f2, 2.0, 1.0) for f1 in f1_vals for f2 in f2_vals]
        for feats in full_grid[:15]:
            r = send_query(client, list(feats), "ATTACK", idx)
            if r:
                all_results.append(r)
                print_row(r)
            idx += 1
            time.sleep(random.uniform(0.05, 0.12))

    # ── Final Summary ─────────────────────────────────────────────────
    print(f"\n\n{'='*62}")
    print(f"  SUMMARY")
    print(f"{'='*62}")

    by_type = {}
    for r in all_results:
        by_type.setdefault(r["type"], []).append(r)

    for qtype, results in by_type.items():
        scored = [r for r in results if r["tier"] is not None]
        blocked = sum(1 for r in results if r["action"].startswith("BLOCK"))
        if not scored:
            print(f"  {qtype:<12} -> {blocked}/{len(results)} blocked  (scores hidden)")
            continue
        avg  = sum(r["threat_score"] for r in scored) / len(scored)
        maxt = max(r["tier"] for r in scored)
        action = TIER_LABELS.get(maxt, "?")
        print(f"  {qtype:<12} -> Avg Threat: {avg:.4f}  |  Max Tier: T{maxt} ({action})")

    print(f"\n  Tier key: T0=ALLOW  T1/T2=PERTURB  T3=FAKE  T4=BLOCK")
    print(f"  Legend:   [ ]=Safe  [~]=Perturbed  [?]=Fake  [X]=Blocked\n")


if __name__ == "__main__":
    main()
