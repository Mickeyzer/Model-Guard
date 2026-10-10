import math

def extract_behavioral_features(recent_queries: list) -> dict:
    # 0. Handle empty list
    if not recent_queries:
        return {
            "query_count_1min": 0.0,
            "mean_inter_query_time_ms": 0.0,
            "feature_space_entropy": 0.0,
            "prediction_entropy": 0.0,
            "boundary_proximity_score": 0.0,
            "input_variance": 0.0,
            "repetition_ratio": 0.0
        }
    
    n = len(recent_queries)
    
    # 1. query_count_1min
    query_count_1min = float(n)
    
    # 2. mean_inter_query_time_ms
    if n < 2:
        mean_inter_query_time_ms = 0.0
    else:
        # The first query's gap points at a query outside the window (or is
        # 0.0 for a new client), so only the n-1 gaps inside the window count.
        gaps = [q.inter_query_time_ms or 0.0 for q in recent_queries[1:]]
        mean_inter_query_time_ms = float(sum(gaps)) / len(gaps)
        
    # Helper for extracting feature matrix
    # Assuming q.features is a list of 4 floats
    try:
        X = [q.features for q in recent_queries if q.features and len(q.features) == 4]
    except Exception:
        X = []
        
    n_valid_X = len(X)
    
    # 3. feature_space_entropy
    feature_space_entropy = 0.0
    if n_valid_X > 0:
        entropies = []
        for dim in range(4):
            bins = [0] * 10
            for row in X:
                val = row[dim]
                # Bin over [0, 10]
                bin_idx = min(max(int(val), 0), 9)
                bins[bin_idx] += 1
            
            dim_entropy = 0.0
            for count in bins:
                p = count / n_valid_X
                if p > 0:
                    dim_entropy -= p * math.log2(p)
            entropies.append(dim_entropy)
        feature_space_entropy = sum(entropies) / 4.0

    # Features 4 and 5 use the protected model's real outputs, not what was
    # returned. Returned outputs may be perturbed or fake, and scoring those
    # would let ModelGuard's own noise push a client into higher tiers.

    # 4. prediction_entropy
    prediction_entropy = 0.0
    preds = [q.raw_prediction for q in recent_queries if q.raw_prediction is not None]
    if preds:
        counts = {0: 0, 1: 0, 2: 0}
        for p in preds:
            if p in counts:
                counts[p] += 1
            else:
                counts[p] = 1 # Just in case it's not 0,1,2
                
        n_preds = len(preds)
        for count in counts.values():
            p = count / n_preds
            if p > 0:
                prediction_entropy -= p * math.log2(p)

    # 5. boundary_proximity_score
    boundary_proximity_score = 0.0
    probs = [q.raw_probabilities for q in recent_queries if q.raw_probabilities]
    if probs:
        boundary_count = sum(1 for p in probs if max(p) < 0.70)
        boundary_proximity_score = float(boundary_count) / len(probs)

    # 6. input_variance
    input_variance = 0.0
    if n_valid_X > 1:
        variances = []
        for dim in range(4):
            col = [row[dim] for row in X]
            mean_val = sum(col) / n_valid_X
            var = sum((x - mean_val)**2 for x in col) / (n_valid_X - 1)
            variances.append(var)
        input_variance = sum(variances) / 4.0

    # 7. repetition_ratio
    repetition_ratio = 0.0
    if n_valid_X >= 2:
        reps = 0
        for i in range(1, n_valid_X):
            curr = X[i]
            # Check if any prior query is within L2 < 0.1
            has_rep = False
            for j in range(i):
                prev = X[j]
                l2_sq = sum((curr[k] - prev[k])**2 for k in range(4))
                if math.sqrt(l2_sq) < 0.1:
                    has_rep = True
                    break
            if has_rep:
                reps += 1
        repetition_ratio = float(reps) / n_valid_X

    # Final logic: best effort if < 3 queries
    if n < 3:
        feature_space_entropy = 0.0
        prediction_entropy = 0.0
        input_variance = 0.0
        repetition_ratio = 0.0

    return {
        "query_count_1min": query_count_1min,
        "mean_inter_query_time_ms": mean_inter_query_time_ms,
        "feature_space_entropy": feature_space_entropy,
        "prediction_entropy": prediction_entropy,
        "boundary_proximity_score": boundary_proximity_score,
        "input_variance": input_variance,
        "repetition_ratio": repetition_ratio
    }
