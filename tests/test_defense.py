import random

from middleware import defense


def assert_well_formed(resp):
    probs = resp["probabilities"]
    assert abs(sum(probs) - 1.0) < 1e-9
    assert all(0.0 <= p <= 1.0 for p in probs)
    # Same 2-decimal format as the random forest's output
    assert all(round(p, 2) == p for p in probs)
    assert resp["prediction"] == probs.index(max(probs))


def test_get_tier_boundaries():
    assert defense.get_tier(0.0) == 0
    assert defense.get_tier(defense.T1) == 1
    assert defense.get_tier(defense.T2) == 2
    assert defense.get_tier(defense.T3) == 3
    assert defense.get_tier(defense.T4) == 4
    assert defense.get_tier(1.0) == 4


def test_perturbed_responses_are_well_formed():
    random.seed(0)
    for probs in ([1.0, 0.0, 0.0], [0.0, 0.54, 0.46], [0.01, 0.98, 0.01]):
        for sigma in (0.03, 0.08):
            for _ in range(200):
                assert_well_formed(defense.perturb_probabilities(probs, sigma))


def test_fake_responses_are_well_formed():
    random.seed(0)
    for _ in range(500):
        assert_well_formed(defense.fake_response())


def test_fake_tier_copies_real_model_name():
    real = {"prediction": 0, "probabilities": [1.0, 0.0, 0.0], "model": "iris-random-forest-v1"}
    resp, action = defense.apply_defense("1.2.3.4", 3, real)
    assert action == "FAKE"
    assert resp["model"] == "iris-random-forest-v1"
    assert set(resp) == set(real)


def test_tier4_refuses():
    assert defense.apply_defense("1.2.3.4", 4, None) == (None, "BLOCK")


def test_tier2_rate_limits_after_max_requests():
    real = {"prediction": 0, "probabilities": [1.0, 0.0, 0.0], "model": "m"}
    actions = [defense.apply_defense("1.2.3.4", 2, real)[1]
               for _ in range(defense.RATE_LIMIT_MAX_REQUESTS + 1)]
    assert actions[:-1] == ["PERTURB"] * defense.RATE_LIMIT_MAX_REQUESTS
    assert actions[-1] == "RATE_LIMIT"


def test_escalate_lets_noise_through_at_once():
    assert defense.escalate(1, []) == 1
    assert defense.escalate(2, []) == 2


def test_escalate_holds_harsh_tiers_until_streak():
    high = 0.95  # tier 4 at default thresholds
    assert defense.escalate(4, []) == 2
    assert defense.escalate(4, [high]) == 2
    assert defense.escalate(4, [high, high]) == 4
    # One quiet score in the streak keeps it at tier 2
    assert defense.escalate(4, [high, 0.0]) == 2
    # A tier-3 score in the streak caps the result at 3
    assert defense.escalate(4, [0.8, high]) == 3
