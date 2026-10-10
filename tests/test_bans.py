from datetime import timedelta

from middleware.db import Ban, get_active_ban, register_ban, utcnow


def ban_seconds(ban):
    return (ban.banned_until - utcnow()).total_seconds()


def test_bans_escalate_with_each_strike(db):
    expected = [300, 3600, 86400, 86400]  # capped at the last duration
    for strikes, seconds in enumerate(expected, start=1):
        ban = register_ban(db, "1.2.3.4")
        assert ban.strikes == strikes
        assert seconds - 5 <= ban_seconds(ban) <= seconds


def test_strikes_reset_after_staying_clean(db):
    db.add(Ban(client_ip="1.2.3.4", strikes=2, banned_until=utcnow() - timedelta(days=2)))
    db.commit()
    ban = register_ban(db, "1.2.3.4")
    assert ban.strikes == 1
    assert 295 <= ban_seconds(ban) <= 300


def test_active_ban_lookup(db):
    assert get_active_ban(db, "1.2.3.4") is None
    register_ban(db, "1.2.3.4")
    assert get_active_ban(db, "1.2.3.4") is not None
    db.get(Ban, "1.2.3.4").banned_until = utcnow() - timedelta(seconds=1)
    db.commit()
    assert get_active_ban(db, "1.2.3.4") is None
