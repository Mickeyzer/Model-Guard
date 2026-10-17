import os
from datetime import datetime, timezone, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, event, func
from sqlalchemy.orm import declarative_base, sessionmaker, Session

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/modelguard.db")

# check_same_thread=False lets FastAPI's threadpool use SQLite. The default
# pool gives each request its own connection; a shared StaticPool connection
# would let one request's rollback discard another request's writes.
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

@event.listens_for(engine, "connect")
def set_wal_mode(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    # Wait for a concurrent writer instead of failing with "database is locked"
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()

# Escalating ban lengths: 1st strike, 2nd strike, 3rd+ strike
BAN_DURATIONS_SECONDS = [
    int(s) for s in os.getenv("BAN_DURATIONS_SECONDS", "300,3600,86400").split(",")
]
# Strikes reset once a client has stayed clean this long after its last ban
BAN_STRIKE_RESET_SECONDS = int(os.getenv("BAN_STRIKE_RESET_SECONDS", 86400))

def utcnow() -> datetime:
    """Naive UTC, matching how SQLite stores the DateTime columns."""
    return datetime.now(timezone.utc).replace(tzinfo=None)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class QueryLog(Base):
    __tablename__ = "query_log"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, index=True, default=lambda: datetime.now(timezone.utc))
    client_ip = Column(String(45), index=True)
    session_id = Column(String(64))
    features = Column(JSON)
    raw_prediction = Column(Integer, nullable=True)
    raw_probabilities = Column(JSON, nullable=True)
    returned_prediction = Column(Integer)
    returned_probabilities = Column(JSON)
    inter_query_time_ms = Column(Float)
    threat_score = Column(Float)
    tier_applied = Column(Integer)

class DefenseLog(Base):
    __tablename__ = "defense_log"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, index=True, default=lambda: datetime.now(timezone.utc))
    client_ip = Column(String(45), index=True)
    session_id = Column(String(64))
    threat_score = Column(Float)
    tier_applied = Column(Integer)
    action_taken = Column(String(32))
    query_count_in_window = Column(Integer)

class Ban(Base):
    __tablename__ = "ban"

    client_ip = Column(String(45), primary_key=True)
    strikes = Column(Integer, default=0)
    banned_until = Column(DateTime)

def init_db() -> None:
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def log_query(db: Session, client_ip: str, session_id: str, features: list,
              raw_prediction: int, raw_probabilities: list,
              returned_prediction: int, returned_probabilities: list,
              inter_query_time_ms: float, threat_score: float,
              tier_applied: int) -> QueryLog:
    db_query = QueryLog(
        client_ip=client_ip,
        session_id=session_id,
        features=features,
        raw_prediction=raw_prediction,
        raw_probabilities=raw_probabilities,
        returned_prediction=returned_prediction,
        returned_probabilities=returned_probabilities,
        inter_query_time_ms=inter_query_time_ms,
        threat_score=threat_score,
        tier_applied=tier_applied
    )
    db.add(db_query)
    db.commit()
    db.refresh(db_query)
    return db_query

def log_defense_event(db: Session, client_ip: str, session_id: str,
                      threat_score: float, tier_applied: int,
                      action_taken: str, query_count_in_window: int) -> DefenseLog:
    db_defense = DefenseLog(
        client_ip=client_ip,
        session_id=session_id,
        threat_score=threat_score,
        tier_applied=tier_applied,
        action_taken=action_taken,
        query_count_in_window=query_count_in_window
    )
    db.add(db_defense)
    db.commit()
    db.refresh(db_defense)
    return db_defense

def log_request(db: Session, client_ip: str, session_id: str, features: list,
                raw_prediction: int | None, raw_probabilities: list | None,
                returned_prediction: int | None, returned_probabilities: list | None,
                inter_query_time_ms: float, threat_score: float, tier_applied: int,
                action_taken: str, query_count_in_window: int) -> None:
    """Write the query and its defense event in one transaction, before the
    response goes out, so the next request's window already includes it."""
    now = utcnow()
    db.add(QueryLog(
        timestamp=now,
        client_ip=client_ip,
        session_id=session_id,
        features=features,
        raw_prediction=raw_prediction,
        raw_probabilities=raw_probabilities,
        returned_prediction=returned_prediction,
        returned_probabilities=returned_probabilities,
        inter_query_time_ms=inter_query_time_ms,
        threat_score=threat_score,
        tier_applied=tier_applied,
    ))
    db.add(DefenseLog(
        timestamp=now,
        client_ip=client_ip,
        session_id=session_id,
        threat_score=threat_score,
        tier_applied=tier_applied,
        action_taken=action_taken,
        query_count_in_window=query_count_in_window,
    ))
    db.commit()

def get_active_ban(db: Session, client_ip: str) -> Ban | None:
    ban = db.get(Ban, client_ip)
    if ban and ban.banned_until and ban.banned_until > utcnow():
        return ban
    return None

def register_ban(db: Session, client_ip: str) -> Ban:
    """Add a strike and ban the client; each strike bans for longer."""
    now = utcnow()
    ban = db.get(Ban, client_ip)
    if ban is None:
        ban = Ban(client_ip=client_ip, strikes=0)
        db.add(ban)
    elif ban.banned_until and now - ban.banned_until > timedelta(seconds=BAN_STRIKE_RESET_SECONDS):
        ban.strikes = 0
    ban.strikes = (ban.strikes or 0) + 1
    duration = BAN_DURATIONS_SECONDS[min(ban.strikes, len(BAN_DURATIONS_SECONDS)) - 1]
    ban.banned_until = now + timedelta(seconds=duration)
    db.commit()
    return ban

def get_recent_queries(db: Session, client_ip: str, window_seconds: int = 300,
                       limit: int = 100) -> list[QueryLog]:
    # The newest `limit` queries in the window, oldest first. Five minutes lets
    # a slow sweep (a few queries a minute) build up enough history to score;
    # the cap keeps the O(n^2) repetition check cheap for fast clients.
    cutoff = utcnow() - timedelta(seconds=window_seconds)
    rows = db.query(QueryLog).filter(
        QueryLog.client_ip == client_ip,
        QueryLog.timestamp >= cutoff
    ).order_by(QueryLog.timestamp.desc()).limit(limit).all()
    return rows[::-1]

def get_last_query_timestamp(db: Session, client_ip: str) -> datetime | None:
    result = db.query(QueryLog.timestamp).filter(
        QueryLog.client_ip == client_ip
    ).order_by(QueryLog.timestamp.desc()).first()
    # Return as naive UTC to match what SQLite stores
    return result[0] if result else None

def get_query_count_today(db: Session) -> int:
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return db.query(QueryLog).filter(QueryLog.timestamp >= today_start).count()

def get_unique_ip_count(db: Session, window_hours: int = 24) -> int:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    return db.query(func.count(func.distinct(QueryLog.client_ip))).filter(
        QueryLog.timestamp >= cutoff
    ).scalar() or 0

def get_attack_count(db: Session, window_hours: int = 24) -> int:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    return db.query(DefenseLog).filter(
        DefenseLog.timestamp >= cutoff,
        DefenseLog.tier_applied >= 3
    ).count()

def get_blocked_ip_count(db: Session, window_hours: int = 24) -> int:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    return db.query(func.count(func.distinct(DefenseLog.client_ip))).filter(
        DefenseLog.timestamp >= cutoff,
        DefenseLog.action_taken == "BLOCK"
    ).scalar() or 0

def get_recent_defense_events(db: Session, limit: int = 20) -> list[DefenseLog]:
    return db.query(DefenseLog).order_by(DefenseLog.timestamp.desc()).limit(limit).all()

def get_query_volume_by_minute(db: Session, last_n_minutes: int = 30) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=last_n_minutes)
    
    # SQLite-specific strftime for grouping by minute
    results = db.query(
        func.strftime('%Y-%m-%d %H:%M:00', QueryLog.timestamp).label("minute"),
        func.count(QueryLog.id).label("count"),
        func.max(QueryLog.tier_applied).label("max_tier")
    ).filter(
        QueryLog.timestamp >= cutoff
    ).group_by("minute").order_by("minute").all()
    
    return [
        {
            "minute": datetime.strptime(r[0], '%Y-%m-%d %H:%M:00').replace(tzinfo=timezone.utc),
            "count": r[1],
            "max_tier": r[2]
        }
        for r in results if r[0] is not None
    ]

def get_threat_score_distribution(db: Session, bins: int = 10) -> list[dict]:
    scores = db.query(QueryLog.threat_score).all()
    scores = [s[0] for s in scores if s[0] is not None]
    
    bin_edges = [i / bins for i in range(bins + 1)]
    distribution = []
    
    for i in range(bins):
        start = bin_edges[i]
        end = bin_edges[i+1]
        
        if i == bins - 1:
            count = sum(1 for s in scores if start <= s <= end)
        else:
            count = sum(1 for s in scores if start <= s < end)
            
        distribution.append({
            "bin_start": start,
            "bin_end": end,
            "count": count
        })
    return distribution

def get_tier_distribution(db: Session) -> dict[int, int]:
    results = db.query(
        DefenseLog.tier_applied,
        func.count(DefenseLog.id)
    ).group_by(DefenseLog.tier_applied).all()
    
    # Ensure tiers 0-4 exist in dict even if 0 count
    dist = {i: 0 for i in range(5)}
    for tier, count in results:
        if tier is not None:
            dist[tier] = count
    return dist

def get_ip_stats(db: Session, client_ip: str) -> dict:
    query_count = db.query(QueryLog).filter(QueryLog.client_ip == client_ip).count()
    
    latest_query = db.query(QueryLog).filter(
        QueryLog.client_ip == client_ip
    ).order_by(QueryLog.timestamp.desc()).first()
    
    latest_defense = db.query(DefenseLog).filter(
        DefenseLog.client_ip == client_ip
    ).order_by(DefenseLog.timestamp.desc()).first()
    
    threat_score = latest_query.threat_score if latest_query else 0.0
    latest_tier = latest_defense.tier_applied if latest_defense else 0
    is_blocked = get_active_ban(db, client_ip) is not None
    
    return {
        "query_count": query_count,
        "latest_threat_score": threat_score,
        "latest_tier": latest_tier,
        "is_blocked": is_blocked
    }
