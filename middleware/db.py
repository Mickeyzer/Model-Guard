import os
from datetime import datetime, timezone, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, event, func
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.pool import StaticPool

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/modelguard.db")

# We use connect_args={"check_same_thread": False} for FastAPI with SQLite
engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)

@event.listens_for(engine, "connect")
def set_wal_mode(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()

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

def get_recent_queries(db: Session, client_ip: str, window_seconds: int = 60) -> list[QueryLog]:
    # Use naive UTC to match what SQLite stores (naive datetimes)
    cutoff = datetime.utcnow() - timedelta(seconds=window_seconds)
    return db.query(QueryLog).filter(
        QueryLog.client_ip == client_ip,
        QueryLog.timestamp >= cutoff
    ).order_by(QueryLog.timestamp.asc()).all()

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
        DefenseLog.action_taken == "BLOCKED"
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
    is_blocked = (latest_defense.action_taken == "BLOCKED") if latest_defense else False
    
    return {
        "query_count": query_count,
        "latest_threat_score": threat_score,
        "latest_tier": latest_tier,
        "is_blocked": is_blocked
    }
