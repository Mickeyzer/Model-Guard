import os
from datetime import datetime, timedelta, timezone

import altair as alt
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

st.set_page_config(page_title="ModelGuard", layout="wide", initial_sidebar_state="collapsed")

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///../data/modelguard.db")
engine = create_engine(DATABASE_URL)

# SQLite stores timestamps as naive UTC text like "2026-10-05 13:00:00.123456".
# Comparisons are string comparisons, so the cutoff must use the same format:
# isoformat()'s "T" separator sorts after the space and would drop most rows.
SQLITE_TS_FORMAT = "%Y-%m-%d %H:%M:%S"

WINDOWS = {"15 min": (timedelta(minutes=15), "30s"), "1 hour": (timedelta(hours=1), "1min"), "24 hours": (timedelta(hours=24), "30min")}

# What the client actually received, in escalation order
OUTCOMES = {
    "ALLOW": ("real answer", "#5b6b7f"),
    "PERTURB": ("noised answer", "#d29922"),
    "FAKE": ("fake answer", "#db6d28"),
    "RATE_LIMIT": ("refused", "#f85149"),
    "BLOCK": ("refused", "#f85149"),
}
OUTCOME_ORDER = ["real answer", "noised answer", "fake answer", "refused"]
OUTCOME_COLORS = ["#5b6b7f", "#d29922", "#db6d28", "#f85149"]

TIERS = [
    ("T0", "Pass through", "Real model output, untouched."),
    ("T1", "Light noise", "Probabilities jittered so harvested labels drift."),
    ("T2", "Noise + rate limit", "Heavier noise, 50 requests per minute."),
    ("T3", "Fake answers", "Plausible outputs in the real model's format."),
    ("T4", "IP ban", "Escalates 5 min, 1 h, 24 h. Holds after the score drops."),
]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
#MainMenu, footer, header {visibility: hidden;}
html, body, [class*="css"], .stMarkdown, .stDataFrame {font-family: 'IBM Plex Sans', sans-serif;}
.block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1280px;}
.mg-head {display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: .5rem;
          border-bottom: 1px solid #262d36; padding-bottom: .9rem; margin-bottom: 1.1rem;}
.mg-title {font-size: 1.35rem; font-weight: 600; letter-spacing: -.01em; color: #e6eaef;}
.mg-sub {color: #8a94a3; font-size: .88rem; margin-left: .6rem;}
.mg-live {font-family: 'IBM Plex Mono', monospace; font-size: .78rem; color: #8a94a3;}
.mg-dot {display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: #3fb950; margin-right: 6px;}
.mg-stats {display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); border: 1px solid #262d36; border-radius: 6px; margin-bottom: 1.4rem;}
.mg-stat {padding: .75rem .95rem; border-right: 1px solid #262d36;}
.mg-stat:last-child {border-right: none;}
.mg-stat .k {color: #8a94a3; font-size: .74rem; text-transform: uppercase; letter-spacing: .06em;}
.mg-stat .v {font-family: 'IBM Plex Mono', monospace; font-size: 1.45rem; color: #e6eaef; margin-top: .15rem;}
.mg-stat .v.warn {color: #db6d28;} .mg-stat .v.bad {color: #f85149;}
.mg-h {font-size: .8rem; text-transform: uppercase; letter-spacing: .07em; color: #8a94a3; margin: 0 0 .5rem 0;}
.mg-tier {display: grid; grid-template-columns: 2.2rem 1fr; gap: .1rem .6rem; padding: .45rem 0; border-bottom: 1px solid #1c222a; font-size: .86rem;}
.mg-tier .t {font-family: 'IBM Plex Mono', monospace; color: #8a94a3;}
.mg-tier .n {color: #e6eaef;} .mg-tier .d {grid-column: 2; color: #8a94a3; font-size: .8rem;}
.mg-empty {border: 1px dashed #303843; border-radius: 6px; padding: 1.1rem; color: #8a94a3; font-size: .88rem;}
.mg-empty code {color: #c9d1d9;}
@media (max-width: 900px) {.mg-stats {grid-template-columns: repeat(3, minmax(0, 1fr));}
  .mg-stat:nth-child(3) {border-right: none;} .mg-stat:nth-child(-n+3) {border-bottom: 1px solid #262d36;}}
</style>
""", unsafe_allow_html=True)


def mask_ip(ip: str) -> str:
    # The dashboard is public, so never show a visitor's full address
    if not isinstance(ip, str):
        return "unknown"
    if "." in ip:
        return ".".join(ip.split(".")[:3]) + ".x"
    return ":".join(ip.split(":")[:3]) + ":…"


def ago(ts: pd.Timestamp, now: datetime) -> str:
    s = int((now - ts).total_seconds())
    if s < 60:
        return f"{s}s ago"
    if s < 3600:
        return f"{s // 60}m ago"
    return f"{s // 3600}h ago"


def load(span: timedelta):
    now = datetime.now(timezone.utc)
    cutoff = (now - span).strftime(SQLITE_TS_FORMAT)
    with engine.connect() as conn:
        events = pd.read_sql(text(
            "SELECT timestamp, client_ip, threat_score, tier_applied, action_taken "
            "FROM defense_log WHERE timestamp >= :ts ORDER BY timestamp DESC"), conn, params={"ts": cutoff})
        try:
            bans = pd.read_sql(text("SELECT client_ip, strikes, banned_until FROM ban WHERE banned_until > :now"),
                               conn, params={"now": now.strftime(SQLITE_TS_FORMAT)})
        except OperationalError:
            # Older databases have no ban table yet
            bans = pd.DataFrame(columns=["client_ip", "strikes", "banned_until"])
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True, format="mixed")
    bans["banned_until"] = pd.to_datetime(bans["banned_until"], utc=True, format="mixed")
    events["outcome"] = events["action_taken"].map(lambda a: OUTCOMES.get(a, (a, ""))[0])
    return now, events, bans


head_l, head_r = st.columns([4, 1], vertical_alignment="center")
head_l.markdown(
    '<span class="mg-title">ModelGuard</span>'
    '<span class="mg-sub">Model-extraction defense in front of an Iris classifier</span>',
    unsafe_allow_html=True)
window = head_r.selectbox("Window", list(WINDOWS), index=1, label_visibility="collapsed")


@st.fragment(run_every=5)
def render(window: str):
    span, bucket = WINDOWS[window]
    now, ev, bans = load(span)

    st.markdown(
        f'<div class="mg-head"><span class="mg-live"><span class="mg-dot"></span>'
        f'live, refreshed {now.strftime("%H:%M:%S")} UTC</span>'
        f'<span class="mg-live">GET /health on :8000 · POST /predict</span></div>',
        unsafe_allow_html=True)

    n = len(ev)
    deceived = int(ev["action_taken"].isin(["PERTURB", "FAKE"]).sum())
    refused = int(ev["action_taken"].isin(["BLOCK", "RATE_LIMIT"]).sum())
    real_pct = f"{(ev['action_taken'] == 'ALLOW').mean() * 100:.0f}%" if n else "-"
    stats = [
        ("Requests", f"{n:,}", ""),
        ("Clients", f"{ev['client_ip'].nunique():,}", ""),
        ("Real answers", real_pct, ""),
        ("Noised or faked", f"{deceived:,}", "warn" if deceived else ""),
        ("Refused", f"{refused:,}", "bad" if refused else ""),
        ("Active bans", f"{len(bans):,}", "bad" if len(bans) else ""),
    ]
    st.markdown('<div class="mg-stats">' + "".join(
        f'<div class="mg-stat"><div class="k">{k}</div><div class="v {c}">{v}</div></div>' for k, v, c in stats)
        + "</div>", unsafe_allow_html=True)

    if ev.empty:
        st.markdown(
            f"""<div class="mg-empty">No traffic in the last {window}. Send a request to
            <code>POST /predict</code> on port 8000 with <code>{{"features": [5.1, 3.5, 1.4, 0.2]}}</code>,
            or run <code>demo/simulate_attack.py</code> with <code>MODELGUARD_URL</code> pointed at this host
            to watch a grid sweep escalate to a ban.</div>""", unsafe_allow_html=True)
        return

    main, side = st.columns([2.3, 1], gap="large")

    with main:
        st.markdown('<p class="mg-h">What clients received</p>', unsafe_allow_html=True)
        # Naive UTC so the browser doesn't shift the axis into local time.
        # Bars get explicit start/end on both axes: Vega-Lite won't stack bars
        # that have x2, so the stack is computed here in escalation order.
        ts = (ev.assign(t=ev["timestamp"].dt.floor(bucket).dt.tz_localize(None))
              .groupby(["t", "outcome"]).size().reset_index(name="requests"))
        ts["rank"] = ts["outcome"].map(OUTCOME_ORDER.index)
        ts = ts.sort_values(["t", "rank"])
        ts["top"] = ts.groupby("t")["requests"].cumsum()
        ts["base"] = ts["top"] - ts["requests"]
        ts["t_end"] = ts["t"] + pd.Timedelta(bucket) * 0.85
        chart = alt.Chart(ts).mark_bar().encode(
            x=alt.X("t:T", title=None, axis=alt.Axis(format="%H:%M", grid=False, labelColor="#8a94a3", domainColor="#262d36", tickColor="#262d36")),
            x2="t_end:T",
            y=alt.Y("top:Q", title=None, axis=alt.Axis(gridColor="#1c222a", labelColor="#8a94a3", domain=False, ticks=False, tickMinStep=1)),
            y2="base:Q",
            color=alt.Color("outcome:N", scale=alt.Scale(domain=OUTCOME_ORDER, range=OUTCOME_COLORS),
                            legend=alt.Legend(orient="top", title=None, labelColor="#c9d1d9", symbolType="square")),
            tooltip=[alt.Tooltip("t:T", title="from (UTC)", format="%H:%M:%S"), "outcome:N", "requests:Q"],
        ).properties(height=260).configure_view(strokeWidth=0).configure(background="transparent")
        st.altair_chart(chart, width="stretch")

        st.markdown('<p class="mg-h">Clients</p>', unsafe_allow_html=True)
        clients = (ev.sort_values("timestamp")
                   .groupby("client_ip")
                   .agg(requests=("action_taken", "size"), peak=("threat_score", "max"),
                        last_tier=("tier_applied", "last"), last_outcome=("outcome", "last"), last_seen=("timestamp", "max"))
                   .reset_index())
        banned = dict(zip(bans["client_ip"], bans["banned_until"]))
        clients["ban"] = clients["client_ip"].map(
            lambda ip: f"{int((banned[ip] - now).total_seconds() // 60)} min left" if ip in banned else "")
        clients["client"] = clients["client_ip"].map(mask_ip)
        clients["last_seen"] = clients["last_seen"].map(lambda t: ago(t, now))
        clients["last_tier"] = clients["last_tier"].map(lambda t: f"T{int(t)}")
        st.dataframe(
            clients.sort_values("peak", ascending=False)[["client", "requests", "peak", "last_tier", "last_outcome", "ban", "last_seen"]],
            hide_index=True, width="stretch",
            column_config={
                "client": "Client",
                "requests": st.column_config.NumberColumn("Requests", format="%d"),
                "peak": st.column_config.ProgressColumn("Peak threat score", min_value=0.0, max_value=1.0, format="%.2f", color="#8a94a3"),
                "last_tier": "Tier",
                "last_outcome": "Last response",
                "ban": "Ban",
                "last_seen": "Seen",
            })

    with side:
        st.markdown('<p class="mg-h">Defense tiers</p>', unsafe_allow_html=True)
        st.markdown("".join(
            f'<div class="mg-tier"><span class="t">{t}</span><span class="n">{name}</span><span class="d">{desc}</span></div>'
            for t, name, desc in TIERS), unsafe_allow_html=True)

    st.markdown('<p class="mg-h" style="margin-top:1.2rem">Event log</p>', unsafe_allow_html=True)
    log = ev.head(50).copy()
    log["time"] = log["timestamp"].dt.strftime("%H:%M:%S")
    log["client"] = log["client_ip"].map(mask_ip)
    log["tier"] = log["tier_applied"].map(lambda t: f"T{int(t)}")
    st.dataframe(
        log[["time", "client", "threat_score", "tier", "action_taken", "outcome"]],
        hide_index=True, width="stretch", height=320,
        column_config={
            "time": "Time (UTC)",
            "client": "Client",
            "threat_score": st.column_config.NumberColumn("Score", format="%.3f"),
            "tier": "Tier",
            "action_taken": "Action",
            "outcome": "Client got",
        })


render(window)
