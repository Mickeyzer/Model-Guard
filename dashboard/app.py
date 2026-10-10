import streamlit as st
import pandas as pd
import altair as alt
import time
import os
from datetime import datetime, timezone, timedelta
from sqlalchemy import create_engine, text

# ---- Config & Styling ----
st.set_page_config(
    page_title="ModelGuard — Live Threat Monitor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a minimal, professional look
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
    }
    /* Style the metrics */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# ---- DB Connection ----
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///../data/modelguard.db")
engine = create_engine(DATABASE_URL)

# ---- Auto-Refresh ----
# This ensures the dashboard stays live
st.empty() # Placeholder for visual top spacing

# ---- Data Fetching ----
now = datetime.now(timezone.utc)
twenty_four_hours_ago = now - timedelta(hours=24)
thirty_mins_ago = now - timedelta(minutes=30)

# SQLite stores timestamps as naive UTC text like "2026-10-05 13:00:00.123456".
# Comparisons are string comparisons, so the cutoff must use the same format:
# isoformat()'s "T" separator sorts after the space and would drop most rows.
SQLITE_TS_FORMAT = "%Y-%m-%d %H:%M:%S"
ts_24h = (now - timedelta(hours=24)).strftime(SQLITE_TS_FORMAT)

with engine.connect() as conn:
    # 1. Top Metrics (Last 24h)
    q_total = "SELECT COUNT(*) FROM query_log WHERE timestamp >= :ts"
    q_ips = "SELECT COUNT(DISTINCT client_ip) FROM query_log WHERE timestamp >= :ts"
    # Tier 3+ (fake or block); tiers 1-2 are precautionary noise, not attacks
    q_attacks = "SELECT COUNT(*) FROM defense_log WHERE timestamp >= :ts AND tier_applied >= 3"
    q_blocks = "SELECT COUNT(DISTINCT client_ip) FROM defense_log WHERE timestamp >= :ts AND action_taken = 'BLOCK'"
    
    total_queries = conn.execute(text(q_total), {"ts": ts_24h}).scalar() or 0
    active_ips = conn.execute(text(q_ips), {"ts": ts_24h}).scalar() or 0
    attacks_detected = conn.execute(text(q_attacks), {"ts": ts_24h}).scalar() or 0
    ips_blocked = conn.execute(text(q_blocks), {"ts": ts_24h}).scalar() or 0



    # 3. Tier Distribution & Threat Score Histogram (Last 24h)
    # Fetch all tiers and threat scores for the histograms
    q_dist = "SELECT threat_score, tier_applied FROM defense_log WHERE timestamp >= :ts"
    df_dist = pd.read_sql(text(q_dist), conn, params={"ts": ts_24h})
    
    # 4. Recent Defense Events
    q_recent = """
        SELECT timestamp, client_ip, threat_score, tier_applied, action_taken
        FROM defense_log
        ORDER BY timestamp DESC
        LIMIT 20
    """
    df_recent = pd.read_sql(text(q_recent), conn)

# ---- Render UI ----
st.title("ModelGuard — Live Threat Monitor")
st.caption(f"Last updated: {now.strftime('%H:%M:%S')} UTC")

# Row 1: Metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Queries (24h)", f"{total_queries:,}")
c2.metric("Active IPs (24h)", f"{active_ips:,}")
c3.metric("Attacks Detected (24h)", f"{attacks_detected:,}")
c4.metric("IPs Blocked (24h)", f"{ips_blocked:,}")

st.divider()

# Color mapping for tiers
color_scale = alt.Scale(
    domain=[0, 1, 2, 3, 4],
    range=['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#8b0000']
)



st.divider()

# Row 3: Two Columns
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Tier Distribution (24h)")
    if not df_dist.empty:
        tier_counts = df_dist['tier_applied'].value_counts().reset_index()
        tier_counts.columns = ['Tier', 'Count']
        
        bar_chart = alt.Chart(tier_counts).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X('Tier:O', title="Tier Level", axis=alt.Axis(labelAngle=0, grid=False)),
            y=alt.Y('Count:Q', title="Number of Events", axis=alt.Axis(grid=True, gridOpacity=0.2)),
            color=alt.Color('Tier:O', scale=color_scale, legend=None),
            tooltip=['Tier', 'Count']
        ).properties(height=300)
        st.altair_chart(bar_chart, use_container_width=True)
    else:
        st.info("No data available.")

with col_right:
    st.subheader("Threat Score Distribution (24h)")
    if not df_dist.empty:
        hist_chart = alt.Chart(df_dist).mark_bar(color="#3498db").encode(
            x=alt.X('threat_score:Q', bin=alt.Bin(maxbins=10, extent=[0.0, 1.0]), title="Threat Score", axis=alt.Axis(grid=False)),
            y=alt.Y('count():Q', title="Count", axis=alt.Axis(grid=True, gridOpacity=0.2)),
            tooltip=['count()']
        ).properties(height=300)
        st.altair_chart(hist_chart, use_container_width=True)
    else:
        st.info("No data available.")

st.divider()

# Row 4: Recent Events
st.subheader("Recent Defense Events")
if not df_recent.empty:
    # Rename columns for display
    df_display = df_recent.rename(columns={
        "timestamp": "Timestamp",
        "client_ip": "IP",
        "threat_score": "Threat Score",
        "tier_applied": "Tier",
        "action_taken": "Action"
    })
    
    # Format float
    df_display["Threat Score"] = df_display["Threat Score"].apply(lambda x: f"{x:.4f}")
    
    # Styling function
    def highlight_rows(row):
        action = row["Action"]
        if action in ("BLOCK", "RATE_LIMIT"):
            return ['background-color: rgba(231, 76, 60, 0.2)'] * len(row)
        elif action == "FAKE":
            return ['background-color: rgba(243, 156, 18, 0.2)'] * len(row)
        elif action == "PERTURB":
            return ['background-color: rgba(241, 196, 15, 0.2)'] * len(row)
        return [''] * len(row)

    styled_df = df_display.style.apply(highlight_rows, axis=1)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
else:
    st.info("No recent defense events.")

# Auto-refresh logic at the very end
time.sleep(5)
st.rerun()
