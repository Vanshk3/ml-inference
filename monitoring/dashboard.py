"""
Real-time monitoring dashboard for the Fleet AI inference server.
Run alongside the server: streamlit run monitoring/dashboard.py
"""

import time
import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Fleet AI — Inference Monitor", page_icon="📡", layout="wide")

st.markdown("""
<style>
.metric-card{background:#f8f8f6;border:0.5px solid #e0ddd6;border-radius:10px;padding:.9rem 1.1rem;text-align:center}
.metric-label{font-size:12px;color:#888;margin-bottom:4px}
.metric-value{font-size:24px;font-weight:600;color:#1a1a1a}
.ok{color:#1D9E75;font-weight:600}.bad{color:#D85A30;font-weight:600}
</style>
""", unsafe_allow_html=True)

st.markdown("## Fleet AI — Inference Server Monitor")
st.markdown("Live latency, throughput, and prediction distribution.")

auto_refresh = st.sidebar.checkbox("Auto-refresh every 3s", value=True)
api_url = st.sidebar.text_input("API base URL", value=API_BASE)
st.sidebar.markdown("---")
if st.sidebar.button("Ping /health"):
    try:
        r = requests.get(f"{api_url}/health", timeout=3)
        st.sidebar.json(r.json())
    except Exception as e:
        st.sidebar.error(str(e))

def fetch(ep):
    try:
        r = requests.get(f"{api_url}{ep}", timeout=3)
        return r.json() if r.status_code == 200 else None
    except:
        return None

health  = fetch("/health")
metrics = fetch("/metrics")
logs_r  = fetch("/metrics/logs?n=200")
logs    = logs_r.get("logs", []) if logs_r else []

if health:
    st_html = '<span class="ok">ONLINE</span>'
    uptime  = f"{int(health.get('uptime_seconds',0)//60)}m {int(health.get('uptime_seconds',0)%60)}s"
else:
    st_html = '<span class="bad">OFFLINE — run: uvicorn app.server:app --host 0.0.0.0 --port 8000</span>'
    uptime  = "—"

st.markdown(f"**Status:** {st_html} &nbsp; Uptime: {uptime}", unsafe_allow_html=True)
if not health:
    st.stop()

st.markdown("<br>", unsafe_allow_html=True)
c1,c2,c3,c4,c5,c6 = st.columns(6)
for col, label, value in [
    (c1, "Total requests",  str(metrics.get("total_requests",0))),
    (c2, "Success rate",    f"{metrics.get('success_rate',0)}%"),
    (c3, "Avg latency",     f"{metrics.get('avg_latency_ms',0)}ms"),
    (c4, "p95 latency",     f"{metrics.get('p95_latency_ms',0)}ms"),
    (c5, "p99 latency",     f"{metrics.get('p99_latency_ms',0)}ms"),
    (c6, "Memory",          f"{health.get('memory_mb',0)}MB"),
]:
    with col:
        st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
col_lat, col_pred = st.columns(2, gap="large")

with col_lat:
    st.markdown("#### Latency over time (ms)")
    if logs:
        df = pd.DataFrame(logs)
        df = df[df["latency_ms"].notna()].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").tail(100)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["latency_ms"], mode="lines+markers",
            line=dict(color="#7F77DD", width=1.5), marker=dict(size=4)))
        fig.add_hline(y=metrics.get("avg_latency_ms",0), line_dash="dash", line_color="#1D9E75",
            annotation_text=f"avg {metrics.get('avg_latency_ms',0)}ms")
        fig.update_layout(height=280, margin=dict(l=40,r=20,t=20,b=40),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Arial",size=12), xaxis_title="Time", yaxis_title="ms")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No requests yet — send some predictions first.")

with col_pred:
    st.markdown("#### Prediction distribution")
    d = metrics.get("defective_pct", 0)
    g = metrics.get("good_pct", 0)
    if d + g > 0:
        fig = go.Figure(go.Pie(labels=["Good","Defective"], values=[g,d],
            marker_colors=["#1D9E75","#D85A30"], hole=0.5, textinfo="label+percent"))
        fig.update_layout(height=280, margin=dict(l=0,r=0,t=20,b=0),
            paper_bgcolor="rgba(0,0,0,0)", showlegend=False, font=dict(family="Arial",size=13))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No predictions yet.")

st.markdown("---")
st.markdown("#### Recent request log")
if logs:
    df_logs = pd.DataFrame(logs[::-1][:50])[["timestamp","endpoint","prediction","confidence","latency_ms","status_code"]]
    df_logs.columns = ["Time","Endpoint","Prediction","Confidence (%)","Latency (ms)","Status"]
    st.dataframe(df_logs, use_container_width=True, hide_index=True, height=300)
else:
    st.info("No logs yet.")

if auto_refresh:
    time.sleep(3)
    st.rerun()