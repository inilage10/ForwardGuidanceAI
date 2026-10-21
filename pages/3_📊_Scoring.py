"""
Página 3: Scoring
Calcula y muestra los scores comparativos de guidance entre empresas.
"""

import streamlit as st
import sys
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
import config

st.set_page_config(page_title="Scoring - ForwardGuidanceAI", page_icon="📊", layout="wide")

from src.sector_manager import render_sector_sidebar
active_sector = render_sector_sidebar()

st.title(f"📊 Scoring de Forward Guidance: {config.SECTOR}")
st.markdown(
    f"Sector activo: **{config.SECTOR}** — Convierte el guidance extraído en **scores numéricos comparables** "
    "entre las empresas líderes y genera rankings trimestrales."
)

st.divider()

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Check for guidance data ─────────────────────────────────────────
all_guidance_path = config.PROCESSED_DIR / "all_guidance.json"

if not all_guidance_path.exists():
    st.warning("⚠️ No hay datos de guidance. Ejecuta primero **🧠 Analizar Guidance**.")
    st.stop()

# ── Calculate Scores ────────────────────────────────────────────────
scores_path = config.SCORES_DIR / "scores.csv"

if st.button("🔄 Recalcular Scores", type="primary") or not scores_path.exists():
    try:
        from src.scorer import score_all
        with st.spinner("Calculando scores..."):
            scores_df = score_all()
        st.success(f"✅ Scores calculados para {len(scores_df)} filings.")
    except Exception as e:
        st.error(f"Error calculando scores: {e}")
        st.exception(e)
        st.stop()

# Load scores and filter by active sector
try:
    scores_df = pd.read_csv(scores_path)
    scores_df = scores_df[scores_df["ticker"].isin(config.TICKERS)]
except Exception as e:
    st.error(f"Error cargando scores: {e}")
    st.stop()

if scores_df.empty:
    st.warning(f"No hay scores calculados todavía para el sector **{config.SECTOR}**. Ve a **📥 Recopilar Datos** y **🧠 Analizar Guidance** para procesarlo.")
    st.stop()

# ── Latest Rankings ─────────────────────────────────────────────────
st.subheader(f"🏆 Ranking Actual ({config.SECTOR})")

# Get latest score per company
scores_df["filing_date"] = pd.to_datetime(scores_df["filing_date"], errors="coerce")
latest_scores = scores_df.sort_values("filing_date").groupby("ticker").last().reset_index()
latest_scores = latest_scores.sort_values("composite_score", ascending=False)

# Display ranking
col1, col2 = st.columns([2, 1])

with col1:
    display_cols = ["ticker", "company_name", "composite_score", "overall_sentiment", "filing_date"]
    dimension_cols = [c for c in latest_scores.columns if c in config.GUIDANCE_DIMENSIONS]
    display_df = latest_scores[display_cols + dimension_cols].copy()
    display_df["filing_date"] = display_df["filing_date"].dt.strftime("%Y-%m-%d")

    # Add rank
    display_df.insert(0, "Rank", range(1, len(display_df) + 1))

    # Format scores
    score_cols = ["composite_score", "overall_sentiment"] + dimension_cols
    for col in score_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(3)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "composite_score": st.column_config.ProgressColumn(
                "Score Compuesto",
                min_value=-1,
                max_value=1,
                format="%.3f",
            ),
        },
    )

with col2:
    # Bar chart of composite scores
    fig = px.bar(
        latest_scores,
        x="composite_score",
        y="ticker",
        orientation="h",
        color="composite_score",
        color_continuous_scale="RdYlGn",
        range_color=[-1, 1],
        title="Score Compuesto por Empresa",
    )
    fig.update_layout(height=400, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Heatmap of Dimensions ──────────────────────────────────────────
st.subheader("🗺️ Heatmap de Dimensiones del Guidance")

if dimension_cols:
    heatmap_data = latest_scores.set_index("ticker")[dimension_cols]

    fig_heat = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=[d.replace("_", " ").title() for d in dimension_cols],
        y=heatmap_data.index,
        colorscale="RdYlGn",
        zmid=0,
        zmin=-1,
        zmax=1,
        text=heatmap_data.values.round(2),
        texttemplate="%{text}",
        textfont={"size": 11},
    ))
    fig_heat.update_layout(
        title="Score por Dimensión y Empresa",
        height=500,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

st.divider()

# ── Score Evolution Over Time ───────────────────────────────────────
st.subheader("📈 Evolución del Score en el Tiempo")

fig_evolution = px.line(
    scores_df.sort_values("filing_date"),
    x="filing_date",
    y="composite_score",
    color="ticker",
    markers=True,
    title="Evolución del Score Compuesto por Empresa",
)
fig_evolution.update_layout(
    height=500,
    xaxis_title="Fecha del Filing",
    yaxis_title="Score Compuesto",
    yaxis=dict(range=[-1, 1]),
)
st.plotly_chart(fig_evolution, use_container_width=True)

# ── Sector Statistics ───────────────────────────────────────────────
st.divider()
st.subheader("📉 Estadísticas del Sector")

col_a, col_b, col_c, col_d = st.columns(4)

with col_a:
    st.metric("Score medio sector", f"{latest_scores['composite_score'].mean():.3f}")
with col_b:
    st.metric("Score máximo", f"{latest_scores['composite_score'].max():.3f}")
with col_c:
    st.metric("Score mínimo", f"{latest_scores['composite_score'].min():.3f}")
with col_d:
    st.metric("Desviación estándar", f"{latest_scores['composite_score'].std():.3f}")
