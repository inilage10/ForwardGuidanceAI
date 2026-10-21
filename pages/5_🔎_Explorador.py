"""
Página 5: Explorador
Detalle individual por empresa: guidance extraído, citas, evolución del score y precio.
"""

import streamlit as st
import sys
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
import config

st.set_page_config(page_title="Explorador - ForwardGuidanceAI", page_icon="🔎", layout="wide")

from src.sector_manager import render_sector_sidebar
active_sector = render_sector_sidebar()

st.title(f"🔎 Explorador de Empresas: {config.SECTOR}")
st.markdown(
    f"Sector activo: **{config.SECTOR}** — Analiza en detalle el *forward guidance* de cada empresa: "
    "citas originales del 10-K/10-Q, evolución del score y comparación con su cotización bursátil."
)

st.divider()

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Company Selector ────────────────────────────────────────────────
current_tickers = list(config.TICKERS.keys())
selected_ticker = st.selectbox(
    "Selecciona una empresa",
    options=current_tickers,
    format_func=lambda t: f"{t} - {config.TICKERS.get(t, t)}",
    key=f"explorer_ticker_{active_sector}",
)

st.divider()

# ── Load Data ───────────────────────────────────────────────────────
all_guidance_path = config.PROCESSED_DIR / "all_guidance.json"
scores_path = config.SCORES_DIR / "scores.csv"
prices_path = config.PRICES_DIR / f"{selected_ticker}_prices.csv"

# ── Section 1: Company Overview ─────────────────────────────────────
st.subheader(f"🏢 {config.TICKERS[selected_ticker]} ({selected_ticker})")

col1, col2 = st.columns(2)

# Price chart
with col1:
    st.markdown("**📈 Evolución del Precio**")
    if prices_path.exists():
        prices_df = pd.read_csv(prices_path)
        prices_df["Date"] = pd.to_datetime(prices_df["Date"])

        fig_price = px.line(
            prices_df, x="Date", y="Close",
            title=f"Precio de {selected_ticker}",
        )
        fig_price.update_layout(height=350, xaxis_title="", yaxis_title="Precio (USD)")
        st.plotly_chart(fig_price, use_container_width=True)
    else:
        st.info("Precios no disponibles. Descárgalos en la página de **📈 Backtesting**.")

# Score evolution
with col2:
    st.markdown("**📊 Evolución del Score**")
    if scores_path.exists():
        scores_df = pd.read_csv(scores_path)
        ticker_scores = scores_df[scores_df["ticker"] == selected_ticker].copy()
        ticker_scores["filing_date"] = pd.to_datetime(ticker_scores["filing_date"], errors="coerce")
        ticker_scores = ticker_scores.sort_values("filing_date")

        if not ticker_scores.empty:
            fig_score = go.Figure()
            fig_score.add_trace(go.Scatter(
                x=ticker_scores["filing_date"],
                y=ticker_scores["composite_score"],
                mode="lines+markers",
                name="Score Compuesto",
                line=dict(color="#667eea", width=3),
                marker=dict(size=10),
            ))
            fig_score.add_trace(go.Scatter(
                x=ticker_scores["filing_date"],
                y=ticker_scores["overall_sentiment"],
                mode="lines+markers",
                name="Sentiment General",
                line=dict(color="#f59e0b", width=2, dash="dash"),
                marker=dict(size=8),
            ))
            fig_score.update_layout(
                height=350,
                title=f"Score de Guidance - {selected_ticker}",
                xaxis_title="",
                yaxis_title="Score",
                yaxis=dict(range=[-1, 1]),
            )
            st.plotly_chart(fig_score, use_container_width=True)
        else:
            st.info("No hay scores para esta empresa.")
    else:
        st.info("Scores no calculados. Ejecuta **📊 Scoring** primero.")

st.divider()

# ── Section 2: Guidance Details ─────────────────────────────────────
st.subheader("📋 Detalle del Forward Guidance")

if all_guidance_path.exists():
    with open(all_guidance_path, "r", encoding="utf-8") as f:
        all_guidance = json.load(f)

    ticker_guidance = all_guidance.get(selected_ticker, [])

    if ticker_guidance:
        # Sort by filing date
        ticker_guidance_sorted = sorted(
            ticker_guidance,
            key=lambda x: x.get("filing_date", ""),
            reverse=True,
        )

        for idx, guidance in enumerate(ticker_guidance_sorted):
            filing_date = guidance.get("filing_date", "N/A")
            filing_type = guidance.get("filing_type", "N/A")
            sentiment = guidance.get("overall_sentiment", "N/A")
            summary = guidance.get("summary", "Sin resumen disponible")
            risks = guidance.get("key_risks", [])

            with st.expander(
                f"📄 {filing_date} | {filing_type} | Sentiment: {sentiment}",
                expanded=(idx == 0),
            ):
                st.markdown(f"**Resumen:** {summary}")

                if risks:
                    st.markdown("**⚠️ Riesgos clave:**")
                    for risk in risks:
                        st.markdown(f"- {risk}")

                # Dimension details
                guidance_data = guidance.get("guidance", {})
                if guidance_data:
                    st.markdown("**📊 Dimensiones del Guidance:**")

                    dim_rows = []
                    for dim_name, dim_data in guidance_data.items():
                        if isinstance(dim_data, dict):
                            dim_rows.append({
                                "Dimensión": dim_name.replace("_", " ").title(),
                                "Dirección": dim_data.get("direction", "N/A"),
                                "Magnitud": dim_data.get("magnitude", "N/A"),
                                "Confianza": dim_data.get("confidence", "N/A"),
                                "Cita": dim_data.get("quote", dim_data.get("direct_quote", "N/A")),
                            })

                    if dim_rows:
                        dim_df = pd.DataFrame(dim_rows)
                        st.dataframe(dim_df, use_container_width=True, hide_index=True)

                        # Spider chart for this filing
                        dimensions = [r["Dimensión"] for r in dim_rows]
                        from src.scorer import DIRECTION_MAP, MAGNITUDE_MAP
                        scores_list = []
                        for r in dim_rows:
                            d_val = DIRECTION_MAP.get(str(r["Dirección"]).lower(), 0)
                            m_val = MAGNITUDE_MAP.get(str(r["Magnitud"]).lower(), 0.5)
                            c_val = float(r["Confianza"]) if isinstance(r["Confianza"], (int, float)) else 0.5
                            scores_list.append(d_val * m_val * c_val)

                        fig_radar = go.Figure()
                        fig_radar.add_trace(go.Scatterpolar(
                            r=scores_list + [scores_list[0]],
                            theta=dimensions + [dimensions[0]],
                            fill="toself",
                            name=filing_date,
                            line=dict(color="#667eea"),
                        ))
                        fig_radar.update_layout(
                            polar=dict(radialaxis=dict(range=[-1, 1])),
                            title=f"Perfil de Guidance - {filing_date}",
                            height=400,
                        )
                        st.plotly_chart(fig_radar, use_container_width=True)

    else:
        st.info("No hay datos de guidance para esta empresa. Ejecuta **🧠 Analizar Guidance**.")
else:
    st.info("No hay datos de guidance. Ejecuta **🧠 Analizar Guidance** primero.")

# ── Section 3: Comparison with Sector ───────────────────────────────
st.divider()
st.subheader("🏆 Posición en el Sector")

if scores_path.exists():
    scores_df = pd.read_csv(scores_path)
    scores_df["filing_date"] = pd.to_datetime(scores_df["filing_date"], errors="coerce")

    # Latest scores for all companies
    latest = scores_df.sort_values("filing_date").groupby("ticker").last().reset_index()
    latest = latest.sort_values("composite_score", ascending=False)
    latest["rank"] = range(1, len(latest) + 1)

    # Highlight selected company
    company_rank = latest[latest["ticker"] == selected_ticker]["rank"].values
    rank_display = company_rank[0] if len(company_rank) > 0 else "N/A"

    col_a, col_b = st.columns(2)

    with col_a:
        st.metric(
            "Posición en el sector",
            f"#{rank_display} de {len(latest)}",
        )

    with col_b:
        company_score = latest[latest["ticker"] == selected_ticker]["composite_score"].values
        score_display = f"{company_score[0]:.3f}" if len(company_score) > 0 else "N/A"
        sector_avg = latest["composite_score"].mean()
        delta = company_score[0] - sector_avg if len(company_score) > 0 else 0
        st.metric(
            "Score vs Media Sector",
            score_display,
            delta=f"{delta:+.3f}",
        )

    # Bar chart with highlight
    colors = ["#667eea" if t == selected_ticker else "#c3cfe2" for t in latest["ticker"]]
    fig_compare = go.Figure(go.Bar(
        x=latest["ticker"],
        y=latest["composite_score"],
        marker_color=colors,
        text=latest["composite_score"].round(3),
        textposition="auto",
    ))
    fig_compare.update_layout(
        title="Score Compuesto - Todas las Empresas",
        height=400,
        yaxis=dict(range=[-1, 1]),
        xaxis_title="",
        yaxis_title="Score Compuesto",
    )
    st.plotly_chart(fig_compare, use_container_width=True)
