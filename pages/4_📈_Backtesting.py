"""
Página 4: Backtesting
Compara la señal de guidance con el comportamiento bursátil posterior.
"""

import streamlit as st
import sys
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
import config

st.set_page_config(page_title="Backtesting - ForwardGuidanceAI", page_icon="📈", layout="wide")

from src.sector_manager import render_sector_sidebar
active_sector = render_sector_sidebar()

st.title(f"📈 Backtesting de Señal: {config.SECTOR}")
st.markdown(
    f"Sector activo: **{config.SECTOR}** — Comprueba si las empresas con mejor *forward guidance* presentan un "
    "**comportamiento bursátil superior** (alpha sectorial) a 30, 90 y 180 días."
)

st.divider()

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ── Check prerequisites ─────────────────────────────────────────────
scores_path = config.SCORES_DIR / "scores.csv"
if not scores_path.exists():
    st.warning("⚠️ No hay scores calculados. Ejecuta primero **📊 Scoring**.")
    st.stop()

# Check if current sector has scores
scores_check = pd.read_csv(scores_path)
sector_scores = scores_check[scores_check["ticker"].isin(config.TICKERS)]
if sector_scores.empty:
    st.warning(f"⚠️ No hay scores para el sector **{config.SECTOR}**. Ve a **📥 Recopilar Datos** y **🧠 Analizar Guidance** primero.")
    st.stop()

# ── Step 1: Download Market Data ────────────────────────────────────
st.subheader("1️⃣ Descargar Datos de Mercado")

prices_downloaded = all(
    (config.PRICES_DIR / f"{t}_prices.csv").exists()
    for t in config.TICKERS
)

if prices_downloaded:
    st.success("✅ Precios históricos ya descargados para todas las empresas.")
else:
    if st.button("📥 Descargar Precios Históricos (yfinance)", type="secondary"):
        try:
            from src.market_data import download_all_prices
            with st.spinner("Descargando precios..."):
                all_prices = download_all_prices()
            st.success(f"✅ Precios descargados para {len(all_prices)} empresas.")
        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)

st.divider()

# ── Step 2: Run Backtest ────────────────────────────────────────────
st.subheader("2️⃣ Ejecutar Backtest")

backtest_path = config.SCORES_DIR / "backtest_results.json"

col1, col2 = st.columns(2)
with col1:
    return_window = st.selectbox(
        "Ventana de retorno principal",
        options=[30, 90, 180],
        index=1,
        format_func=lambda x: f"{x} días ({x//30} {'mes' if x==30 else 'meses'})",
    )

with col2:
    score_column = st.selectbox(
        "Variable de score",
        options=["composite_score", "overall_sentiment"],
        index=0,
    )

if st.button("🚀 Ejecutar Backtest", type="primary", use_container_width=True):
    try:
        from src.backtester import run_backtest
        from src.market_data import download_all_prices, calculate_sector_returns

        with st.spinner("Calculando retornos y ejecutando backtest..."):
            results = run_backtest()

        st.success("✅ Backtest completado.")

    except Exception as e:
        st.error(f"Error en backtest: {e}")
        st.exception(e)

# ── Step 3: Show Results ────────────────────────────────────────────
st.divider()
st.subheader("3️⃣ Resultados del Backtest")

if backtest_path.exists():
    with open(backtest_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    # ── Key Metrics ─────────────────────────────────────────────
    st.markdown("### 📊 Métricas Principales")

    for window in [30, 90, 180]:
        window_key = f"window_{window}d"
        if window_key in results:
            wr = results[window_key]
            st.markdown(f"#### Ventana de {window} días")

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                ic_val = wr.get("ic", {}).get("ic", "N/A")
                ic_display = f"{ic_val:.3f}" if isinstance(ic_val, (int, float)) else ic_val
                st.metric("Information Coefficient", ic_display)
            with m2:
                p_val = wr.get("ic", {}).get("p_value", "N/A")
                p_display = f"{p_val:.4f}" if isinstance(p_val, (int, float)) else p_val
                significant = "✅" if isinstance(p_val, (int, float)) and p_val < 0.05 else "❌"
                st.metric("p-value", f"{p_display} {significant}")
            with m3:
                ls_spread = wr.get("long_short", {}).get("long_short_spread", "N/A")
                ls_display = f"{ls_spread:.2%}" if isinstance(ls_spread, (int, float)) else ls_spread
                st.metric("Long-Short Spread", ls_display)
            with m4:
                hr = wr.get("long_short", {}).get("hit_rate", "N/A")
                hr_display = f"{hr:.1%}" if isinstance(hr, (int, float)) else hr
                st.metric("Hit Rate", hr_display)

    st.divider()

    # ── Scatter Plot: Score vs Returns ──────────────────────────
    st.markdown("### 📉 Score vs Retorno Posterior")

    # Load merged data if available
    merged_path = config.SCORES_DIR / "merged_scores_returns.csv"
    if merged_path.exists():
        merged_df = pd.read_csv(merged_path)

        return_col = f"alpha_{return_window}d"
        if return_col in merged_df.columns:
            fig_scatter = px.scatter(
                merged_df,
                x=score_column,
                y=return_col,
                color="ticker",
                hover_data=["filing_date", "company_name"],
                trendline="ols",
                title=f"Score de Guidance vs Alpha {return_window}d",
                labels={
                    score_column: "Score de Guidance",
                    return_col: f"Alpha {return_window}d (%)",
                },
            )
            fig_scatter.update_layout(height=500)
            st.plotly_chart(fig_scatter, use_container_width=True)

        # ── Quantile Returns Chart ──────────────────────────────
        st.markdown("### 📊 Retornos por Cuantil de Score")

        if score_column in merged_df.columns and return_col in merged_df.columns:
            merged_df["score_quantile"] = pd.qcut(
                merged_df[score_column], q=3, labels=["Bottom", "Middle", "Top"],
                duplicates="drop"
            )
            quantile_returns = merged_df.groupby("score_quantile")[return_col].mean().reset_index()

            fig_q = px.bar(
                quantile_returns,
                x="score_quantile",
                y=return_col,
                color="score_quantile",
                color_discrete_map={"Bottom": "#ef4444", "Middle": "#f59e0b", "Top": "#22c55e"},
                title=f"Retorno Medio por Cuantil de Score ({return_window}d)",
            )
            fig_q.update_layout(height=400)
            st.plotly_chart(fig_q, use_container_width=True)

    # ── Summary Text ────────────────────────────────────────────
    st.divider()
    st.markdown("### 📝 Resumen del Backtest")

    summary = results.get("summary", "")
    if summary:
        st.markdown(summary)
    else:
        st.info("No hay resumen disponible. Ejecuta el backtest completo.")

else:
    st.info(
        "No hay resultados de backtest todavía. Descarga los precios "
        "y ejecuta el backtest usando los botones de arriba."
    )

# ── Interpretation Guide ────────────────────────────────────────────
st.divider()

with st.expander("📖 Guía de Interpretación"):
    st.markdown("""
    **Information Coefficient (IC):**
    - Correlación de Spearman entre el score de guidance y el retorno posterior
    - IC > 0.05 sugiere poder predictivo; IC > 0.10 es una señal fuerte
    - p-value < 0.05 indica significancia estadística

    **Long-Short Spread:**
    - Retorno del tercil superior menos retorno del tercil inferior de score
    - Un spread positivo indica que las empresas con mejor guidance tienden a rendir más

    **Hit Rate:**
    - % de períodos en los que el grupo con mejor score superó al grupo con peor score
    - Hit Rate > 60% sugiere consistencia de la señal

    **Alpha:**
    - Retorno de la empresa menos el retorno medio del sector
    - Elimina el efecto del mercado y aísla el rendimiento relativo
    """)
