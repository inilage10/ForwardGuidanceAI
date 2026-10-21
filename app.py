"""
ForwardGuidanceAI - Main Streamlit Application
================================================
IA para identificar empresas con mejores perspectivas bursátiles
dentro del sector Tecnología, analizando el forward guidance
de filings de la SEC.

Authors: Iñigo Lage, Ignacio García
"""

import streamlit as st

# ── Page Configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="ForwardGuidanceAI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .stMetric {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ── Main Page ───────────────────────────────────────────────────────
st.markdown('<p class="main-header">🔮 ForwardGuidanceAI</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">'
    'Sistema de IA para identificar empresas con mejores perspectivas bursátiles '
    'a partir del análisis del <em>forward guidance</em> corporativo'
    '</p>',
    unsafe_allow_html=True,
)

# ── Status Dashboard ────────────────────────────────────────────────
import sys
from pathlib import Path

# Add project root to path
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

try:
    import config
    from src.sector_manager import render_sector_sidebar
    active_sector = render_sector_sidebar()

    st.divider()

    # Show sector and companies
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Sector S&P 500 Activo", config.SECTOR)
    with col2:
        st.metric("Empresas en Análisis", len(config.TICKERS))
    with col3:
        st.metric("Período Histórico", f"{config.ANALYSIS_START_DATE[:4]}-{config.ANALYSIS_END_DATE[:4]}")
    with col4:
        st.metric("Proveedor de IA", "Groq / NLP / Ollama")

    st.divider()

    # Company list
    st.subheader("🏢 Empresas en análisis")

    import pandas as pd
    companies_df = pd.DataFrame([
        {"Ticker": ticker, "Empresa": name}
        for ticker, name in config.TICKERS.items()
    ])

    # Check data status
    status_data = []
    for ticker in config.TICKERS:
        raw_dir = config.RAW_DIR / ticker
        raw_count = len(list(raw_dir.glob("*"))) if raw_dir.exists() else 0

        processed_dir = config.PROCESSED_DIR / ticker
        guidance_files = list(processed_dir.glob("*_guidance.json")) if processed_dir.exists() else []
        guidance_count = len(guidance_files)

        prices_file = config.PRICES_DIR / f"{ticker}_prices.csv"
        has_prices = "✅" if prices_file.exists() else "❌"

        status_data.append({
            "Ticker": ticker,
            "Empresa": config.TICKERS[ticker],
            "Filings descargados": raw_count,
            "Guidance analizado": guidance_count,
            "Precios": has_prices,
        })

    status_df = pd.DataFrame(status_data)
    st.dataframe(status_df, use_container_width=True, hide_index=True)

    # Check scores
    scores_file = config.SCORES_DIR / "scores.csv"
    backtest_file = config.SCORES_DIR / "backtest_results.json"

    st.divider()

    st.subheader("📋 Pipeline de Análisis")

    steps = [
        ("📥 Recopilar Datos", "Descargar filings de SEC EDGAR", "1_📥_Recopilar_Datos"),
        ("🧠 Analizar Guidance", "Extraer guidance con Ollama", "2_🧠_Analizar_Guidance"),
        ("📊 Scoring", "Calcular scores comparativos", "3_📊_Scoring"),
        ("📈 Backtesting", "Validar señal vs retornos", "4_📈_Backtesting"),
        ("🔎 Explorador", "Analizar empresas individualmente", "5_🔎_Explorador"),
    ]

    for i, (title, desc, _) in enumerate(steps, 1):
        st.markdown(f"**Paso {i}: {title}** — {desc}")

    st.info(
        "👈 Usa el menú lateral para navegar por cada paso del pipeline. "
        "Comienza por **📥 Recopilar Datos** para descargar los filings de la SEC."
    )

    # Ollama status check
    st.divider()
    st.subheader("🔧 Estado del Sistema")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Ollama (LLM Local)**")
        try:
            from src.analyzer import check_ollama_status
            is_ok, msg = check_ollama_status()
            if is_ok:
                st.success(f"✅ {msg}")
            else:
                st.error(f"❌ {msg}")
                st.markdown(
                    "Para instalar Ollama:\n"
                    "1. Descarga de [ollama.com](https://ollama.com)\n"
                    "2. Ejecuta: `ollama pull llama3.1`\n"
                    "3. Asegúrate de que Ollama está corriendo"
                )
        except Exception as e:
            st.warning(f"⚠️ No se pudo verificar Ollama: {e}")

    with col_b:
        st.markdown("**Datos**")
        raw_total = sum(1 for _ in config.RAW_DIR.rglob("*") if _.is_file()) if config.RAW_DIR.exists() else 0
        st.markdown(f"- Filings descargados: **{raw_total}**")
        st.markdown(f"- Scores calculados: **{'✅' if scores_file.exists() else '❌'}**")
        st.markdown(f"- Backtest completado: **{'✅' if backtest_file.exists() else '❌'}**")

except ImportError as e:
    st.error(f"Error importando configuración: {e}")
    st.info("Asegúrate de haber instalado las dependencias: `pip install -r requirements.txt`")
except Exception as e:
    st.error(f"Error: {e}")
    st.exception(e)

# ── Footer ──────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align: center; color: #9ca3af; font-size: 0.85rem;'>"
    "ForwardGuidanceAI · Análisis de Servicios Financieros · "
    "Universidad Pontificia Comillas · 2026"
    "</div>",
    unsafe_allow_html=True,
)
