"""
Página 1: Recopilar Datos
Descarga filings de SEC EDGAR para las empresas del sector.
"""

import streamlit as st
import sys
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
import config

st.set_page_config(page_title="Recopilar Datos - ForwardGuidanceAI", page_icon="📥", layout="wide")

from src.sector_manager import render_sector_sidebar, get_active_sector_details
active_sector = render_sector_sidebar()

st.title("📥 Recopilar Datos de SEC EDGAR y Mercado")
st.markdown(
    f"Sector activo: **{config.SECTOR}** — Descarga filings **10-K** (anuales) y **10-Q** (trimestrales) "
    f"de la SEC y cotizaciones bursátiles para las 10 empresas líderes por capitalización bursátil."
)

st.divider()

# ── Top 10 Sector Table ─────────────────────────────────────────────
st.subheader(f"🏆 Top 10 por Market Cap: {config.SECTOR}")
sector_details = get_active_sector_details()
details_df = pd.DataFrame(sector_details)

# Check local filing counts
for idx, row in details_df.iterrows():
    tk = row["Ticker"]
    raw_dir = config.RAW_DIR / tk
    f_count = len([f for f in raw_dir.rglob("*") if f.is_file() and f.suffix in ('.htm', '.html', '.txt')]) if raw_dir.exists() else 0
    p_file = config.PRICES_DIR / f"{tk}_prices.csv"
    p_status = "✅ Descargado" if p_file.exists() else "Pendiente"
    details_df.loc[idx, "Filings Descargados"] = f_count
    details_df.loc[idx, "Precios Históricos"] = p_status

st.dataframe(details_df, use_container_width=True, hide_index=True)

st.divider()

# ── Configuration ───────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Selección de empresas")
    current_tickers = list(config.TICKERS.keys())
    selected_tickers = st.multiselect(
        "Empresas a descargar",
        options=current_tickers,
        default=current_tickers,
        format_func=lambda t: f"{t} - {config.TICKERS.get(t, t)}",
        key=f"tickers_download_{active_sector}",
    )

with col2:
    st.subheader("⚙️ Configuración SEC EDGAR")
    filing_types = st.multiselect(
        "Tipos de filing",
        options=["10-K", "10-Q"],
        default=config.FILING_TYPES,
    )
    st.markdown(f"**Período:** {config.ANALYSIS_START_DATE} → {config.ANALYSIS_END_DATE}")
    email = st.text_input(
        "Email para SEC User-Agent (obligatorio)",
        value=config.SEC_EDGAR_EMAIL or "research@university.edu",
        help="La SEC requiere un email de contacto válido en todas las peticiones",
    )

st.divider()

if not email:
    st.warning("⚠️ Introduce un email válido para poder descargar de la SEC.")
    st.stop()

if st.button("🚀 Descargar Filings", type="primary", use_container_width=True):
    try:
        from src.collector import collect_all, get_cik, get_filings_list, download_filing

        progress_bar = st.progress(0)
        status_text = st.empty()
        log_container = st.container()

        total_tickers = len(selected_tickers)
        all_results = {}

        for i, ticker in enumerate(selected_tickers):
            status_text.markdown(f"**Procesando {ticker} ({i+1}/{total_tickers})...**")

            with log_container:
                try:
                    # Get CIK
                    st.text(f"  📡 Obteniendo CIK para {ticker}...")
                    cik = get_cik(ticker)
                    st.text(f"  ✅ CIK: {cik}")

                    # Get filings list
                    ticker_filings = []
                    for ft in filing_types:
                        st.text(f"  📋 Buscando filings {ft}...")
                        filings = get_filings_list(cik, ft)
                        st.text(f"  ✅ Encontrados {len(filings)} filings {ft}")
                        ticker_filings.extend(filings)

                    # Download each filing
                    downloaded = []
                    for j, filing in enumerate(ticker_filings):
                        save_dir = config.RAW_DIR / ticker
                        save_dir.mkdir(parents=True, exist_ok=True)
                        try:
                            path = download_filing(
                                cik, filing["accession_number"],
                                filing["primary_document"], save_dir
                            )
                            downloaded.append(str(path))
                            st.text(f"  💾 [{j+1}/{len(ticker_filings)}] {filing['filing_date']} - {filing['form_type']}")
                        except Exception as e:
                            st.text(f"  ⚠️ Error descargando {filing['filing_date']}: {e}")

                    all_results[ticker] = downloaded

                    # Download prices from yfinance
                    p_file = config.PRICES_DIR / f"{ticker}_prices.csv"
                    if not p_file.exists():
                        try:
                            from src.market_data import download_prices
                            download_prices(ticker)
                            st.text(f"  📈 Cotizaciones de mercado descargadas para {ticker}")
                        except Exception as pe:
                            st.text(f"  ⚠️ Precios no descargados: {pe}")

                except Exception as e:
                    st.error(f"❌ Error con {ticker}: {e}")

            progress_bar.progress((i + 1) / total_tickers)

        status_text.markdown("**✅ Descarga completada**")
        st.success(
            f"Se han descargado filings y precios para {len(all_results)} empresas de {config.SECTOR}. "
            f"Continúa al paso **🧠 Analizar Guidance**."
        )
        st.balloons()

    except ImportError as e:
        st.error(f"Error importando módulos: {e}")
        st.info("Asegúrate de haber instalado las dependencias: `pip install -r requirements.txt`")
    except Exception as e:
        st.error(f"Error durante la descarga: {e}")
        st.exception(e)
