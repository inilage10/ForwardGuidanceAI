"""
Página 2: Analizar Guidance
Ejecuta el LLM (Ollama) sobre los filings para extraer forward guidance.
"""

import streamlit as st
import sys
import json
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
import config

st.set_page_config(page_title="Analizar Guidance - ForwardGuidanceAI", page_icon="🧠", layout="wide")

from src.sector_manager import render_sector_sidebar
active_sector = render_sector_sidebar()

st.title("🧠 Analizar Forward Guidance con IA")
st.markdown(
    f"Sector activo: **{config.SECTOR}** — Extrae el *forward guidance* corporativo "
    "de los filings de la SEC mediante IA o NLP cuantitativo para convertirlos en señales estructuradas."
)

st.divider()

# ── Estado del Sistema ───────────────────────────────────────────────
with st.expander("🔧 Diagnóstico de proveedores de IA y estado local", expanded=False):
    col_a, col_b = st.columns(2)
    with col_a:
        try:
            from src.analyzer import check_ollama_status
            is_ok, msg = check_ollama_status()
            if is_ok:
                st.success(f"Ollama: {msg}")
            else:
                st.warning(f"Ollama: {msg}")
        except Exception as e:
            st.info(f"Ollama: No verificado ({e})")
    with col_b:
        if config.GROQ_API_KEY:
            st.success("Groq Cloud: Clave configurada en entorno")
        else:
            st.info("Groq Cloud: Clave opcional (obtenible en console.groq.com)")

st.divider()

# ── Filing Selection ────────────────────────────────────────────────
st.subheader("📄 Filings disponibles para análisis")

import pandas as pd

# Check for parsed filings
parsed_dir = config.PROCESSED_DIR
parsed_files_exist = False

filing_status = []
for ticker in config.TICKERS:
    raw_dir = config.RAW_DIR / ticker
    if raw_dir.exists():
        raw_files = [f for f in raw_dir.rglob("*") if f.is_file() and f.suffix in ('.htm', '.html', '.txt')]
    else:
        raw_files = []

    # Check for existing guidance analysis
    proc_dir = config.PROCESSED_DIR / ticker
    if proc_dir.exists():
        guidance_files = list(proc_dir.glob("*_guidance.json"))
    else:
        guidance_files = []

    filing_status.append({
        "Ticker": ticker,
        "Empresa": config.TICKERS[ticker],
        "Filings disponibles": len(raw_files),
        "Ya analizados": len(guidance_files),
        "Pendientes": max(0, len(raw_files) - len(guidance_files)),
    })

    if raw_files:
        parsed_files_exist = True

status_df = pd.DataFrame(filing_status)
st.dataframe(status_df, use_container_width=True, hide_index=True)

if not parsed_files_exist:
    st.warning(f"⚠️ No hay filings descargados todavía para las empresas de **{config.SECTOR}**. Ve a **📥 Recopilar Datos** primero para descargarlos con un solo clic.")
    st.stop()

# ── Analysis Options ────────────────────────────────────────────────
st.divider()
st.subheader("⚙️ Opciones de análisis")

col1, col2 = st.columns(2)

with col1:
    current_tickers = list(config.TICKERS.keys())
    selected_tickers = st.multiselect(
        "Empresas a analizar",
        options=current_tickers,
        default=current_tickers,
        format_func=lambda t: f"{t} - {config.TICKERS.get(t, t)}",
        key=f"tickers_analyze_{active_sector}",
    )
    skip_existing = st.checkbox("Saltar filings ya analizados", value=True)

with col2:
    backend_mode = st.radio(
        "Proveedor de IA",
        [
            "⚡ Motor Cuantitativo NLP (100% Local, ~2 seg todo el dataset, Sin Claves)",
            "🚀 Groq Cloud API (Llama 3.3 70B en la nube, Requiere Clave Gratis)",
            "💻 Ollama Local (LLM en tu CPU, ej. Qwen 2.5)"
        ],
        index=0,
    )

    if "NLP" in backend_mode:
        groq_key = ""
        model_name = "nlp_local"
        st.success("✨ **Recomendado:** Análisis financiero instantáneo y determinista basado en el léxico Loughran & McDonald. 0 errores de red, 0 coste.")
    elif "Groq" in backend_mode:
        groq_key = st.text_input(
            "Groq API Key (Gratis)",
            value=config.GROQ_API_KEY,
            type="password",
            help="Consigue tu clave gratis en https://console.groq.com/keys",
        )
        if not groq_key:
            st.info("👉 **[Consigue una clave gratis en console.groq.com](https://console.groq.com/keys)** (tarda 30 seg, sin tarjeta)")
        
        # Populate live Groq models or use stable Llama 3.3 70B
        live_groq_models = []
        if groq_key:
            try:
                from src.analyzer import get_groq_models
                live_groq_models = get_groq_models(groq_key)
            except Exception:
                pass
        
        groq_choices = ["groq:llama-3.3-70b-versatile"]
        for gm in live_groq_models:
            tagged = f"groq:{gm}"
            if tagged not in groq_choices and "whisper" not in gm and "guard" not in gm:
                groq_choices.append(tagged)

        model_name = st.selectbox(
            "Modelo Groq",
            options=groq_choices,
            index=0,
            help="llama-3.3-70b-versatile es el modelo oficial insignia de Meta en Groq.",
        )
    else:
        groq_key = ""
        try:
            from src.analyzer import get_available_models, check_ollama_status
            is_ok, msg = check_ollama_status()
            if not is_ok:
                st.warning(f"⚠️ {msg}")
            avail_models = get_available_models()
        except Exception:
            avail_models = []

        model_options = []
        for m in ["qwen2.5:1.5b", "qwen2.5:0.5b"]:
            if m in avail_models and m not in model_options:
                model_options.append(m)
        for m in avail_models:
            if m not in model_options:
                model_options.append(m)
        if not model_options:
            model_options = ["qwen2.5:1.5b", "qwen2.5:0.5b"]

        model_name = st.selectbox(
            "Modelo LLM (Ollama)",
            options=model_options,
            index=0,
            help="qwen2.5:1.5b ofrece buena calidad de análisis en local.",
        )

# ── Run Analysis ────────────────────────────────────────────────────
st.divider()

if st.button("🚀 Ejecutar Análisis de Guidance", type="primary", use_container_width=True):
    try:
        import importlib
        import src.analyzer
        import src.parser
        importlib.reload(src.analyzer)
        importlib.reload(src.parser)
        from src.parser import parse_filing
        from src.analyzer import analyze_filing

        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.container()

        total_analyzed = 0
        total_from_cache = 0
        total_errors = 0
        all_results = {}

        # Count total filings to process
        total_filings = 0
        for ticker in selected_tickers:
            raw_dir = config.RAW_DIR / ticker
            if raw_dir.exists():
                raw_files = [f for f in raw_dir.rglob("*") if f.is_file() and f.suffix in ('.htm', '.html', '.txt')]
                total_filings += len(raw_files)

        processed = 0

        for ticker in selected_tickers:
            status_text.markdown(f"**Analizando {ticker} - {config.TICKERS[ticker]}...**")

            raw_dir = config.RAW_DIR / ticker
            if not raw_dir.exists():
                continue

            raw_files = sorted([f for f in raw_dir.rglob("*") if f.is_file() and f.suffix in ('.htm', '.html', '.txt')])

            ticker_results = []

            for filepath in raw_files:
                processed += 1

                # Accurate filing date and type extraction
                fname = filepath.name.lower()
                if "10-k" in fname or "10k" in fname:
                    filing_type = "10-K"
                else:
                    filing_type = "10-Q"

                import re
                m1 = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", filepath.name)
                m2 = re.search(r"[-_](20\d{2})(\d{2})(\d{2})", filepath.name)
                m3 = re.search(r"(20\d{2})(\d{2})(\d{2})", filepath.name)
                m4 = re.search(r"\d{10}-(\d{2})-\d{6}", filepath.name)
                if m1:
                    filing_date = f"{m1.group(1)}-{m1.group(2)}-{m1.group(3)}"
                elif m2:
                    filing_date = f"{m2.group(1)}-{m2.group(2)}-{m2.group(3)}"
                elif m3:
                    filing_date = f"{m3.group(1)}-{m3.group(2)}-{m3.group(3)}"
                elif m4:
                    filing_date = f"20{m4.group(1)}-06-30"
                else:
                    filing_date = "unknown"

                # Check cache
                cache_dir = config.PROCESSED_DIR / ticker
                cache_file_1 = cache_dir / f"{filing_date}_{filing_type}_guidance.json"
                cache_file_2 = cache_dir / f"{filepath.stem}_guidance.json"
                existing_cache = cache_file_1 if cache_file_1.exists() else (cache_file_2 if cache_file_2.exists() else None)

                if skip_existing and existing_cache:
                    total_from_cache += 1
                    with results_container:
                        st.text(f"  ⏭️ {ticker} - {filing_date} ({filepath.stem}) [ya en caché]")
                    progress_bar.progress(processed / max(total_filings, 1))
                    try:
                        with open(existing_cache, "r", encoding="utf-8") as f:
                            ticker_results.append(json.load(f))
                    except Exception:
                        pass
                    continue

                with results_container:
                    st.text(f"  🧠 Analizando {ticker} - {filing_date} ({filepath.name}) con {model_name}...")

                try:
                    # Parse the filing
                    sections = parse_filing(filepath)

                    # Analyze with LLM or NLP
                    result = analyze_filing(
                        ticker=ticker,
                        filing_date=filing_date,
                        filing_type=filing_type,
                        sections=sections,
                        model=model_name,
                        api_key=groq_key if groq_key else None,
                        force_reanalyze=(not skip_existing),
                    )

                    if "groq" in model_name.lower():
                        import time
                        time.sleep(0.3)  # Gentle rate limit pacing

                    if result:
                        ticker_results.append(result)
                        total_analyzed += 1
                        with results_container:
                            sentiment = result.get("overall_sentiment", "N/A")
                            st.text(f"  ✅ {ticker} - {filing_date}: sentiment={sentiment}")
                    else:
                        total_errors += 1
                        with results_container:
                            st.text(f"  ⚠️ {ticker} - {filing_date}: no se pudo extraer guidance")

                except Exception as e:
                    total_errors += 1
                    with results_container:
                        st.text(f"  ❌ Error: {e}")

                progress_bar.progress(processed / max(total_filings, 1))

            all_results[ticker] = ticker_results

        # Save all results
        all_guidance_path = config.PROCESSED_DIR / "all_guidance.json"
        with open(all_guidance_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        # Automatically update scores.csv and sector rankings
        try:
            from src.scorer import score_all
            score_all(all_guidance=all_results)
            st.info("📊 Tabla de scores y rankings actualizada automáticamente.")
        except Exception as e:
            st.warning(f"No se pudo recalcular scores automáticamente: {e}")

        status_text.markdown("**✅ Proceso completado**")
        if total_analyzed == 0 and total_from_cache > 0:
            st.success(
                f"⚡ Se cargaron **{total_from_cache}** filings ya analizados previamente de la caché (0 pendientes, 0 errores). "
                f"Todos tus datos están listos para consultar en el menú lateral (**📊 Scoring** y **📈 Backtesting**).\n\n"
                f"*(Si deseas forzar a Groq a que re-analice todos los informes desde cero, desmarca la casilla 'Saltar filings ya analizados' y pulsa el botón)*"
            )
        else:
            st.success(
                f"Analizados **{total_analyzed}** filings correctamente ({total_from_cache} desde caché). "
                f"Errores: **{total_errors}**. "
                f"Continúa al paso **📊 Scoring**."
            )

    except Exception as e:
        st.error(f"Error durante el análisis: {e}")
        st.exception(e)

# ── Show existing results ───────────────────────────────────────────
st.divider()
st.subheader("📋 Resultados existentes")

all_guidance_path = config.PROCESSED_DIR / "all_guidance.json"
if all_guidance_path.exists():
    with open(all_guidance_path, "r", encoding="utf-8") as f:
        all_guidance = json.load(f)

    for ticker, results in all_guidance.items():
        if results:
            with st.expander(f"📊 {ticker} - {config.TICKERS.get(ticker, '')} ({len(results)} filings)"):
                for r in results:
                    sentiment = r.get("overall_sentiment", "N/A")
                    summary = r.get("summary", "Sin resumen")
                    filing_date = r.get("filing_date", "N/A")
                    st.markdown(f"**{filing_date}** | Sentiment: `{sentiment}`")
                    st.markdown(f"> {summary[:300]}...")
                    st.markdown("---")
else:
    st.info("No hay resultados de análisis todavía. Ejecuta el análisis arriba.")
