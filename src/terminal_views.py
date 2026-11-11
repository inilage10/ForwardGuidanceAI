"""
ForwardGuidanceAI - Terminal Modular Views & Focus Mode
========================================================
Implementa la nueva interfaz centrada en buscador dinamico, cinta de mercado,
dashboard modular de 3 paneles (50% Noticias / 25% Forward Guidance / 25% Precio)
y Modo Enfoque en pantalla completa con Backtesting integrado.
"""

from __future__ import annotations

import html
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.market_terminal import (
    HOME_CHART_ASSETS,
    format_currency_price,
    get_chart_ohlcv,
    get_company_info,
    get_company_news,
    get_global_corporate_events_cached,
    get_market_tape_quotes,
    get_recent_sec_events_cached,
    get_ticker_snapshot,
    resolve_company_search,
)
from src.pipeline_runner import (
    find_company_sector_peers,
    inspect_ticker_pipeline_state,
    run_company_event_backtest,
    run_company_guidance_pipeline,
    run_sector_cross_backtest,
)
from src.scorer import load_ticker_guidance, score_company
from src.sector_manager import DEFAULT_SECTORS, get_available_sectors
from src.ui_theme import (
    BG_ELEVATED,
    BORDER_HAIRLINE,
    COLOR_DOWN,
    COLOR_NEUTRAL,
    COLOR_UP,
    TEXT_MUTED,
    TEXT_PRIMARY,
    build_backtest_scatter_chart,
    build_quantile_bar_chart,
    build_radar_chart,
    build_score_evolution_chart,
    build_tradingview_chart,
    render_company_quote_banner_html,
    render_pvalue_metric_html,
    render_ticker_tape_html,
)

logger = logging.getLogger(__name__)

QUICK_PICK_TICKERS = [
    ("AAPL", "Apple (NASDAQ)"),
    ("NVDA", "NVIDIA (NASDAQ)"),
    ("MSFT", "Microsoft (NASDAQ)"),
    ("GOOGL", "Alphabet (NASDAQ)"),
    ("AMZN", "Amazon (NASDAQ)"),
    ("JPM", "JPMorgan (NYSE)"),
    ("ITX.MC", "Inditex (Bolsa de Madrid)"),
    ("SAN.MC", "Santander (Bolsa de Madrid)"),
]


# ── Session State Helpers ───────────────────────────────────────────
def init_terminal_state() -> None:
    defaults = {
        "selected_ticker": None,
        "focus_module": None,          # None | "news" | "guidance" | "chart"
        "search_candidates": [],
        "search_error": "",
        "home_asset_symbol": "^GSPC",
        "home_period": "1A",
        "home_chart_type": "candles",
        "mod_c_period": "1A",
        "ai_engine": "nlp_local",
        "groq_api_key": config.GROQ_API_KEY,
        "force_rerun_ticker": None,
        "last_pipeline_logs": {},      # ticker -> list[str]
        "ticker_bt_result": {},        # ticker -> dict
        "sector_bt_result": {},        # sector -> dict
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def select_company(ticker: str) -> None:
    """Activate modular dashboard for the specified company ticker."""
    tk = ticker.upper().strip()
    st.session_state["selected_ticker"] = tk
    st.session_state["focus_module"] = None
    st.session_state["search_candidates"] = []
    st.session_state["search_error"] = ""


def _handle_search_submit(query: str) -> None:
    q = (query or "").strip()
    if not q:
        st.session_state["search_error"] = "Introduce un ticker o nombre de empresa para buscar."
        return

    best, candidates = resolve_company_search(q)
    if not best:
        st.session_state["search_error"] = (
            f"No se encontraron resultados para '{q}'. Prueba con el nombre completo "
            "o el ticker con sufijo de bolsa (ej. AAPL, ITX.MC, SAN.MC, MC.PA, SAP.DE)."
        )
        st.session_state["search_candidates"] = []
        return

    q_up = q.upper()

    def _is_secondary_listing(cand: Dict[str, str]) -> bool:
        t = cand["ticker"].upper()
        raw_ex = cand.get("raw_exchange", "").upper()
        nm_u = cand.get("name", "").upper()
        if t.endswith((".DE", ".F", ".MU", ".DU", ".HM", ".HA", ".BE", ".SG", ".TI", ".NE")):
            return True
        if raw_ex in {"PNK", "OQX", "OBB", "OTC", "BER", "DUS", "MUN", "STU", "HAN", "FRA", "HAM"}:
            return True
        if len(t) == 5 and "." not in t and t.endswith(("F", "Y")):
            return True
        return "/ADR" in nm_u or " ADR" in nm_u or "-" in t

    primary_cands = [c for c in candidates if not _is_secondary_listing(c)]
    has_madrid = any(c["ticker"].endswith(".MC") for c in primary_cands)
    has_us_primary_dual = any(
        "." not in c["ticker"] and (bool(c.get("cik")) or c.get("curated") == "true")
        for c in primary_cands
    )
    has_genuine_dual = has_madrid and has_us_primary_dual

    # Immediate jump when unambiguous, explicit exchange suffix, or only 1 primary listing (ignoring OTC ADRs/XETRA wrappers)
    if "." in q_up or len(candidates) == 1 or len(primary_cands) == 1:
        target = primary_cands[0]["ticker"] if len(primary_cands) == 1 else best["ticker"]
        select_company(target)
        return

    if best["ticker"] == q_up and not has_genuine_dual:
        select_company(best["ticker"])
        return

    if best.get("curated") == "true" and not has_genuine_dual:
        select_company(best["ticker"])
        return

    st.session_state["search_error"] = ""
    st.session_state["search_candidates"] = candidates


# ── Collapsible Sidebar: Terminal Controls & Market Directory ───────
def render_sidebar_controls() -> None:
    with st.sidebar:
        st.markdown("### TERMINAL · CONFIGURACIÓN")

        engine_choice = st.radio(
            "Motor de Análisis Forward Guidance",
            options=["nlp_local", "groq:llama-3.3-70b-versatile", "ollama"],
            format_func=lambda m: {
                "nlp_local": "NLP Financiero Local (Loughran-McDonald · ~2s)",
                "groq:llama-3.3-70b-versatile": "Groq Cloud LLM (Llama 3.3 70B)",
                "ollama": f"Ollama Local ({config.OLLAMA_MODEL})",
            }.get(m, m),
            index=0,
            key="sidebar_engine_radio",
        )
        st.session_state["ai_engine"] = engine_choice

        if "groq" in engine_choice:
            key_val = st.text_input(
                "Groq API Key",
                value=st.session_state.get("groq_api_key", ""),
                type="password",
                help="Clave gratuita de console.groq.com/keys",
            )
            st.session_state["groq_api_key"] = key_val

        st.divider()
        st.markdown("### DIRECTORIO (S&P 500 & IBEX 35)")
        sectors = [s for s in get_available_sectors() if "Big Tech" not in s]
        chosen_sec = st.selectbox("Índice / Sector", options=sectors, key="sidebar_dir_sector")

        peers = DEFAULT_SECTORS.get(chosen_sec, {})
        for tk, info in peers.items():
            label = f"{tk}  ·  {info['name']}  ({info.get('cap', '')})"
            if st.button(label, key=f"side_jump_{chosen_sec}_{tk}", use_container_width=True):
                select_company(tk)
                st.rerun()

        if st.session_state.get("selected_ticker"):
            st.divider()
            if st.button("← Volver a Pantalla de Inicio", key="side_home_btn", use_container_width=True):
                st.session_state["selected_ticker"] = None
                st.session_state["focus_module"] = None
                st.rerun()


# ── Helper Formatters ───────────────────────────────────────────────
def _fmt_price(val: Optional[float], is_yield: bool = False) -> str:
    if val is None:
        return "--"
    if is_yield:
        return f"{val:.2f}%"
    if val >= 1000:
        return f"{val:,.2f}"
    return f"{val:.2f}"


def _delta_color(val: float) -> str:
    if val > 0.0001:
        return COLOR_UP
    if val < -0.0001:
        return COLOR_DOWN
    return COLOR_NEUTRAL


# ── 1. NUEVA PANTALLA DE INICIO ─────────────────────────────────────
def render_home_view() -> None:
    # Masthead strip
    engine_label = (
        "NLP FINANCIERO"
        if "nlp" in st.session_state["ai_engine"]
        else ("GROQ 70B" if "groq" in st.session_state["ai_engine"] else "OLLAMA LOCAL")
    )
    st.markdown(
        f"""
        <div class="term-masthead">
            <div class="term-brand">FORWARDGUIDANCE <span>// EQUITY INTELLIGENCE TERMINAL</span></div>
            <div class="term-status-pill">● SEC EDGAR + YFINANCE &nbsp;|&nbsp; MOTOR: {engine_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1A. Cinta de Mercado (Ticker Tape)
    tape_quotes = get_market_tape_quotes()
    st.markdown(render_ticker_tape_html(tape_quotes), unsafe_allow_html=True)

    # 1B. Buscador Central
    _, search_center, _ = st.columns([0.14, 0.72, 0.14])
    with search_center:
        with st.form("central_search_form", border=False):
            s_col1, s_col2 = st.columns([0.82, 0.18], gap="small")
            with s_col1:
                query_text = st.text_input(
                    "Buscador de empresas o tickers",
                    placeholder="Buscar empresa o ticker global (ej. AAPL, Inditex, ITX.MC, Santander, Iberdrola, LVMH)...",
                    label_visibility="collapsed",
                    key="central_search_input",
                )
            with s_col2:
                submitted = st.form_submit_button("Analizar", type="primary", use_container_width=True)

        if submitted:
            _handle_search_submit(query_text)
            if st.session_state["selected_ticker"]:
                st.rerun()

        if st.session_state.get("search_error"):
            st.warning(st.session_state["search_error"])

        # Candidate disambiguation if user entered partial name or dual-listed equity
        cands = st.session_state.get("search_candidates", [])
        if cands:
            st.caption("Coincidencias encontradas — selecciona mercado y empresa exacta:")
            cand_cols = st.columns(min(len(cands), 4))
            for i, c in enumerate(cands[:4]):
                with cand_cols[i]:
                    ex_tag = f" · {c['exchange']}" if c.get("exchange") else ""
                    btn_lbl = f"{c['ticker']} · {c['name'][:18]}{ex_tag}"
                    if st.button(btn_lbl, key=f"cand_{c['ticker']}_{i}", use_container_width=True):
                        select_company(c["ticker"])
                        st.rerun()

        # Quick-pick row for 1-click access
        q_cols = st.columns(len(QUICK_PICK_TICKERS))
        for idx, (tk, short_nm) in enumerate(QUICK_PICK_TICKERS):
            with q_cols[idx]:
                if st.button(tk, key=f"quick_chip_{tk}", help=short_nm, use_container_width=True):
                    select_company(tk)
                    st.rerun()

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    # 1C. Gráfico Central Inicial (S&P 500 por defecto + Selector de Índices/Activos)
    with st.container(border=True):
        asset_options = [a["symbol"] for a in HOME_CHART_ASSETS]
        asset_labels = {a["symbol"]: f"{a['short']} ({a['symbol']}) · {a['name']}" for a in HOME_CHART_ASSETS}

        ctrl1, ctrl2, ctrl3 = st.columns([0.46, 0.34, 0.20], gap="small")
        with ctrl1:
            selected_asset = st.selectbox(
                "Activo / Índice de Mercado",
                options=asset_options,
                index=asset_options.index(st.session_state["home_asset_symbol"])
                if st.session_state["home_asset_symbol"] in asset_options
                else 0,
                format_func=lambda s: asset_labels.get(s, s),
                label_visibility="collapsed",
                key="home_asset_select",
            )
            st.session_state["home_asset_symbol"] = selected_asset

        with ctrl2:
            period_opts = ["1M", "3M", "6M", "YTD", "1A", "3A", "MAX"]
            selected_period = st.radio(
                "Período",
                options=period_opts,
                index=period_opts.index(st.session_state["home_period"])
                if st.session_state["home_period"] in period_opts
                else 4,
                horizontal=True,
                label_visibility="collapsed",
                key="home_period_radio",
            )
            st.session_state["home_period"] = selected_period

        with ctrl3:
            chart_mode = st.radio(
                "Estilo",
                options=["candles", "area"],
                format_func=lambda x: "Velas" if x == "candles" else "Área",
                index=0 if st.session_state["home_chart_type"] == "candles" else 1,
                horizontal=True,
                label_visibility="collapsed",
                key="home_chart_type_radio",
            )
            st.session_state["home_chart_type"] = chart_mode

        df_idx, stats_idx = get_chart_ohlcv(selected_asset, selected_period)
        asset_meta = next((a for a in HOME_CHART_ASSETS if a["symbol"] == selected_asset), {"short": selected_asset, "name": selected_asset})
        is_yield = selected_asset == "^TNX"

        # Active index live headline bar
        lp = stats_idx.get("last_price")
        dp = stats_idx.get("day_pct", 0.0)
        dc = stats_idx.get("day_change", 0.0)
        d_col = _delta_color(dp)
        arrow = "▲" if dp > 0.005 else ("▼" if dp < -0.005 else "")
        sign = "+" if dp > 0.005 else ""

        st.markdown(
            f"""
            <div style="display:flex;align-items:baseline;justify-content:space-between;padding:4px 4px 2px 4px;border-bottom:1px solid {BORDER_HAIRLINE};margin-bottom:6px;">
                <div>
                    <span style="font-size:1.02rem;font-weight:700;color:{TEXT_PRIMARY};">{html.escape(asset_meta['name'])}</span>
                    <span style="font-family:Consolas,monospace;font-size:0.8rem;color:{TEXT_MUTED};margin-left:8px;">{html.escape(selected_asset)}</span>
                </div>
                <div style="font-family:Consolas,monospace;font-variant-numeric:tabular-nums;">
                    <span style="font-size:1.18rem;font-weight:700;color:{TEXT_PRIMARY};">{_fmt_price(lp, is_yield)}</span>
                    <span style="font-size:0.85rem;font-weight:600;color:{d_col};margin-left:10px;">{arrow} {sign}{dc:,.2f} ({sign}{dp:.2f}%)</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        fig_home = build_tradingview_chart(
            df_idx,
            chart_type=chart_mode,
            height=410,
            show_volume=(selected_asset not in {"^VIX", "^TNX"}),
            overlays=["SMA 50"],
        )
        st.plotly_chart(fig_home, use_container_width=True, config={"displayModeBar": False})

        # Bottom stat strip
        m1, m2, m3, m4, m5 = st.columns(5)
        per_ret = stats_idx.get("period_return", 0.0)
        m1.metric(f"Retorno ({selected_period})", f"{per_ret:+.2f}%")
        m2.metric("Máximo Período", _fmt_price(stats_idx.get("period_high"), is_yield))
        m3.metric("Mínimo Período", _fmt_price(stats_idx.get("period_low"), is_yield))
        m4.metric("Volatilidad Anual", f"{stats_idx.get('volatility_ann', 0.0):.1f}%")
        m5.metric("Max Drawdown", f"{stats_idx.get('max_drawdown', 0.0):.2f}%")


# ── Persistent Top Header inside Company Dashboard ──────────────────
def render_company_header(company_info: Dict[str, str], snapshot: Dict) -> None:
    h_back, h_quote, h_search = st.columns([0.13, 0.57, 0.30], gap="small")

    with h_back:
        if st.button("← Mercado", key="btn_back_home", use_container_width=True):
            st.session_state["selected_ticker"] = None
            st.session_state["focus_module"] = None
            st.rerun()

    with h_quote:
        st.markdown(
            render_company_quote_banner_html(
                company_info["ticker"],
                company_info["name"],
                company_info["sector"],
                snapshot,
                currency=company_info.get("currency", "USD"),
                exchange=company_info.get("exchange", ""),
            ),
            unsafe_allow_html=True,
        )

    with h_search:
        with st.form("topbar_search_form", border=False):
            ts1, ts2 = st.columns([0.72, 0.28], gap="small")
            with ts1:
                new_q = st.text_input(
                    "Cambiar empresa",
                    placeholder="Ticker o empresa (AAPL, Inditex, ITX.MC)...",
                    label_visibility="collapsed",
                    key="topbar_search_input",
                )
            with ts2:
                go_new = st.form_submit_button("Ir", use_container_width=True)

        if go_new and new_q.strip():
            _handle_search_submit(new_q)
            st.rerun()

    # Show disambiguation strip if topbar search found multiple matches
    cands = st.session_state.get("search_candidates", [])
    if cands:
        cand_cols = st.columns(min(len(cands), 5))
        for i, c in enumerate(cands[:5]):
            with cand_cols[i]:
                ex_tag = f" ({c['exchange']})" if c.get("exchange") else ""
                if st.button(f"{c['ticker']} · {c['name'][:14]}{ex_tag}", key=f"top_cand_{c['ticker']}_{i}", use_container_width=True):
                    select_company(c["ticker"])
                    st.rerun()


# ── Module A Content (Noticias & Actualidad) ────────────────────────
def _render_news_cards(news_items: List[Dict], limit: Optional[int] = None) -> None:
    items = news_items[:limit] if limit else news_items
    if not items:
        st.info("No se encontraron titulares recientes para este ticker en las fuentes RSS.")
        return

    for item in items:
        sent = item.get("sentiment", "Neutral")
        sent_cls = (
            "sent-pill-up" if sent == "Positivo"
            else ("sent-pill-down" if sent == "Negativo" else "sent-pill-flat")
        )
        title_esc = html.escape(item["title"])
        src_esc = html.escape(item["source"])
        rel_esc = html.escape(item["published_rel"])
        link_esc = html.escape(item["link"], quote=True)
        desc_esc = html.escape(item.get("summary", ""))

        desc_html = f'<div class="news-desc">{desc_esc}</div>' if desc_esc else ""
        st.markdown(
            f"""
            <div class="news-card">
                <div class="news-meta">
                    <span>{src_esc} &nbsp;·&nbsp; {rel_esc}</span>
                    <span class="{sent_cls}">{sent.upper()}</span>
                </div>
                <div class="news-title"><a href="{link_esc}" target="_blank" rel="noopener noreferrer">{title_esc}</a></div>
                {desc_html}
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_sec_events_table(sec_events: List[Dict]) -> None:
    if not sec_events:
        st.info("No hay eventos regulatorios recientes disponibles en SEC EDGAR para este ticker.")
        return

    for ev in sec_events:
        form = html.escape(ev.get("form_type", ""))
        fdate = html.escape(ev.get("filing_date", ""))
        desc = html.escape(ev.get("description") or ev.get("primary_document", ""))
        items_str = html.escape(str(ev.get("items", "")))
        sec_url = html.escape(ev.get("sec_url", "#"), quote=True)
        items_badge = f" · Items: {items_str}" if items_str else ""
        st.markdown(
            f"""
            <div class="news-card">
                <div class="news-meta">
                    <span>SEC FORM <b>{form}</b>{items_badge}</span>
                    <span>{fdate}</span>
                </div>
                <div class="news-title">
                    <a href="{sec_url}" target="_blank" rel="noopener noreferrer">Documento oficial SEC {form} ({desc}) ↗</a>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_global_events_table(events: List[Dict], ticker: str) -> None:
    if not events:
        st.info(f"No hay calendario de resultados disponible en Yahoo Finance para {ticker}.")
        return

    for ev in events:
        kind = html.escape(ev.get("kind", "Resultados"))
        fdate = html.escape(ev.get("date", ""))
        detail = html.escape(ev.get("detail", ""))
        url = html.escape(ev.get("url", "#"), quote=True)
        st.markdown(
            f"""
            <div class="news-card">
                <div class="news-meta">
                    <span>{kind.upper()}</span>
                    <span>{fdate}</span>
                </div>
                <div class="news-title">
                    <a href="{url}" target="_blank" rel="noopener noreferrer">{html.escape(ticker)} · {detail} ↗</a>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── Module B: Automatic Visible Pipeline Runner & Compact View ──────
def _ensure_and_render_guidance_pipeline(company_info: Dict[str, str], compact: bool = True) -> List[dict]:
    tk = company_info["ticker"]
    state = inspect_ticker_pipeline_state(tk)
    force_rerun = st.session_state.get("force_rerun_ticker") == tk

    if state["needs_auto_run"] or force_rerun:
        engine = st.session_state.get("ai_engine", "nlp_local")
        api_key = st.session_state.get("groq_api_key") or config.GROQ_API_KEY
        logs: List[str] = []

        with st.status(
            f"Ejecutando pipeline Forward Guidance para {tk} ({engine})...",
            expanded=True,
        ) as status_box:
            log_placeholder = st.empty()

            def _ui_log(msg: str) -> None:
                logs.append(msg)
                log_placeholder.code("\n".join(logs[-12:]), language="text")

            run_res = run_company_guidance_pipeline(
                ticker=tk,
                company_name=company_info["name"],
                cik=company_info.get("cik", ""),
                engine=engine,
                api_key=api_key,
                force_reanalyze=force_rerun,
                max_filings=10,
                log_fn=_ui_log,
            )
            st.session_state["last_pipeline_logs"][tk] = logs
            st.session_state["force_rerun_ticker"] = None
            n_recs = len(run_res.get("guidance_records", []))
            if n_recs > 0:
                status_box.update(
                    label=f"Pipeline completado · {n_recs} periodos analizados ({tk})",
                    state="complete",
                    expanded=False,
                )
            else:
                status_box.update(
                    label=f"Sin datos financieros suficientes disponibles para {tk}",
                    state="error",
                    expanded=True,
                )
        return run_res.get("guidance_records", [])

    # Cached records exist -> ensure score CSV sync & show clean pipeline badge
    records = load_ticker_guidance(tk)
    if tk in st.session_state["last_pipeline_logs"] and compact:
        with st.expander(f"✓ Pipeline ejecutado ({len(records)} periodos procesados)", expanded=False):
            st.code("\n".join(st.session_state["last_pipeline_logs"][tk]), language="text")
    return records


def _render_dimension_bars_html(dim_scores: Dict[str, float]) -> str:
    rows = []
    for dim, val in dim_scores.items():
        label = html.escape(dim.replace("_", " ").title())
        v = float(val)
        pct_width = min(50.0, abs(v) * 50.0)
        if v > 0.02:
            bar_color = COLOR_UP
            val_color = COLOR_UP
            bar_left = "50%"
        elif v < -0.02:
            bar_color = COLOR_DOWN
            val_color = COLOR_DOWN
            bar_left = f"{50.0 - pct_width:.1f}%"
        else:
            bar_color = COLOR_NEUTRAL
            val_color = COLOR_NEUTRAL
            bar_left = "49%"
            pct_width = 2.0

        rows.append(
            f'<div class="dim-row">'
            f'<div class="dim-label">{label}</div>'
            f'<div class="dim-track">'
            f'<div style="position:absolute;left:50%;top:0;bottom:0;width:1px;background:#364158;"></div>'
            f'<div style="position:absolute;left:{bar_left};width:{pct_width:.1f}%;top:0;bottom:0;background:{bar_color};border-radius:2px;"></div>'
            f'</div>'
            f'<div class="dim-val" style="color:{val_color};">{v:+.2f}</div>'
            f'</div>'
        )
    return "".join(rows)


def _render_module_b_compact(company_info: Dict[str, str]) -> List[dict]:
    tk = company_info["ticker"]
    hb1, hb2, hb3 = st.columns([0.56, 0.14, 0.30], gap="small")
    with hb1:
        st.markdown('<div class="mod-title">MÓDULO B · FORWARD GUIDANCE AI</div>', unsafe_allow_html=True)
    with hb2:
        if st.button("↻", key=f"rerun_b_{tk}", help="Re-ejecutar análisis SEC en vivo", use_container_width=True):
            st.session_state["force_rerun_ticker"] = tk
            st.rerun()
    with hb3:
        if st.button("⛶ Expandir", key=f"expand_b_{tk}", help="Ampliar y ejecutar Backtesting", use_container_width=True):
            st.session_state["focus_module"] = "guidance"
            st.rerun()

    records = _ensure_and_render_guidance_pipeline(company_info, compact=True)
    if not records:
        st.warning(
            f"No se encontraron datos regulatorios ni estados trimestrales suficientes para **{tk}**."
        )
        return []

    latest = records[-1]
    scored = score_company(latest)
    comp_score = scored["composite_score"]
    sent_score = scored["overall_sentiment"]
    dim_scores = scored["dimension_scores"]
    score_100 = (comp_score + 1.0) * 50.0
    f_date = latest.get("filing_date", "N/A")
    f_type = latest.get("filing_type", "10-Q")
    engine_used = latest.get("analysis_engine", "NLP")
    src_tag = "FUNDAMENTALES" if latest.get("source") == "global_fundamentals" else "FILINGS SEC"

    st.markdown(
        f'<div style="font-size:0.71rem;color:{TEXT_MUTED};margin-bottom:6px;font-family:Consolas,monospace;">'
        f'<span style="color:{COLOR_UP};">●</span> {src_tag} · {len(records)} PERIODOS · ÚLTIMO {html.escape(f_type)} ({html.escape(f_date)}) · {html.escape(engine_used)}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Compact KPI strip with green/red score signal
    k1, k2, k3 = st.columns(3, gap="small")
    sc_color = COLOR_UP if comp_score >= 0.05 else (COLOR_DOWN if comp_score <= -0.05 else TEXT_PRIMARY)
    se_color = COLOR_UP if sent_score >= 0.05 else (COLOR_DOWN if sent_score <= -0.05 else TEXT_PRIMARY)

    k1.markdown(
        f'<div style="background:{BG_ELEVATED};border:1px solid {BORDER_HAIRLINE};border-radius:4px;padding:5px 8px;">'
        f'<div style="font-size:0.64rem;color:{TEXT_MUTED};text-transform:uppercase;">Guidance Score</div>'
        f'<div style="font-family:Consolas,monospace;font-size:1.05rem;font-weight:700;color:{sc_color};">{comp_score:+.3f} <span style="font-size:0.72rem;color:{TEXT_MUTED};">({score_100:.0f}/100)</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    k2.markdown(
        f'<div style="background:{BG_ELEVATED};border:1px solid {BORDER_HAIRLINE};border-radius:4px;padding:5px 8px;">'
        f'<div style="font-size:0.64rem;color:{TEXT_MUTED};text-transform:uppercase;">Sentiment</div>'
        f'<div style="font-family:Consolas,monospace;font-size:1.05rem;font-weight:700;color:{se_color};">{sent_score:+.2f}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    k3.markdown(
        f'<div style="background:{BG_ELEVATED};border:1px solid {BORDER_HAIRLINE};border-radius:4px;padding:5px 8px;">'
        f'<div style="font-size:0.64rem;color:{TEXT_MUTED};text-transform:uppercase;">Histórico</div>'
        f'<div style="font-family:Consolas,monospace;font-size:1.05rem;font-weight:700;color:{TEXT_PRIMARY};">{len(records)} docs</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:5px;'></div>", unsafe_allow_html=True)
    st.markdown(_render_dimension_bars_html(dim_scores), unsafe_allow_html=True)

    summary_txt = latest.get("summary", "")
    if summary_txt:
        st.caption(summary_txt[:150] + ("..." if len(summary_txt) > 150 else ""))

    return records


# ── Dashboard Modular 3 Módulos (50% / 25% / 25%) ───────────────────
def _render_modular_three_panel(company_info: Dict[str, str]) -> None:
    tk = company_info["ticker"]
    col_left, col_right = st.columns([0.5, 0.5], gap="medium")

    # ─── MÓDULO A (50% Izquierda): Noticias y Actualidad ─────────────
    with col_left:
        with st.container(height=705, border=True):
            ha1, ha2 = st.columns([0.74, 0.26], gap="small")
            with ha1:
                st.markdown(f'<div class="mod-title">MÓDULO A · NOTICIAS Y ACTUALIDAD ({tk})</div>', unsafe_allow_html=True)
            with ha2:
                if st.button("⛶ Expandir", key=f"expand_a_{tk}", use_container_width=True):
                    st.session_state["focus_module"] = "news"
                    st.rerun()

            has_cik = bool(company_info.get("cik"))
            sec_tab_label = "Eventos Regulatorios SEC" if has_cik else "Resultados y Eventos"
            tab_news, tab_sec = st.tabs(["Titulares de Mercado", sec_tab_label])
            with tab_news:
                news_items = get_company_news(tk, company_info["name"], limit=14)
                _render_news_cards(news_items)
            with tab_sec:
                if has_cik:
                    sec_events = get_recent_sec_events_cached(company_info.get("cik", ""), limit=12)
                    _render_sec_events_table(sec_events)
                else:
                    global_events = get_global_corporate_events_cached(tk, limit=12)
                    _render_global_events_table(global_events, tk)

    # ─── COLUMNA DERECHA (25% Arriba + 25% Abajo) ───────────────────
    with col_right:
        # MÓDULO B (25% Arriba Derecha): Forward Guidance AI
        with st.container(height=360, border=True):
            guidance_records = _render_module_b_compact(company_info)

        # MÓDULO C (25% Abajo Derecha): Gráfico Interactivo de Precio
        with st.container(height=330, border=True):
            hc1, hc2, hc3 = st.columns([0.44, 0.30, 0.26], gap="small")
            with hc1:
                st.markdown(f'<div class="mod-title">MÓDULO C · PRECIO ({tk})</div>', unsafe_allow_html=True)
            with hc2:
                mod_period = st.selectbox(
                    "Rango",
                    options=["3M", "6M", "1A", "3A"],
                    index=["3M", "6M", "1A", "3A"].index(st.session_state.get("mod_c_period", "1A"))
                    if st.session_state.get("mod_c_period", "1A") in ["3M", "6M", "1A", "3A"]
                    else 2,
                    label_visibility="collapsed",
                    key=f"mod_c_per_{tk}",
                )
                st.session_state["mod_c_period"] = mod_period
            with hc3:
                if st.button("⛶ Expandir", key=f"expand_c_{tk}", use_container_width=True):
                    st.session_state["focus_module"] = "chart"
                    st.rerun()

            df_px, _ = get_chart_ohlcv(tk, mod_period)
            filing_df = None
            if guidance_records:
                f_rows = [
                    {
                        "filing_date": r.get("filing_date"),
                        "filing_type": r.get("filing_type", "10-Q"),
                        "composite_score": score_company(r)["composite_score"],
                    }
                    for r in guidance_records
                ]
                filing_df = pd.DataFrame(f_rows)

            fig_c = build_tradingview_chart(
                df_px,
                chart_type="candles",
                height=245,
                show_volume=True,
                overlays=["SMA 50"],
                filing_events=filing_df,
            )
            st.plotly_chart(fig_c, use_container_width=True, config={"displayModeBar": False})


# ── MODO ENFOQUE (100% Pantalla Completa) ───────────────────────────
def _render_focus_bar(company_info: Dict[str, str], active_focus: str) -> None:
    tk = company_info["ticker"]
    b_exit, b_a, b_b, b_c = st.columns([0.22, 0.24, 0.30, 0.24], gap="small")

    with b_exit:
        if st.button("← Vista Modular (3 Módulos)", key=f"exit_focus_{tk}", use_container_width=True):
            st.session_state["focus_module"] = None
            st.rerun()
    with b_a:
        if st.button(
            "Módulo A · Noticias",
            type="primary" if active_focus == "news" else "secondary",
            key=f"sw_news_{tk}",
            use_container_width=True,
        ):
            st.session_state["focus_module"] = "news"
            st.rerun()
    with b_b:
        if st.button(
            "Módulo B · Guidance y Backtest",
            type="primary" if active_focus == "guidance" else "secondary",
            key=f"sw_guid_{tk}",
            use_container_width=True,
        ):
            st.session_state["focus_module"] = "guidance"
            st.rerun()
    with b_c:
        if st.button(
            "Módulo C · Gráfico Avanzado",
            type="primary" if active_focus == "chart" else "secondary",
            key=f"sw_chart_{tk}",
            use_container_width=True,
        ):
            st.session_state["focus_module"] = "chart"
            st.rerun()


def _render_focus_news(company_info: Dict[str, str]) -> None:
    tk = company_info["ticker"]
    with st.container(border=True):
        has_cik = bool(company_info.get("cik"))
        sub_news = "Yahoo Finance RSS + Google News + SEC EDGAR Filings" if has_cik else "Yahoo Finance RSS + Google News ES/EN + Calendario Corporativo"
        st.markdown(
            f'<div class="mod-header"><span class="mod-title">MODO ENFOQUE · MÓDULO A: NOTICIAS Y ACTUALIDAD ({tk})</span>'
            f'<span class="mod-sub">{sub_news}</span></div>',
            unsafe_allow_html=True,
        )

        f1, f2 = st.columns([0.35, 0.65], gap="medium")
        with f1:
            sent_filter = st.radio(
                "Filtrar por sentimiento de titular",
                options=["Todos", "Positivo", "Negativo", "Neutral"],
                horizontal=True,
                key=f"news_sent_filter_{tk}",
            )
        with f2:
            kw_filter = st.text_input(
                "Filtrar por palabra clave",
                placeholder="Ej. earnings, AI, guidance, margin, China...",
                key=f"news_kw_filter_{tk}",
            )

        news_items = get_company_news(tk, company_info["name"], limit=24)
        filtered = []
        for item in news_items:
            if sent_filter != "Todos" and item.get("sentiment") != sent_filter:
                continue
            if kw_filter.strip() and kw_filter.strip().lower() not in (item["title"] + " " + item.get("summary", "")).lower():
                continue
            filtered.append(item)

        sec_tab_focus_label = "Historial Regulatorio SEC (10-K / 10-Q / 8-K)" if has_cik else "Calendario de Resultados y Dividendos"
        t_news, t_sec = st.tabs([f"Titulares ({len(filtered)})", sec_tab_focus_label])
        with t_news:
            if not filtered:
                st.info("No hay noticias que coincidan con el filtro seleccionado.")
            else:
                c_n1, c_n2 = st.columns(2, gap="medium")
                half = (len(filtered) + 1) // 2
                with c_n1:
                    _render_news_cards(filtered[:half])
                with c_n2:
                    _render_news_cards(filtered[half:])
        with t_sec:
            if has_cik:
                sec_events = get_recent_sec_events_cached(company_info.get("cik", ""), limit=20)
                _render_sec_events_table(sec_events)
            else:
                global_events = get_global_corporate_events_cached(tk, limit=20)
                _render_global_events_table(global_events, tk)


def _render_focus_guidance(company_info: Dict[str, str]) -> None:
    tk = company_info["ticker"]
    has_cik = bool(company_info.get("cik"))
    sub_b = (
        "Análisis dimensional SEC 10-K / 10-Q y validación bursátil aislada"
        if has_cik
        else "Análisis fundamental trimestral global, consenso analistas y backtesting regional"
    )
    with st.container(border=True):
        top_l, top_r = st.columns([0.74, 0.26], gap="small")
        with top_l:
            st.markdown(
                f'<div class="mod-header"><span class="mod-title">MODO ENFOQUE · MÓDULO B: FORWARD GUIDANCE Y BACKTESTING ({tk})</span>'
                f'<span class="mod-sub">{sub_b}</span></div>',
                unsafe_allow_html=True,
            )
        with top_r:
            if st.button("↻ Re-ejecutar Análisis Completo", key=f"focus_rerun_{tk}", use_container_width=True):
                st.session_state["force_rerun_ticker"] = tk
                st.rerun()

        records = _ensure_and_render_guidance_pipeline(company_info, compact=False)
        if not records:
            st.warning(f"No hay reportes SEC ni datos fundamentales trimestrales disponibles para analizar {tk}.")
            return

        tab_detail, tab_backtest = st.tabs([
            "Análisis Detallado, Citas y Evolución",
            "Backtesting Cuantitativo (Vista Aislada)",
        ])

        # ── Tab 1: Deep Guidance Analytics ──────────────────────────
        with tab_detail:
            # Optional dynamic dimension weights
            with st.expander("Ajustar ponderación personalizada de las 7 dimensiones", expanded=False):
                w_cols = st.columns(len(config.GUIDANCE_DIMENSIONS))
                custom_weights = {}
                default_w = {"revenue_growth": 20, "margin_outlook": 20, "capex_plans": 10, "demand_outlook": 15, "earnings_guidance": 20, "innovation_pipeline": 5, "competitive_position": 10}
                for idx, dim in enumerate(config.GUIDANCE_DIMENSIONS):
                    with w_cols[idx]:
                        custom_weights[dim] = st.slider(
                            dim.replace("_", " ").title(),
                            min_value=0,
                            max_value=40,
                            value=default_w.get(dim, 15),
                            step=5,
                            key=f"w_{tk}_{dim}",
                        )

            # Build scores table using custom weights
            rows_eval = []
            for r in records:
                sc = score_company(r)
                d_scores = sc["dimension_scores"]
                tot_w = sum(custom_weights.values())
                comp = (
                    sum(d_scores[d] * (custom_weights[d] / tot_w) for d in config.GUIDANCE_DIMENSIONS)
                    if tot_w > 0
                    else sc["composite_score"]
                )
                row_e = {
                    "filing_date": r.get("filing_date", ""),
                    "filing_type": r.get("filing_type", "10-Q"),
                    "composite_score": comp,
                    "overall_sentiment": sc["overall_sentiment"],
                }
                row_e.update(d_scores)
                rows_eval.append(row_e)

            eval_df = pd.DataFrame(rows_eval).sort_values("filing_date")
            latest_rec = records[-1]
            latest_eval = eval_df.iloc[-1]
            latest_dims = {d: float(latest_eval[d]) for d in config.GUIDANCE_DIMENSIONS}

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Guidance Score Actual", f"{latest_eval['composite_score']:+.3f}")
            m2.metric("Escala 0 - 100", f"{(latest_eval['composite_score'] + 1.0) * 50.0:.1f} / 100")
            m3.metric("Sentiment NLP/LLM", f"{latest_eval['overall_sentiment']:+.2f}")
            is_global_src = latest_rec.get("source") == "global_fundamentals"
            cnt_label = "Periodos Analizados" if is_global_src else "Filings SEC Analizados"
            m4.metric(cnt_label, f"{len(records)} ({latest_rec.get('filing_type', '10-Q')} {latest_rec.get('filing_date', '')})")

            ch1, ch2 = st.columns(2, gap="medium")
            with ch1:
                st.plotly_chart(
                    build_radar_chart(latest_dims, title=f"Perfil 7 Dimensiones ({latest_rec.get('filing_date', '')})"),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
            with ch2:
                st.plotly_chart(
                    build_score_evolution_chart(eval_df, tk),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

            breakdown_title = (
                "#### Desglose Cronológico de Periodos y Fundamentales Globales"
                if is_global_src
                else "#### Desglose Cronológico de Filings y Citas Directas (MD&A)"
            )
            st.markdown(breakdown_title)
            for idx, r in enumerate(reversed(records)):
                f_dt = r.get("filing_date", "N/A")
                f_tp = r.get("filing_type", "10-Q")
                f_se = r.get("overall_sentiment", 0.0)
                with st.expander(f"{f_dt}  |  {f_tp}  |  Sentiment: {f_se:+.2f}", expanded=(idx == 0)):
                    st.markdown(f"**Resumen:** {r.get('summary', 'Sin resumen')}")
                    risks = r.get("key_risks", [])
                    if risks:
                        st.markdown("**Riesgos clave identificados:** " + " · ".join(risks))

                    g_data = r.get("guidance", {})
                    dim_rows = []
                    for d_name, d_info in g_data.items():
                        if isinstance(d_info, dict):
                            dim_rows.append({
                                "Dimensión": d_name.replace("_", " ").title(),
                                "Dirección": d_info.get("direction", "N/A"),
                                "Magnitud": d_info.get("magnitude", "N/A"),
                                "Confianza": d_info.get("confidence", 0.0),
                                "Cita Literal (SEC MD&A)": d_info.get("direct_quote", d_info.get("quote", "")),
                            })
                    if dim_rows:
                        st.dataframe(pd.DataFrame(dim_rows), use_container_width=True, hide_index=True)

        # ── Tab 2: Backtesting Isolated View ────────────────────────
        with tab_backtest:
            st.markdown(
                f"Evalúa si el **Forward Guidance Score** extraído de los filings de **{tk}** "
                "anticipa rentabilidades futuras y exceso de retorno (*Alpha*) sobre el mercado."
            )

            bt_c1, bt_c2, bt_c3 = st.columns([0.34, 0.34, 0.32], gap="small")
            with bt_c1:
                bt_horizon = st.selectbox(
                    "Horizonte de Retorno Post-Filing",
                    options=[30, 90, 180],
                    index=1,
                    format_func=lambda x: f"{x} días ({x//30} {'mes' if x==30 else 'meses'})",
                    key=f"bt_h_{tk}",
                )
            with bt_c2:
                bt_signal = st.selectbox(
                    "Métrica de Señal",
                    options=["composite_score", "overall_sentiment"],
                    format_func=lambda s: "Score Compuesto (7 Dimensiones)" if s == "composite_score" else "Sentiment Global",
                    key=f"bt_sig_{tk}",
                )
            with bt_c3:
                st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
                run_bt_clicked = st.button(f"▶ Ejecutar Backtest de {tk}", type="primary", key=f"run_bt_btn_{tk}", use_container_width=True)

            # Auto-run on initial entry or when user clicks Ejecutar Backtest
            weights_sig = tuple(custom_weights.items())
            cache_key = f"{tk}_{bt_horizon}_{bt_signal}_{weights_sig}"
            if run_bt_clicked or st.session_state["ticker_bt_result"].get("key") != cache_key:
                with st.spinner(f"Calculando retornos post-reporte y Alpha vs índice de referencia para {tk}..."):
                    bt_res = run_company_event_backtest(
                        tk,
                        horizon_days=bt_horizon,
                        score_col=bt_signal,
                        custom_weights=custom_weights,
                    )
                    st.session_state["ticker_bt_result"] = {"key": cache_key, "data": bt_res}

            bt_data = st.session_state["ticker_bt_result"].get("data", {})
            if "error" in bt_data:
                st.warning(bt_data["error"])
            elif bt_data:
                bench_name = bt_data.get("benchmark_name", "S&P 500")
                bm1, bm2, bm3, bm4, bm5, bm6 = st.columns([0.13, 0.16, 0.22, 0.16, 0.17, 0.16])
                bm1.metric("Eventos (Reportes)", f"{bt_data['sample_size']}")
                bm2.metric("Information Coeff (IC)", f"{bt_data['ic']:+.3f}")
                with bm3:
                    st.markdown(
                        render_pvalue_metric_html(
                            bt_data.get("p_value"),
                            sample_size=bt_data.get("sample_size"),
                        ),
                        unsafe_allow_html=True,
                    )
                bm4.metric(f"Retorno Medio ({bt_horizon}d)", f"{bt_data['avg_return']*100:+.2f}%")
                bm5.metric(f"Alpha vs {bench_name}", f"{bt_data['avg_alpha']*100:+.2f}%")
                bm6.metric("Spread Alto vs Bajo", f"{bt_data['high_low_spread']*100:+.2f}%")

                ev_df = bt_data["events_df"]
                ret_col = f"return_{bt_horizon}d"
                alpha_col = f"alpha_{bt_horizon}d"
                y_plot_col = alpha_col if alpha_col in ev_df.columns and ev_df[alpha_col].notna().sum() >= 3 else ret_col

                bc1, bc2 = st.columns(2, gap="medium")
                with bc1:
                    st.plotly_chart(
                        build_backtest_scatter_chart(
                            ev_df,
                            x_col=bt_signal,
                            y_col=y_plot_col,
                            title=f"{tk}: Guidance Score vs Alpha {bench_name} ({bt_horizon}d)",
                        ),
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )
                with bc2:
                    st.plotly_chart(
                        build_tradingview_chart(
                            bt_data["prices_df"],
                            title=f"Señales de Guidance sobre Cotización ({tk})",
                            chart_type="area",
                            height=380,
                            show_volume=False,
                            filing_events=ev_df,
                        ),
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )

                # Event table
                cols_show = [c for c in ["filing_date", "filing_type", "composite_score", "overall_sentiment", "start_price", "return_30d", "return_90d", "return_180d", f"alpha_{bt_horizon}d"] if c in ev_df.columns]
                disp_ev = ev_df[cols_show].sort_values("filing_date", ascending=False).copy()
                for c in ["return_30d", "return_90d", "return_180d", f"alpha_{bt_horizon}d"]:
                    if c in disp_ev.columns:
                        disp_ev[c] = disp_ev[c].apply(lambda x: f"{x*100:+.2f}%" if pd.notna(x) else "N/A")
                ccy_bt = company_info.get("currency", "USD")
                if "start_price" in disp_ev.columns:
                    disp_ev["start_price"] = disp_ev["start_price"].apply(
                        lambda x: format_currency_price(float(x), ccy_bt) if pd.notna(x) else "N/A"
                    )
                for c in ["composite_score", "overall_sentiment"]:
                    if c in disp_ev.columns:
                        disp_ev[c] = disp_ev[c].round(3)
                st.dataframe(disp_ev, use_container_width=True, hide_index=True)

            # ── Section 2 of Backtest Tab: Sector Cross-Sectional Backtest ──
            st.divider()
            sector_name, sector_peers = find_company_sector_peers(tk)
            st.markdown(f"#### Backtest Transversal por Terciles ({sector_name})")

            sb1, sb2 = st.columns([0.55, 0.45], gap="small")
            with sb1:
                run_sec_bt = st.button(
                    f"▶ Ejecutar Backtest Transversal ({sector_name})",
                    key=f"run_sec_bt_{tk}",
                    use_container_width=True,
                )
            with sb2:
                analyze_missing = False
                if sector_peers:
                    analyze_missing = st.button(
                        f"Descargar y Analizar Peers Faltantes ({sector_name})",
                        key=f"sync_peers_bt_{tk}",
                        use_container_width=True,
                    )

            if analyze_missing and sector_peers:
                engine = st.session_state.get("ai_engine", "nlp_local")
                api_key = st.session_state.get("groq_api_key") or config.GROQ_API_KEY
                missing_list = [p for p in sector_peers if inspect_ticker_pipeline_state(p)["needs_auto_run"]]
                if not missing_list:
                    st.success(f"Todas las 10 empresas de {sector_name} ya están analizadas.")
                else:
                    with st.status(f"Analizando {len(missing_list)} empresas restantes de {sector_name}...", expanded=True) as s_box:
                        p_bar = st.progress(0)
                        for idx, peer_tk in enumerate(missing_list, start=1):
                            st.write(f"[{idx}/{len(missing_list)}] Procesando {peer_tk} ({sector_peers[peer_tk]})...")
                            run_company_guidance_pipeline(
                                ticker=peer_tk,
                                company_name=sector_peers[peer_tk],
                                engine=engine,
                                api_key=api_key,
                                max_filings=8,
                            )
                            p_bar.progress(idx / len(missing_list))
                        s_box.update(label=f"Sector {sector_name} sincronizado al 100%", state="complete", expanded=False)
                    run_sec_bt = True

            if run_sec_bt:
                with st.spinner("Ejecutando backtesting transversal Long Top 3 vs Bottom 3..."):
                    sec_res = run_sector_cross_backtest(tk, horizon_days=bt_horizon, score_col=bt_signal)
                    st.session_state["sector_bt_result"] = sec_res

            sec_res = st.session_state.get("sector_bt_result", {})
            if sec_res and "error" in sec_res:
                st.warning(sec_res["error"])
            elif sec_res and "ic" in sec_res:
                ic_i = sec_res["ic"]
                ls_i = sec_res["long_short"]
                st.caption(
                    f"Alcance: **{sec_res['scope_label']}** · Empresas incluidas ({len(sec_res['analyzed_tickers'])}): "
                    f"`{', '.join(sec_res['analyzed_tickers'])}`"
                )
                if sec_res.get("missing_peers"):
                    st.caption(f"Peers pendientes en caché: `{', '.join(sec_res['missing_peers'])}` (puedes descargarlos arriba en 1 clic)")

                sm1, sm2, sm3, sm4 = st.columns([0.23, 0.27, 0.25, 0.25])
                sm1.metric("IC Global (Spearman)", f"{ic_i['ic']:+.3f}")
                with sm2:
                    st.markdown(
                        render_pvalue_metric_html(
                            ic_i.get("p_value"),
                            sample_size=ic_i.get("sample_size"),
                        ),
                        unsafe_allow_html=True,
                    )
                sm3.metric("Long-Short Spread", f"{ls_i['long_short_spread']*100:+.2f}%")
                sm4.metric("Hit Rate", f"{ls_i['hit_rate']*100:.1f}%")

                sc1, sc2 = st.columns(2, gap="medium")
                with sc1:
                    st.plotly_chart(
                        build_quantile_bar_chart(ls_i.get("quantile_returns", {}), bt_horizon),
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )
                with sc2:
                    st.plotly_chart(
                        build_backtest_scatter_chart(
                            sec_res["merged_df"],
                            x_col=bt_signal,
                            y_col=sec_res["target_col"],
                            title=f"{sec_res['scope_label']}: Score vs Alpha ({bt_horizon}d)",
                            color_col="ticker",
                        ),
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )


def _render_focus_chart(company_info: Dict[str, str], guidance_records: List[dict]) -> None:
    tk = company_info["ticker"]
    with st.container(border=True):
        st.markdown(
            f'<div class="mod-header"><span class="mod-title">MODO ENFOQUE · MÓDULO C: GRÁFICO AVANZADO Y TÉCNICO ({tk})</span>'
            f'<span class="mod-sub">TradingView OHLCV + Medias Móviles + Señales SEC EDGAR</span></div>',
            unsafe_allow_html=True,
        )

        cc1, cc2, cc3, cc4 = st.columns([0.34, 0.18, 0.30, 0.18], gap="small")
        with cc1:
            per_list = ["1M", "3M", "6M", "YTD", "1A", "3A", "MAX"]
            sel_per = st.radio(
                "Período",
                options=per_list,
                index=4,
                horizontal=True,
                label_visibility="collapsed",
                key=f"focus_chart_per_{tk}",
            )
        with cc2:
            c_style = st.radio(
                "Estilo",
                options=["candles", "area"],
                format_func=lambda x: "Velas" if x == "candles" else "Área",
                horizontal=True,
                label_visibility="collapsed",
                key=f"focus_chart_style_{tk}",
            )
        with cc3:
            overlays = st.multiselect(
                "Indicadores Técnicos",
                options=["SMA 20", "SMA 50", "SMA 200", "Bollinger"],
                default=["SMA 50"],
                label_visibility="collapsed",
                key=f"focus_chart_ov_{tk}",
            )
        with cc4:
            show_sec = st.checkbox("Marcadores SEC", value=True, key=f"focus_chart_sec_{tk}")

        df_px, stats_px = get_chart_ohlcv(tk, sel_per)
        filing_df = None
        if show_sec and guidance_records:
            filing_df = pd.DataFrame([
                {
                    "filing_date": r.get("filing_date"),
                    "filing_type": r.get("filing_type", "10-Q"),
                    "composite_score": score_company(r)["composite_score"],
                }
                for r in guidance_records
            ])

        fig_full = build_tradingview_chart(
            df_px,
            chart_type=c_style,
            height=520,
            show_volume=True,
            overlays=overlays,
            filing_events=filing_df,
        )
        st.plotly_chart(fig_full, use_container_width=True)

        cm1, cm2, cm3, cm4, cm5, cm6 = st.columns(6)
        cm1.metric("Último Cierre", _fmt_price(stats_px.get("last_price")))
        cm2.metric(f"Retorno ({sel_per})", f"{stats_px.get('period_return', 0.0):+.2f}%")
        cm3.metric("Máximo Período", _fmt_price(stats_px.get("period_high")))
        cm4.metric("Mínimo Período", _fmt_price(stats_px.get("period_low")))
        cm5.metric("Volatilidad Anual", f"{stats_px.get('volatility_ann', 0.0):.1f}%")
        cm6.metric("Max Drawdown", f"{stats_px.get('max_drawdown', 0.0):.2f}%")


def render_company_dashboard() -> None:
    tk = st.session_state["selected_ticker"]
    company_info = get_company_info(tk)
    snapshot = get_ticker_snapshot(tk)

    # Persistent header bar with live quote & search switcher
    render_company_header(company_info, snapshot)
    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

    focus = st.session_state.get("focus_module")
    if focus is None:
        _render_modular_three_panel(company_info)
    else:
        _render_focus_bar(company_info, focus)
        st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
        if focus == "news":
            _render_focus_news(company_info)
        elif focus == "guidance":
            _render_focus_guidance(company_info)
        elif focus == "chart":
            records = load_ticker_guidance(tk)
            _render_focus_chart(company_info, records)
