"""
ForwardGuidanceAI — Institutional Equity & Forward Guidance Terminal.

Entrypoint unificado que reemplaza el flujo secuencial por pasos por una
plataforma centrada en buscador dinámico, cinta de mercado, gráfico inicial
del S&P 500, dashboard modular de 3 paneles (Noticias 50%, Forward Guidance 25%,
Gráfico de Precio 25%) y Modo Enfoque a pantalla completa con backtesting integrado.
"""
from __future__ import annotations

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="ForwardGuidanceAI // Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from src.terminal_views import (
    init_terminal_state,
    render_company_dashboard,
    render_home_view,
    render_sidebar_controls,
)
from src.ui_theme import inject_terminal_css


def main() -> None:
    """Initialize terminal state, apply dark institutional CSS, and route views."""
    init_terminal_state()
    inject_terminal_css()
    render_sidebar_controls()

    selected = st.session_state.get("selected_ticker")
    if selected:
        render_company_dashboard()
    else:
        render_home_view()


if __name__ == "__main__":
    main()
