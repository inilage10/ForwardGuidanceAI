"""
ForwardGuidanceAI - Institutional Dark UI Theme & Chart Components
===================================================================
Estetica Bloomberg / TradingView Dark Mode:
- Fondos grises carbon y negro obisidiana (#0b0e14, #131722, #1a1f2e)
- Colores verde (#26a69a / #00c853) y rojo (#ef5350) reservados estrictamente
  para retornos de mercado, senales de guidance y graficos financieros.
"""

from __future__ import annotations

import html
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Palette Constants ───────────────────────────────────────────────
BG_CANVAS = "#0b0e14"
BG_PANEL = "#131722"
BG_ELEVATED = "#1a1f2e"
BORDER_HAIRLINE = "#222938"
BORDER_STRONG = "#2e374b"
TEXT_PRIMARY = "#e6edf3"
TEXT_MUTED = "#848e9c"
ACCENT_BLUE = "#2962ff"
ACCENT_SLATE = "#94a3b8"
COLOR_UP = "#26a69a"      # TradingView teal-green
COLOR_DOWN = "#ef5350"    # TradingView red
COLOR_NEUTRAL = "#787b86"


def inject_terminal_css() -> None:
    """Inject mandatory institutional dark mode CSS across the Streamlit app."""
    st.markdown(
        f"""
        <style>
        /* Force Obsidian/Charcoal Dark Canvas */
        html, body, [data-testid="stAppViewContainer"], .stApp {{
            background-color: {BG_CANVAS} !important;
            color: {TEXT_PRIMARY} !important;
            font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
        }}

        [data-testid="stHeader"] {{
            background-color: rgba(11, 14, 20, 0.92) !important;
            border-bottom: 1px solid {BORDER_HAIRLINE};
        }}

        [data-testid="stSidebar"] {{
            background-color: #0e121a !important;
            border-right: 1px solid {BORDER_HAIRLINE} !important;
        }}

        .block-container {{
            padding-top: 1.15rem !important;
            padding-bottom: 1.5rem !important;
            padding-left: 1.6rem !important;
            padding-right: 1.6rem !important;
            max-width: 100% !important;
        }}

        /* Modular Bordered Containers (Bloomberg / TradingView Panels) */
        [data-testid="stVerticalBlockBorderWrapper"] {{
            background-color: {BG_PANEL} !important;
            border: 1px solid {BORDER_HAIRLINE} !important;
            border-radius: 6px !important;
            padding: 0.75rem 0.9rem !important;
        }}

        /* Inputs & Selectboxes */
        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div {{
            background-color: {BG_ELEVATED} !important;
            border: 1px solid {BORDER_STRONG} !important;
            border-radius: 5px !important;
            color: {TEXT_PRIMARY} !important;
        }}
        input {{
            color: {TEXT_PRIMARY} !important;
        }}

        /* Buttons: clean charcoal institutional controls */
        button[kind="secondary"] {{
            background-color: {BG_ELEVATED} !important;
            border: 1px solid {BORDER_STRONG} !important;
            color: {TEXT_PRIMARY} !important;
            border-radius: 4px !important;
            font-size: 0.82rem !important;
            font-weight: 500 !important;
            padding: 0.3rem 0.7rem !important;
            transition: border-color 0.15s ease, background-color 0.15s ease;
        }}
        button[kind="secondary"]:hover {{
            background-color: #232a3b !important;
            border-color: #4b5563 !important;
            color: #ffffff !important;
        }}
        button[kind="primary"] {{
            background-color: {ACCENT_BLUE} !important;
            border: 1px solid {ACCENT_BLUE} !important;
            color: #ffffff !important;
            border-radius: 4px !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
        }}
        button[kind="primary"]:hover {{
            background-color: #1e53e5 !important;
        }}

        /* Tabs */
        button[data-baseweb="tab"] {{
            color: {TEXT_MUTED} !important;
            font-weight: 600 !important;
            font-size: 0.84rem !important;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            color: {TEXT_PRIMARY} !important;
            border-bottom-color: {ACCENT_BLUE} !important;
        }}

        /* Metrics */
        [data-testid="stMetric"] {{
            background-color: {BG_ELEVATED} !important;
            border: 1px solid {BORDER_HAIRLINE} !important;
            border-radius: 5px !important;
            padding: 0.55rem 0.75rem !important;
        }}
        [data-testid="stMetricLabel"] {{
            color: {TEXT_MUTED} !important;
            font-size: 0.74rem !important;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        [data-testid="stMetricValue"] {{
            color: {TEXT_PRIMARY} !important;
            font-family: "JetBrains Mono", "Consolas", monospace !important;
            font-variant-numeric: tabular-nums;
            font-size: 1.25rem !important;
        }}

        /* Dataframes & Tables */
        [data-testid="stDataFrame"] {{
            border: 1px solid {BORDER_HAIRLINE};
            border-radius: 5px;
        }}

        /* Status / Expander */
        [data-testid="stExpander"] {{
            background-color: {BG_ELEVATED} !important;
            border: 1px solid {BORDER_HAIRLINE} !important;
            border-radius: 5px !important;
        }}

        /* Custom Terminal Utility Classes */
        .term-masthead {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.35rem 0.1rem 0.55rem 0.1rem;
            border-bottom: 1px solid {BORDER_HAIRLINE};
            margin-bottom: 0.6rem;
        }}
        .term-brand {{
            font-family: "JetBrains Mono", "Consolas", monospace;
            font-weight: 700;
            font-size: 0.92rem;
            letter-spacing: 0.08em;
            color: {TEXT_PRIMARY};
        }}
        .term-brand span {{
            color: {TEXT_MUTED};
            font-weight: 400;
        }}
        .term-status-pill {{
            font-family: "JetBrains Mono", "Consolas", monospace;
            font-size: 0.72rem;
            color: {TEXT_MUTED};
            background: {BG_PANEL};
            border: 1px solid {BORDER_HAIRLINE};
            padding: 3px 9px;
            border-radius: 4px;
        }}

        /* Ticker Tape Ribbon */
        .ticker-tape-wrap {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(128px, 1fr));
            gap: 6px;
            background-color: {BG_PANEL};
            border: 1px solid {BORDER_HAIRLINE};
            border-radius: 6px;
            padding: 8px 10px;
            margin-bottom: 0.9rem;
        }}
        .tape-item {{
            display: flex;
            flex-direction: column;
            padding: 5px 8px;
            border-radius: 4px;
            background-color: #0e121b;
            border: 1px solid #1b2232;
        }}
        .tape-sym {{
            font-size: 0.68rem;
            font-weight: 700;
            color: {TEXT_MUTED};
            letter-spacing: 0.05em;
            text-transform: uppercase;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .tape-row {{
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            gap: 6px;
            margin-top: 2px;
            font-family: "JetBrains Mono", "Consolas", monospace;
            font-variant-numeric: tabular-nums;
        }}
        .tape-price {{
            font-size: 0.84rem;
            font-weight: 600;
            color: {TEXT_PRIMARY};
        }}
        .tape-chg-up {{
            font-size: 0.74rem;
            font-weight: 600;
            color: {COLOR_UP};
        }}
        .tape-chg-down {{
            font-size: 0.74rem;
            font-weight: 600;
            color: {COLOR_DOWN};
        }}
        .tape-chg-flat {{
            font-size: 0.74rem;
            font-weight: 600;
            color: {COLOR_NEUTRAL};
        }}

        /* Module Header Strip */
        .mod-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid {BORDER_HAIRLINE};
            padding-bottom: 6px;
            margin-bottom: 8px;
        }}
        .mod-title {{
            font-family: "JetBrains Mono", "Consolas", monospace;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            color: {TEXT_PRIMARY};
            text-transform: uppercase;
        }}
        .mod-sub {{
            font-size: 0.72rem;
            color: {TEXT_MUTED};
        }}

        /* News Card */
        .news-card {{
            background-color: #0f141f;
            border: 1px solid #1d2434;
            border-radius: 5px;
            padding: 9px 11px;
            margin-bottom: 8px;
            transition: border-color 0.15s ease;
        }}
        .news-card:hover {{
            border-color: #364158;
        }}
        .news-meta {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.70rem;
            color: {TEXT_MUTED};
            margin-bottom: 4px;
            font-family: "JetBrains Mono", "Consolas", monospace;
        }}
        .news-title a {{
            color: {TEXT_PRIMARY} !important;
            text-decoration: none;
            font-weight: 600;
            font-size: 0.86rem;
            line-height: 1.32;
        }}
        .news-title a:hover {{
            color: #60a5fa !important;
            text-decoration: underline;
        }}
        .news-desc {{
            font-size: 0.76rem;
            color: #9ca3af;
            margin-top: 4px;
            line-height: 1.38;
        }}
        .sent-pill-up {{
            color: {COLOR_UP};
            border: 1px solid rgba(38, 166, 154, 0.35);
            background: rgba(38, 166, 154, 0.08);
            padding: 1px 6px;
            border-radius: 3px;
            font-size: 0.66rem;
            font-weight: 600;
        }}
        .sent-pill-down {{
            color: {COLOR_DOWN};
            border: 1px solid rgba(239, 83, 80, 0.35);
            background: rgba(239, 83, 80, 0.08);
            padding: 1px 6px;
            border-radius: 3px;
            font-size: 0.66rem;
            font-weight: 600;
        }}
        .sent-pill-flat {{
            color: {COLOR_NEUTRAL};
            border: 1px solid rgba(120, 123, 134, 0.3);
            background: rgba(120, 123, 134, 0.08);
            padding: 1px 6px;
            border-radius: 3px;
            font-size: 0.66rem;
            font-weight: 600;
        }}

        /* Guidance Dimension Bar */
        .dim-row {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            padding: 3px 0;
            font-size: 0.76rem;
            border-bottom: 1px solid rgba(34, 41, 56, 0.5);
        }}
        .dim-label {{
            color: #cbd5e1;
            width: 38%;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .dim-track {{
            flex: 1;
            height: 6px;
            background: #1b2232;
            border-radius: 3px;
            overflow: hidden;
            position: relative;
        }}
        .dim-val {{
            width: 58px;
            text-align: right;
            font-family: "JetBrains Mono", "Consolas", monospace;
            font-variant-numeric: tabular-nums;
            font-weight: 600;
            font-size: 0.74rem;
        }}

        /* Custom thin scrollbars */
        ::-webkit-scrollbar {{
            width: 6px;
            height: 6px;
        }}
        ::-webkit-scrollbar-track {{
            background: {BG_CANVAS};
        }}
        ::-webkit-scrollbar-thumb {{
            background: #2a3346;
            border-radius: 3px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: #3d4963;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_ticker_tape_html(quotes: List[Dict]) -> str:
    """Render institutional monochrome dark ticker tape with green/red returns only."""
    cards = []
    for q in quotes:
        short = html.escape(q["short"])
        price = q.get("price")
        pct = q.get("pct", 0.0)
        is_yield = q.get("is_yield", False)

        if price is None:
            price_str = "--"
            pct_str = "0.00%"
            cls = "tape-chg-flat"
        else:
            if is_yield:
                price_str = f"{price:.2f}%"
            elif price >= 1000:
                price_str = f"{price:,.2f}"
            else:
                price_str = f"{price:.2f}"

            if pct > 0.005:
                pct_str = f"▲ +{pct:.2f}%"
                cls = "tape-chg-up"
            elif pct < -0.005:
                pct_str = f"▼ {pct:.2f}%"
                cls = "tape-chg-down"
            else:
                pct_str = "0.00%"
                cls = "tape-chg-flat"

        cards.append(
            f'<div class="tape-item">'
            f'<div class="tape-sym">{short}</div>'
            f'<div class="tape-row">'
            f'<span class="tape-price">{price_str}</span>'
            f'<span class="{cls}">{pct_str}</span>'
            f'</div>'
            f'</div>'
        )

    return '<div class="ticker-tape-wrap">' + "".join(cards) + "</div>"


def render_company_quote_banner_html(
    ticker: str,
    name: str,
    sector: str,
    snapshot: Dict,
) -> str:
    """Render sleek Bloomberg-style header strip for the active searched company."""
    price = snapshot.get("price")
    pct = snapshot.get("pct", 0.0)
    delta = snapshot.get("delta", 0.0)

    if price is None:
        price_html = '<span style="color:#848e9c;">--</span>'
        delta_html = ""
    else:
        price_str = f"${price:,.2f}"
        if pct > 0.005:
            color = COLOR_UP
            arrow = "▲"
            sign = "+"
        elif pct < -0.005:
            color = COLOR_DOWN
            arrow = "▼"
            sign = ""
        else:
            color = COLOR_NEUTRAL
            arrow = ""
            sign = ""
        price_html = f'<span style="font-family:Consolas,monospace;font-size:1.3rem;font-weight:700;color:{TEXT_PRIMARY};">{price_str}</span>'
        delta_html = (
            f'<span style="font-family:Consolas,monospace;font-size:0.88rem;font-weight:600;color:{color};margin-left:8px;">'
            f'{arrow} {sign}{delta:,.2f} ({sign}{pct:.2f}%)</span>'
        )

    return (
        f'<div style="display:flex;align-items:center;flex-wrap:wrap;gap:12px;padding:4px 2px;">'
        f'<span style="font-family:Consolas,monospace;font-size:1.15rem;font-weight:800;'
        f'background:#1a1f2e;border:1px solid #2e374b;padding:3px 9px;border-radius:4px;color:{TEXT_PRIMARY};">{html.escape(ticker)}</span>'
        f'<span style="font-size:1.08rem;font-weight:600;color:{TEXT_PRIMARY};">{html.escape(name)}</span>'
        f'<span style="font-size:0.74rem;color:{TEXT_MUTED};background:#131722;border:1px solid {BORDER_HAIRLINE};'
        f'padding:2px 8px;border-radius:4px;">{html.escape(sector)}</span>'
        f'<div style="margin-left:auto;display:flex;align-items:baseline;">{price_html}{delta_html}</div>'
        f'</div>'
    )


def render_pvalue_metric_html(
    p_value: Optional[float],
    sample_size: Optional[int] = None,
    alpha: float = 0.05,
) -> str:
    """Render dark institutional KPI card highlighting statistical significance above small p-value."""
    insufficient = sample_size is not None and sample_size < 4
    valid_p = p_value is not None and not np.isnan(p_value)

    if insufficient or not valid_p:
        is_sig = False
        p_str = f"p-value = {float(p_value):.4f} (muestra < 4)" if valid_p else "p-value = N/D"
    else:
        p_num = float(p_value)
        is_sig = p_num < alpha
        p_val_fmt = "< 0.0001" if p_num < 0.0001 else f"= {p_num:.4f}"
        cmp_hint = f"< {alpha:.2f}" if is_sig else f"≥ {alpha:.2f}"
        p_str = f"p-value {p_val_fmt} ({cmp_hint})"

    status_text = "Significativo" if is_sig else "No significativo"
    status_color = TEXT_PRIMARY if is_sig else ACCENT_SLATE
    border_accent = f"border-left:3px solid {ACCENT_BLUE};" if is_sig else ""
    dot_html = (
        f'<span style="color:{ACCENT_BLUE};font-size:0.85rem;margin-right:5px;">●</span>'
        if is_sig
        else ""
    )

    return (
        f'<div style="background-color:{BG_ELEVATED};border:1px solid {BORDER_HAIRLINE};{border_accent}'
        f'border-radius:5px;padding:0.50rem 0.75rem;display:flex;flex-direction:column;justify-content:center;">'
        f'<div style="color:{TEXT_MUTED};font-size:0.74rem;text-transform:uppercase;letter-spacing:0.04em;'
        f'line-height:1.25;margin-bottom:2px;">SIGNIFICATIVIDAD</div>'
        f'<div style="color:{status_color};font-size:1.18rem;font-weight:700;line-height:1.25;">'
        f'{dot_html}{status_text}</div>'
        f'<div style="color:{TEXT_MUTED};font-family:\'JetBrains Mono\',\'Consolas\',monospace;'
        f'font-variant-numeric:tabular-nums;font-size:0.74rem;margin-top:2px;">{html.escape(p_str)}</div>'
        f'</div>'
    )


def build_tradingview_chart(
    df: pd.DataFrame,
    title: str = "",
    chart_type: str = "candles",
    height: int = 440,
    show_volume: bool = True,
    overlays: Optional[List[str]] = None,
    filing_events: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """Create a crisp dark TradingView-style OHLCV/Area chart with optional SEC filing markers."""
    overlays = overlays or []
    has_vol = show_volume and "Volume" in df.columns and df["Volume"].fillna(0).sum() > 0

    if has_vol:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.78, 0.22],
        )
    else:
        fig = make_subplots(rows=1, cols=1)

    if df.empty:
        fig.update_layout(
            height=height,
            paper_bgcolor=BG_PANEL,
            plot_bgcolor="#0e121b",
            font=dict(color=TEXT_MUTED),
            annotations=[dict(text="Sin datos disponibles", showarrow=False, font=dict(color=TEXT_MUTED, size=13))],
        )
        return fig

    has_ohlc = all(c in df.columns for c in ["Open", "High", "Low", "Close"])
    closes = df["Close"].astype(float)
    is_net_up = float(closes.iloc[-1]) >= float(closes.iloc[0]) if len(closes) >= 2 else True
    line_color = COLOR_UP if is_net_up else COLOR_DOWN
    fill_rgba = "rgba(38, 166, 154, 0.14)" if is_net_up else "rgba(239, 83, 80, 0.14)"

    if chart_type == "candles" and has_ohlc:
        fig.add_trace(
            go.Candlestick(
                x=df["Date"],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Precio",
                increasing_line_color=COLOR_UP,
                increasing_fillcolor=COLOR_UP,
                decreasing_line_color=COLOR_DOWN,
                decreasing_fillcolor=COLOR_DOWN,
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["Close"],
                mode="lines",
                name="Cierre",
                line=dict(color=line_color, width=2),
                fill="tozeroy",
                fillcolor=fill_rgba,
            ),
            row=1,
            col=1,
        )

    # Technical Overlays
    if "SMA 20" in overlays and len(df) >= 20:
        sma20 = closes.rolling(20).mean()
        fig.add_trace(
            go.Scatter(x=df["Date"], y=sma20, mode="lines", name="SMA 20", line=dict(color="#60a5fa", width=1.2)),
            row=1,
            col=1,
        )
    if "SMA 50" in overlays and len(df) >= 50:
        sma50 = closes.rolling(50).mean()
        fig.add_trace(
            go.Scatter(x=df["Date"], y=sma50, mode="lines", name="SMA 50", line=dict(color="#eab308", width=1.2)),
            row=1,
            col=1,
        )
    if "SMA 200" in overlays and len(df) >= 200:
        sma200 = closes.rolling(200).mean()
        fig.add_trace(
            go.Scatter(x=df["Date"], y=sma200, mode="lines", name="SMA 200", line=dict(color="#cbd5e1", width=1.2, dash="dot")),
            row=1,
            col=1,
        )
    if "Bollinger" in overlays and len(df) >= 20:
        mid = closes.rolling(20).mean()
        std = closes.rolling(20).std()
        fig.add_trace(
            go.Scatter(x=df["Date"], y=mid + 2 * std, mode="lines", name="BB Sup", line=dict(color="#475569", width=1, dash="dash")),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=df["Date"], y=mid - 2 * std, mode="lines", name="BB Inf", line=dict(color="#475569", width=1, dash="dash")),
            row=1,
            col=1,
        )

    # Overlay SEC Filing Guidance Signals on Price Chart
    if filing_events is not None and not filing_events.empty:
        df_dates = df[["Date", "Close"]].copy()
        df_dates["Date"] = pd.to_datetime(df_dates["Date"])
        ev_copy = filing_events.copy()
        ev_copy["filing_date_dt"] = pd.to_datetime(ev_copy["filing_date"], errors="coerce")
        ev_copy = ev_copy.dropna(subset=["filing_date_dt"])
        ev_copy = ev_copy[
            (ev_copy["filing_date_dt"] >= df_dates["Date"].min())
            & (ev_copy["filing_date_dt"] <= df_dates["Date"].max())
        ]

        if not ev_copy.empty:
            marker_x, marker_y, marker_colors, marker_syms, marker_text = [], [], [], [], []
            for _, row in ev_copy.iterrows():
                f_dt = row["filing_date_dt"]
                sub = df_dates[df_dates["Date"] >= f_dt]
                if sub.empty:
                    continue
                px_val = float(sub.iloc[0]["Close"])
                sc = float(row.get("composite_score", 0.0))
                f_type = row.get("filing_type", "10-Q")
                marker_x.append(sub.iloc[0]["Date"])
                marker_y.append(px_val)
                if sc >= 0.08:
                    marker_colors.append(COLOR_UP)
                    marker_syms.append("triangle-up")
                elif sc <= -0.08:
                    marker_colors.append(COLOR_DOWN)
                    marker_syms.append("triangle-down")
                else:
                    marker_colors.append("#94a3b8")
                    marker_syms.append("circle")
                marker_text.append(f"SEC {f_type} ({f_dt.strftime('%Y-%m-%d')})<br>Guidance Score: {sc:+.3f}")

            if marker_x:
                fig.add_trace(
                    go.Scatter(
                        x=marker_x,
                        y=marker_y,
                        mode="markers",
                        name="Filings SEC",
                        marker=dict(size=10, color=marker_colors, symbol=marker_syms, line=dict(width=1, color="#0b0e14")),
                        text=marker_text,
                        hoverinfo="text",
                    ),
                    row=1,
                    col=1,
                )

    # Volume Subplot
    if has_vol:
        if has_ohlc:
            vol_colors = [
                "rgba(38, 166, 154, 0.45)" if c >= o else "rgba(239, 83, 80, 0.45)"
                for o, c in zip(df["Open"], df["Close"])
            ]
        else:
            vol_colors = "rgba(148, 163, 184, 0.35)"

        fig.add_trace(
            go.Bar(
                x=df["Date"],
                y=df["Volume"],
                name="Volumen",
                marker_color=vol_colors,
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(showgrid=False, tickfont=dict(size=9, color=TEXT_MUTED), row=2, col=1)

    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=TEXT_PRIMARY), x=0.01) if title else None,
        height=height,
        margin=dict(l=10, r=14, t=34 if title else 12, b=10),
        paper_bgcolor=BG_PANEL,
        plot_bgcolor="#0e121b",
        font=dict(color=TEXT_MUTED, size=11),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
            font=dict(size=10, color=TEXT_MUTED),
            bgcolor="rgba(0,0,0,0)",
        ),
    )

    fig.update_xaxes(
        gridcolor="#1a2130",
        zerolinecolor="#1a2130",
        showspikes=True,
        spikecolor="#475569",
        spikethickness=1,
        tickfont=dict(color=TEXT_MUTED, size=10),
    )
    fig.update_yaxes(
        gridcolor="#1a2130",
        zerolinecolor="#1a2130",
        side="right",
        showspikes=True,
        spikecolor="#475569",
        spikethickness=1,
        tickfont=dict(color=TEXT_MUTED, size=10),
        row=1,
        col=1,
    )
    return fig


def build_radar_chart(dim_scores: Dict[str, float], title: str = "Perfil de 7 Dimensiones") -> go.Figure:
    """Create institutional dark radar chart for the 7 guidance dimensions."""
    labels = [k.replace("_", " ").title() for k in dim_scores.keys()]
    vals = [float(v) for v in dim_scores.values()]
    if labels:
        labels_closed = labels + [labels[0]]
        vals_closed = vals + [vals[0]]
    else:
        labels_closed, vals_closed = [], []

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=vals_closed,
            theta=labels_closed,
            fill="toself",
            fillcolor="rgba(41, 98, 255, 0.22)",
            line=dict(color="#60a5fa", width=2),
            marker=dict(size=5, color="#93c5fd"),
            name="Score",
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=TEXT_PRIMARY)),
        height=350,
        margin=dict(l=40, r=40, t=42, b=24),
        paper_bgcolor=BG_PANEL,
        plot_bgcolor=BG_PANEL,
        font=dict(color=TEXT_MUTED, size=10),
        polar=dict(
            bgcolor="#0e121b",
            radialaxis=dict(
                range=[-1, 1],
                gridcolor="#1e2536",
                linecolor="#2a3245",
                tickfont=dict(size=9, color=TEXT_MUTED),
            ),
            angularaxis=dict(
                gridcolor="#1e2536",
                linecolor="#2a3245",
                tickfont=dict(size=10, color=TEXT_PRIMARY),
            ),
        ),
        showlegend=False,
    )
    return fig


def build_score_evolution_chart(scores_df: pd.DataFrame, ticker: str) -> go.Figure:
    """Plot Forward Guidance composite score and sentiment evolution over SEC filings."""
    fig = go.Figure()
    if scores_df.empty:
        fig.update_layout(height=330, paper_bgcolor=BG_PANEL, plot_bgcolor="#0e121b")
        return fig

    df = scores_df.copy()
    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    df = df.dropna(subset=["filing_date"]).sort_values("filing_date")

    fig.add_hline(y=0, line_dash="dot", line_color="#475569", line_width=1)

    fig.add_trace(
        go.Scatter(
            x=df["filing_date"],
            y=df["composite_score"],
            mode="lines+markers",
            name="Guidance Score",
            line=dict(color="#60a5fa", width=2.5),
            marker=dict(
                size=8,
                color=[COLOR_UP if v >= 0 else COLOR_DOWN for v in df["composite_score"]],
                line=dict(width=1, color=BG_CANVAS),
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["filing_date"],
            y=df["overall_sentiment"],
            mode="lines+markers",
            name="Sentiment NLP/LLM",
            line=dict(color="#94a3b8", width=1.8, dash="dash"),
            marker=dict(size=5, color="#94a3b8"),
        )
    )

    fig.update_layout(
        title=dict(text=f"Evolución Histórica de Guidance ({ticker})", font=dict(size=13, color=TEXT_PRIMARY)),
        height=350,
        margin=dict(l=15, r=15, t=42, b=20),
        paper_bgcolor=BG_PANEL,
        plot_bgcolor="#0e121b",
        font=dict(color=TEXT_MUTED, size=10),
        hovermode="x unified",
        yaxis=dict(range=[-1.05, 1.05], gridcolor="#1a2130", title="Score [-1, +1]"),
        xaxis=dict(gridcolor="#1a2130"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def build_backtest_scatter_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    color_col: Optional[str] = None,
) -> go.Figure:
    """Create dark scatter plot with numpy OLS regression line (zero statsmodels dependency)."""
    fig = go.Figure()
    valid = df.dropna(subset=[x_col, y_col]).copy()
    if valid.empty:
        fig.update_layout(height=360, paper_bgcolor=BG_PANEL, plot_bgcolor="#0e121b")
        return fig

    fig.add_hline(y=0, line_dash="dot", line_color="#334155", line_width=1)
    fig.add_vline(x=0, line_dash="dot", line_color="#334155", line_width=1)

    if color_col and color_col in valid.columns:
        for grp_name, grp in valid.groupby(color_col):
            fig.add_trace(
                go.Scatter(
                    x=grp[x_col],
                    y=grp[y_col] * 100.0,
                    mode="markers",
                    name=str(grp_name),
                    marker=dict(size=8, opacity=0.88),
                    text=[
                        f"{r.get('ticker', '')} | {r.get('filing_date', '')}<br>Score: {r[x_col]:+.3f}<br>Retorno: {r[y_col]*100:+.2f}%"
                        for _, r in grp.iterrows()
                    ],
                    hoverinfo="text",
                )
            )
    else:
        colors = [COLOR_UP if v >= 0 else COLOR_DOWN for v in valid[y_col]]
        fig.add_trace(
            go.Scatter(
                x=valid[x_col],
                y=valid[y_col] * 100.0,
                mode="markers",
                name="Filings",
                marker=dict(size=10, color=colors, line=dict(width=1, color=BG_CANVAS)),
                text=[
                    f"{r.get('filing_date', '')} ({r.get('filing_type', '')})<br>Score: {r[x_col]:+.3f}<br>Retorno: {r[y_col]*100:+.2f}%"
                    for _, r in valid.iterrows()
                ],
                hoverinfo="text",
            )
        )

    # OLS trendline using numpy
    if len(valid) >= 3 and valid[x_col].nunique() >= 2:
        try:
            slope, intercept = np.polyfit(valid[x_col].astype(float), valid[y_col].astype(float) * 100.0, 1)
            x_line = np.linspace(valid[x_col].min(), valid[x_col].max(), 30)
            y_line = slope * x_line + intercept
            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    name="Tendencia OLS",
                    line=dict(color="#94a3b8", width=1.8, dash="dash"),
                )
            )
        except Exception:
            pass

    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=TEXT_PRIMARY)),
        height=380,
        margin=dict(l=15, r=15, t=42, b=20),
        paper_bgcolor=BG_PANEL,
        plot_bgcolor="#0e121b",
        font=dict(color=TEXT_MUTED, size=10),
        xaxis=dict(title="Guidance Score", gridcolor="#1a2130"),
        yaxis=dict(title="Retorno / Alpha (%)", gridcolor="#1a2130"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
    )
    return fig


def build_quantile_bar_chart(quantile_returns: Dict[str, float], horizon_days: int) -> go.Figure:
    """Create dark bar chart for Bottom / Middle / Top quantile returns."""
    labels = ["Bottom", "Middle", "Top"]
    vals = [quantile_returns.get(l, 0.0) * 100.0 for l in labels]
    colors = [COLOR_DOWN, COLOR_NEUTRAL, COLOR_UP]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=vals,
            marker_color=colors,
            text=[f"{v:+.2f}%" for v in vals],
            textposition="outside",
            textfont=dict(color=TEXT_PRIMARY, size=11),
        )
    )
    fig.add_hline(y=0, line_color="#334155", line_width=1)
    fig.update_layout(
        title=dict(text=f"Retorno Medio por Tercil de Guidance ({horizon_days}d)", font=dict(size=13, color=TEXT_PRIMARY)),
        height=380,
        margin=dict(l=15, r=15, t=42, b=20),
        paper_bgcolor=BG_PANEL,
        plot_bgcolor="#0e121b",
        font=dict(color=TEXT_MUTED, size=10),
        yaxis=dict(title="Alpha / Retorno (%)", gridcolor="#1a2130"),
        xaxis=dict(gridcolor="#1a2130"),
        showlegend=False,
    )
    return fig
