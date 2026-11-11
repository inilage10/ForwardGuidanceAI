"""
ForwardGuidanceAI - Market Terminal Data Service
=================================================
Servicio centralizado para:
1. Cinta de mercado en tiempo real (indices globales y activos macro).
2. Datos OHLCV historicos e indicadores tecnicos para graficos interactivos.
3. Buscador universal de empresas/tickers (S&P 500 + SEC EDGAR + Yahoo Finance).
4. Agregador de noticias financieras (Yahoo Finance RSS + Google News RSS) y eventos SEC.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.collector import SEC_HEADERS, get_recent_sec_events
from src.sector_manager import DEFAULT_SECTORS

logger = logging.getLogger(__name__)

# ── Major Indices & Benchmarks ──────────────────────────────────────
TAPE_INDICES: List[Dict[str, str]] = [
    {"symbol": "^GSPC", "short": "S&P 500", "name": "S&P 500 Index", "cat": "Índice USA"},
    {"symbol": "^NDX", "short": "NASDAQ 100", "name": "Nasdaq 100 Index", "cat": "Índice USA"},
    {"symbol": "^DJI", "short": "DOW 30", "name": "Dow Jones Industrial Average", "cat": "Índice USA"},
    {"symbol": "^RUT", "short": "RUSSELL 2K", "name": "Russell 2000 Index", "cat": "Índice USA"},
    {"symbol": "^VIX", "short": "VIX", "name": "CBOE Volatility Index", "cat": "Volatilidad"},
    {"symbol": "^IBEX", "short": "IBEX 35", "name": "IBEX 35 España", "cat": "Índice Europa"},
    {"symbol": "^STOXX50E", "short": "EURO STOXX", "name": "Euro Stoxx 50", "cat": "Índice Europa"},
    {"symbol": "^TNX", "short": "US 10Y", "name": "US Treasury 10Y Yield", "cat": "Renta Fija"},
    {"symbol": "GC=F", "short": "ORO", "name": "Gold Futures", "cat": "Materias Primas"},
    {"symbol": "BTC-USD", "short": "BTC/USD", "name": "Bitcoin USD", "cat": "Cripto"},
]

HOME_CHART_ASSETS: List[Dict[str, str]] = TAPE_INDICES + [
    {"symbol": "^IXIC", "short": "NASDAQ COMP", "name": "Nasdaq Composite", "cat": "Índice USA"},
    {"symbol": "EURUSD=X", "short": "EUR/USD", "name": "Euro / US Dollar", "cat": "Divisas"},
    {"symbol": "CL=F", "short": "WTI CRUDE", "name": "Crude Oil WTI", "cat": "Materias Primas"},
]

PERIOD_MAP: Dict[str, Tuple[str, str]] = {
    "1M": ("1mo", "1d"),
    "3M": ("3mo", "1d"),
    "6M": ("6mo", "1d"),
    "YTD": ("ytd", "1d"),
    "1A": ("1y", "1d"),
    "3A": ("3y", "1d"),
    "MAX": ("max", "1wk"),
}

SEC_TICKERS_CACHE = config.DATA_DIR / "sec_company_tickers.json"

_POS_NEWS_WORDS = {
    "beat", "beats", "surge", "surges", "rally", "rallies", "upgrade", "upgraded",
    "record", "growth", "soar", "soars", "gain", "gains", "bullish", "profit",
    "outperform", "raises", "raised", "boost", "strong", "breakout", "approve",
    "cleared", "clearing", "milestone", "expand", "wins", "partner", "partnership",
    "buyback", "dividend", "top", "tops",
}
_NEG_NEWS_WORDS = {
    "miss", "misses", "drop", "drops", "plunge", "plunges", "downgrade", "downgraded",
    "probe", "lawsuit", "cut", "cuts", "slump", "slumps", "warning", "warns",
    "bearish", "loss", "losses", "fall", "falls", "decline", "declines", "investigation",
    "recall", "fine", "fined", "delay", "delays", "layoff", "layoffs", "risk",
    "concern", "concerns", "tumble", "selloff",
}


# ── Curated S&P 500 + SEC EDGAR Company Directory ───────────────────
def get_curated_companies_map() -> Dict[str, Dict[str, str]]:
    """Build dictionary of ticker -> metadata from the 11 GICS S&P 500 sectors."""
    directory: Dict[str, Dict[str, str]] = {}
    for sector_name, companies in DEFAULT_SECTORS.items():
        if "Big Tech" in sector_name:
            display_sector = "Information Technology / Mega Cap"
        else:
            display_sector = sector_name

        for ticker, info in companies.items():
            if ticker not in directory or "Big Tech" not in sector_name:
                directory[ticker] = {
                    "ticker": ticker,
                    "name": info["name"],
                    "cik": info.get("cik", ""),
                    "cap": info.get("cap", ""),
                    "sector": display_sector,
                    "curated": "true",
                }
    return directory


@st.cache_data(ttl=86400, show_spinner=False)
def load_sec_company_directory() -> List[Dict[str, str]]:
    """Load the full SEC EDGAR public company directory (~10,400 US issuers)."""
    raw_data = None

    # Use local disk cache if fresh (< 7 days)
    if SEC_TICKERS_CACHE.exists():
        try:
            age_sec = datetime.now().timestamp() - SEC_TICKERS_CACHE.stat().st_mtime
            if age_sec < 7 * 86400:
                raw_data = json.loads(SEC_TICKERS_CACHE.read_text(encoding="utf-8"))
        except Exception:
            raw_data = None

    if raw_data is None:
        try:
            resp = requests.get(
                "https://www.sec.gov/files/company_tickers.json",
                headers=SEC_HEADERS,
                timeout=12,
            )
            resp.raise_for_status()
            raw_data = resp.json()
            SEC_TICKERS_CACHE.parent.mkdir(parents=True, exist_ok=True)
            SEC_TICKERS_CACHE.write_text(json.dumps(raw_data), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not fetch SEC company_tickers.json: {e}")
            if SEC_TICKERS_CACHE.exists():
                try:
                    raw_data = json.loads(SEC_TICKERS_CACHE.read_text(encoding="utf-8"))
                except Exception:
                    raw_data = {}
            else:
                raw_data = {}

    curated = get_curated_companies_map()
    entries: List[Dict[str, str]] = []
    seen = set()

    # Keep curated companies first with crisp display names & sectors
    for tk, info in curated.items():
        seen.add(tk)
        entries.append(info)

    if isinstance(raw_data, dict):
        for item in raw_data.values():
            tk = str(item.get("ticker", "")).upper().strip()
            if not tk or tk in seen:
                continue
            seen.add(tk)
            title = str(item.get("title", tk)).strip()
            # Format overly ALL-CAPS corporate titles cleanly
            clean_title = title.title() if title.isupper() else title
            entries.append({
                "ticker": tk,
                "name": clean_title,
                "cik": str(item.get("cik_str", "")).zfill(10),
                "cap": "",
                "sector": "Mercado USA (SEC EDGAR)",
                "curated": "false",
            })

    return entries


def get_company_info(ticker: str) -> Dict[str, str]:
    """Return metadata dict for a ticker (name, cik, sector, cap)."""
    tk_up = ticker.upper().strip()
    curated = get_curated_companies_map()
    if tk_up in curated:
        return curated[tk_up]

    for item in load_sec_company_directory():
        if item["ticker"] == tk_up:
            return item

    return {
        "ticker": tk_up,
        "name": tk_up,
        "cik": "",
        "cap": "",
        "sector": "Renta Variable",
        "curated": "false",
    }


def resolve_company_search(query: str) -> Tuple[Optional[Dict[str, str]], List[Dict[str, str]]]:
    """Resolve user search text into (best_match, candidates).

    Prioritizes:
    1) Exact ticker match (e.g. 'AAPL', 'PLTR', 'NVDA')
    2) Curated S&P 500 company name match (e.g. 'Apple', 'Microsoft', 'JPMorgan')
    3) SEC EDGAR prefix / company title match
    4) Yahoo Finance Search fallback
    """
    q = (query or "").strip()
    if not q:
        return None, []

    q_upper = q.upper()
    q_lower = q.lower()
    directory = load_sec_company_directory()

    # Support strings like "AAPL - Apple"
    if " - " in q:
        maybe_tk = q.split(" - ", 1)[0].strip().upper()
        for item in directory:
            if item["ticker"] == maybe_tk:
                return item, [item]

    # 1. Exact ticker match
    for item in directory:
        if item["ticker"] == q_upper:
            return item, [item]

    # 2. Curated exact/substring company name match
    curated_matches = [
        item for item in directory
        if item.get("curated") == "true"
        and (q_lower in item["name"].lower() or item["ticker"].startswith(q_upper))
    ]
    if curated_matches:
        curated_matches.sort(
            key=lambda x: (
                0 if x["name"].lower() == q_lower else (1 if x["name"].lower().startswith(q_lower) else 2),
                len(x["name"]),
            )
        )
        return curated_matches[0], curated_matches[:8]

    # 3. Full SEC EDGAR directory match
    sec_matches = []
    for item in directory:
        tk = item["ticker"]
        nm = item["name"].lower()
        if tk.startswith(q_upper) or q_lower in nm:
            score = 0
            if tk.startswith(q_upper):
                score = 1
            elif nm.startswith(q_lower):
                score = 2
            elif re.search(rf"\b{re.escape(q_lower)}\b", nm):
                score = 3
            else:
                score = 4
            sec_matches.append((score, len(item["name"]), item))

    if sec_matches:
        sec_matches.sort(key=lambda t: (t[0], t[1]))
        candidates = [t[2] for t in sec_matches[:10]]
        return candidates[0], candidates

    # 4. Fallback via Yahoo Finance Search
    try:
        search = yf.Search(q, max_results=6)
        quotes = getattr(search, "quotes", []) or []
        yf_candidates = []
        for qt in quotes:
            sym = str(qt.get("symbol", "")).upper().strip()
            qtype = qt.get("quoteType", "")
            if sym and qtype in ("EQUITY", "ETF", ""):
                yf_candidates.append({
                    "ticker": sym,
                    "name": qt.get("shortname") or qt.get("longname") or sym,
                    "cik": "",
                    "cap": "",
                    "sector": qt.get("sector") or "Mercado Global",
                    "curated": "false",
                })
        if yf_candidates:
            return yf_candidates[0], yf_candidates
    except Exception as e:
        logger.debug(f"yf.Search fallback failed for '{q}': {e}")

    return None, []


# ── Ticker Tape & Market Quotes ─────────────────────────────────────
@st.cache_data(ttl=120, show_spinner=False)
def get_market_tape_quotes() -> List[Dict]:
    """Fetch latest quote and daily % change for major global indices."""
    symbols = [idx["symbol"] for idx in TAPE_INDICES]
    results: List[Dict] = []

    try:
        raw = yf.download(
            symbols,
            period="7d",
            interval="1d",
            progress=False,
            auto_adjust=False,
            threads=True,
        )
        close_df = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    except Exception as e:
        logger.warning(f"Batch ticker tape download failed: {e}")
        close_df = pd.DataFrame()

    for idx in TAPE_INDICES:
        sym = idx["symbol"]
        price = None
        delta = 0.0
        pct = 0.0

        try:
            if not close_df.empty and sym in close_df.columns:
                series = close_df[sym].dropna()
                if len(series) >= 2:
                    price = float(series.iloc[-1])
                    prev = float(series.iloc[-2])
                    delta = price - prev
                    pct = (delta / prev * 100.0) if prev != 0 else 0.0
                elif len(series) == 1:
                    price = float(series.iloc[-1])
        except Exception:
            pass

        results.append({
            "symbol": sym,
            "short": idx["short"],
            "name": idx["name"],
            "price": price,
            "delta": delta,
            "pct": pct,
            "is_yield": sym == "^TNX",
        })

    return results


@st.cache_data(ttl=120, show_spinner=False)
def get_ticker_snapshot(symbol: str) -> Dict:
    """Get quick latest price, previous close, and daily % change for a symbol."""
    sym = symbol.upper().strip()
    try:
        df = yf.download(sym, period="7d", interval="1d", progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        if not df.empty and "Close" in df.columns:
            closes = df["Close"].dropna()
            if len(closes) >= 2:
                last_p = float(closes.iloc[-1])
                prev_p = float(closes.iloc[-2])
                delta = last_p - prev_p
                pct = (delta / prev_p * 100.0) if prev_p else 0.0
                vol = float(df["Volume"].dropna().iloc[-1]) if "Volume" in df.columns and not df["Volume"].dropna().empty else 0.0
                return {
                    "price": last_p,
                    "prev_close": prev_p,
                    "delta": delta,
                    "pct": pct,
                    "volume": vol,
                }
            elif len(closes) == 1:
                return {"price": float(closes.iloc[-1]), "prev_close": float(closes.iloc[-1]), "delta": 0.0, "pct": 0.0, "volume": 0.0}
    except Exception as e:
        logger.debug(f"Snapshot fetch failed for {sym}: {e}")

    # Fallback to local price CSV if cached
    csv_file = config.PRICES_DIR / f"{sym}_prices.csv"
    if csv_file.exists():
        try:
            cdf = pd.read_csv(csv_file)
            if len(cdf) >= 2 and "Close" in cdf.columns:
                last_p = float(cdf["Close"].iloc[-1])
                prev_p = float(cdf["Close"].iloc[-2])
                delta = last_p - prev_p
                pct = (delta / prev_p * 100.0) if prev_p else 0.0
                return {"price": last_p, "prev_close": prev_p, "delta": delta, "pct": pct, "volume": 0.0}
        except Exception:
            pass

    return {"price": None, "prev_close": None, "delta": 0.0, "pct": 0.0, "volume": 0.0}


@st.cache_data(ttl=180, show_spinner=False)
def get_chart_ohlcv(symbol: str, period_key: str = "1A") -> Tuple[pd.DataFrame, Dict]:
    """Download OHLCV series for charting and calculate risk/return summary metrics."""
    period, interval = PERIOD_MAP.get(period_key, ("1y", "1d"))
    df = pd.DataFrame()

    try:
        raw = yf.download(
            symbol,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
        )
        if not raw.empty:
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [col[0] for col in raw.columns]
            raw = raw.reset_index()
            date_col = "Date" if "Date" in raw.columns else raw.columns[0]
            raw["Date"] = pd.to_datetime(raw[date_col]).dt.tz_localize(None)
            cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in raw.columns]
            df = raw[cols].dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)
    except Exception as e:
        logger.warning(f"yfinance OHLCV error for {symbol}: {e}")

    # Local CSV fallback for cached stock prices
    if df.empty:
        clean_sym = symbol.upper().strip()
        local_csv = config.PRICES_DIR / f"{clean_sym}_prices.csv"
        if local_csv.exists():
            try:
                cdf = pd.read_csv(local_csv)
                cdf["Date"] = pd.to_datetime(cdf["Date"]).dt.tz_localize(None)
                cdf = cdf.sort_values("Date").reset_index(drop=True)
                days_map = {"1M": 30, "3M": 90, "6M": 180, "1A": 365, "3A": 1095}
                if period_key in days_map:
                    cutoff = cdf["Date"].max() - pd.Timedelta(days=days_map[period_key])
                    cdf = cdf[cdf["Date"] >= cutoff]
                elif period_key == "YTD":
                    yr_start = pd.Timestamp(year=cdf["Date"].max().year, month=1, day=1)
                    cdf = cdf[cdf["Date"] >= yr_start]
                df = cdf
            except Exception:
                pass

    stats = {
        "last_price": None,
        "day_change": 0.0,
        "day_pct": 0.0,
        "period_return": 0.0,
        "period_high": None,
        "period_low": None,
        "volatility_ann": 0.0,
        "max_drawdown": 0.0,
        "sharpe": 0.0,
    }

    if not df.empty and len(df) >= 2:
        closes = df["Close"].astype(float)
        last_p = float(closes.iloc[-1])
        prev_p = float(closes.iloc[-2])
        first_p = float(closes.iloc[0])

        day_chg = last_p - prev_p
        day_pct = (day_chg / prev_p * 100.0) if prev_p else 0.0
        per_ret = ((last_p - first_p) / first_p * 100.0) if first_p else 0.0

        high_val = float(df["High"].max()) if "High" in df.columns else float(closes.max())
        low_val = float(df["Low"].min()) if "Low" in df.columns else float(closes.min())

        rets = closes.pct_change().dropna()
        ann_factor = 52.0 if interval == "1wk" else 252.0
        vol_ann = float(rets.std() * np.sqrt(ann_factor) * 100.0) if len(rets) > 2 else 0.0
        mean_ann = float(rets.mean() * ann_factor * 100.0) if len(rets) > 2 else 0.0
        sharpe = (mean_ann / vol_ann) if vol_ann > 0 else 0.0

        cummax = closes.cummax()
        dd = (closes - cummax) / cummax
        max_dd = float(dd.min() * 100.0)

        stats.update({
            "last_price": last_p,
            "day_change": day_chg,
            "day_pct": day_pct,
            "period_return": per_ret,
            "period_high": high_val,
            "period_low": low_val,
            "volatility_ann": vol_ann,
            "max_drawdown": max_dd,
            "sharpe": sharpe,
        })

    return df, stats


# ── News & Regulatory Filings Aggregator ────────────────────────────
def _classify_headline_sentiment(title: str) -> str:
    """Classify financial headline sentiment using domain keyword lexicons."""
    tokens = set(re.findall(r"[a-z]+", title.lower()))
    pos_hits = len(tokens & _POS_NEWS_WORDS)
    neg_hits = len(tokens & _NEG_NEWS_WORDS)
    if pos_hits > neg_hits:
        return "Positivo"
    if neg_hits > pos_hits:
        return "Negativo"
    return "Neutral"


def _clean_rss_summary(raw_desc: str) -> str:
    if not raw_desc:
        return ""
    try:
        soup = BeautifulSoup(raw_desc, "html.parser")
        txt = re.sub(r"\s+", " ", soup.get_text(" ")).strip()
        return txt[:240] + ("..." if len(txt) > 240 else "")
    except Exception:
        return re.sub(r"<[^>]+>", "", raw_desc)[:240].strip()


def _parse_rss_datetime(pub_str: str) -> Tuple[float, str]:
    if not pub_str:
        return 0.0, "Reciente"
    try:
        dt = parsedate_to_datetime(pub_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        diff_sec = max(0, int((now - dt).total_seconds()))
        if diff_sec < 3600:
            mins = max(1, diff_sec // 60)
            rel = f"Hace {mins}m"
        elif diff_sec < 86400:
            hrs = diff_sec // 3600
            rel = f"Hace {hrs}h"
        else:
            days = diff_sec // 86400
            rel = f"Hace {days}d · {dt.strftime('%d %b')}"
        return dt.timestamp(), rel
    except Exception:
        return 0.0, pub_str[:16]


@st.cache_data(ttl=300, show_spinner=False)
def get_company_news(ticker: str, company_name: str = "", limit: int = 18) -> List[Dict]:
    """Fetch & deduplicate real-time financial news from Yahoo Finance RSS and Google News RSS."""
    tk = ticker.upper().strip()
    articles: List[Dict] = []
    seen_prefixes = set()

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ForwardGuidanceAI/2.0"}

    # Source 1: Yahoo Finance RSS
    try:
        y_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={tk}&region=US&lang=en-US"
        r = requests.get(y_url, headers=headers, timeout=8)
        if r.ok:
            root = ET.fromstring(r.content)
            for item in root.findall(".//item"):
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                pub_raw = (item.findtext("pubDate") or "").strip()
                desc = _clean_rss_summary(item.findtext("description") or "")
                if not title:
                    continue
                norm_key = re.sub(r"[^a-z0-9]", "", title.lower())[:45]
                if norm_key in seen_prefixes:
                    continue
                seen_prefixes.add(norm_key)
                ts, rel_time = _parse_rss_datetime(pub_raw)
                articles.append({
                    "title": title,
                    "link": link,
                    "source": "Yahoo Finance",
                    "published_ts": ts,
                    "published_rel": rel_time,
                    "summary": desc,
                    "sentiment": _classify_headline_sentiment(title),
                })
    except Exception as e:
        logger.debug(f"Yahoo RSS error for {tk}: {e}")

    # Source 2: Google News Financial RSS
    try:
        clean_name = re.sub(r"\b(Inc\.?|Corp\.?|Corporation|plc|Co\.?|Holdings)\b", "", company_name, flags=re.I).strip()
        q_str = f"{tk} {clean_name} stock".strip() if clean_name and clean_name.upper() != tk else f"{tk} stock"
        g_url = f"https://news.google.com/rss/search?q={requests.utils.quote(q_str)}+when:14d&hl=en-US&gl=US&ceid=US:en"
        r2 = requests.get(g_url, headers=headers, timeout=8)
        if r2.ok:
            root2 = ET.fromstring(r2.content)
            for item in root2.findall(".//item"):
                raw_title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                pub_raw = (item.findtext("pubDate") or "").strip()
                source_el = item.findtext("source") or "Google News"

                # Google News appends " - SourceName" at the end of title
                if " - " in raw_title and raw_title.endswith(source_el):
                    title = raw_title[: -(len(source_el) + 3)].strip()
                else:
                    title = raw_title

                if not title:
                    continue
                norm_key = re.sub(r"[^a-z0-9]", "", title.lower())[:45]
                if norm_key in seen_prefixes:
                    continue
                seen_prefixes.add(norm_key)
                ts, rel_time = _parse_rss_datetime(pub_raw)
                articles.append({
                    "title": title,
                    "link": link,
                    "source": source_el,
                    "published_ts": ts,
                    "published_rel": rel_time,
                    "summary": "",
                    "sentiment": _classify_headline_sentiment(title),
                })
    except Exception as e:
        logger.debug(f"Google News RSS error for {tk}: {e}")

    articles.sort(key=lambda x: x["published_ts"], reverse=True)
    return articles[:limit]


@st.cache_data(ttl=900, show_spinner=False)
def get_recent_sec_events_cached(cik: str, limit: int = 12) -> List[Dict]:
    """Cached fetch of recent 10-K, 10-Q, 8-K SEC EDGAR events for Module A."""
    if not cik:
        return []
    return get_recent_sec_events(cik, limit=limit)
