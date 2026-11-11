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
import unicodedata
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
MADRID_TICKERS_CACHE = config.DATA_DIR / "madrid_company_tickers.json"

EXCHANGE_LABELS: Dict[str, str] = {
    "MCE": "Bolsa de Madrid",
    "BME": "Bolsa de Madrid",
    "NYQ": "NYSE",
    "NMS": "NASDAQ",
    "NGM": "NASDAQ",
    "NCM": "NASDAQ",
    "ASE": "NYSE American",
    "PCX": "NYSE Arca",
    "LSE": "London SE",
    "PAR": "Euronext Paris",
    "GER": "XETRA",
    "FRA": "Frankfurt",
    "AMS": "Euronext Amsterdam",
    "MIL": "Borsa Italiana",
    "BRU": "Euronext Brussels",
    "LIS": "Euronext Lisbon",
    "EBS": "SIX Swiss",
    "TOR": "TSX Toronto",
    "TYO": "Tokyo SE",
    "HKG": "HKEX Hong Kong",
    "ASX": "ASX Australia",
    "SAO": "B3 Brasil",
    "MEX": "BMV México",
    "CPH": "Nasdaq Copenhagen",
    "STO": "Nasdaq Stockholm",
    "OSL": "Oslo Børs",
    "PNK": "OTC USA",
    "OQX": "OTC USA",
    "OBB": "OTC USA",
}

PRIMARY_EXCHANGES = {
    "NYQ", "NMS", "NGM", "NCM", "ASE", "MCE", "BME", "LSE", "PAR", "GER",
    "AMS", "MIL", "BRU", "LIS", "EBS", "TOR", "TYO", "HKG", "ASX", "CPH",
    "STO", "OSL", "SAO", "MEX", "KRX", "NSI", "BOM",
}

SECONDARY_EXCHANGE_SUFFIXES = (".NE", ".TI", ".HM", ".DU", ".HA", ".SG", ".BE", ".MU", ".F", ".VI", ".PR")
SECONDARY_EXCHANGES = {"PNK", "OQX", "OBB", "NEO", "TLO", "HAN", "DUS", "HAM", "STU", "BER", "MUN"}

CURRENCY_SYMBOLS: Dict[str, str] = {
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "GBp": "p",
    "JPY": "¥",
    "CHF": "CHF ",
    "CAD": "C$",
    "HKD": "HK$",
    "AUD": "A$",
    "BRL": "R$",
    "MXN": "MX$",
    "DKK": "kr ",
    "SEK": "kr ",
    "NOK": "kr ",
}

SUFFIX_CURRENCY_HINTS: Dict[str, Tuple[str, str]] = {
    ".MC": ("Bolsa de Madrid", "EUR"),
    ".PA": ("Euronext Paris", "EUR"),
    ".DE": ("XETRA", "EUR"),
    ".AS": ("Euronext Amsterdam", "EUR"),
    ".MI": ("Borsa Italiana", "EUR"),
    ".BR": ("Euronext Brussels", "EUR"),
    ".LS": ("Euronext Lisbon", "EUR"),
    ".L": ("London SE", "GBP"),
    ".SW": ("SIX Swiss", "CHF"),
    ".TO": ("TSX Toronto", "CAD"),
    ".T": ("Tokyo SE", "JPY"),
    ".HK": ("HKEX", "HKD"),
    ".AX": ("ASX Australia", "AUD"),
    ".SA": ("B3 Brasil", "BRL"),
    ".MX": ("BMV México", "MXN"),
    ".CO": ("Nasdaq Copenhagen", "DKK"),
    ".ST": ("Nasdaq Stockholm", "SEK"),
    ".OL": ("Oslo Børs", "NOK"),
}

MADRID_SEARCH_ALIASES: Dict[str, str] = {
    "ITX.MC": "inditex zara pull bear massimo dutti bershka stradivarius",
    "IBE.MC": "iberdrola",
    "SAN.MC": "santander banco santander",
    "BBVA.MC": "bbva banco bilbao vizcaya argentaria",
    "CABK.MC": "caixabank caixa la caixa",
    "AMS.MC": "amadeus it group",
    "AENA.MC": "aena aeropuertos",
    "FER.MC": "ferrovial",
    "TEF.MC": "telefonica movistar",
    "REP.MC": "repsol",
    "IDR.MC": "indra sistemas",
    "ACS.MC": "acs actividades construccion servicios",
    "NTGY.MC": "naturgy gas natural fenosa",
    "ELE.MC": "endesa",
    "ENG.MC": "enagas",
    "RED.MC": "redeia red electrica",
    "CLNX.MC": "cellnex telecom",
    "MRL.MC": "merlin properties",
    "COL.MC": "inmobiliaria colonial",
    "ANA.MC": "acciona",
    "ANE.MC": "acciona energia",
    "GRF.MC": "grifols",
    "MTS.MC": "arcelormittal arcelor",
    "BKT.MC": "bankinter",
    "SAB.MC": "sabadell banco sabadell",
    "UNI.MC": "unicaja banco",
    "MAP.MC": "mapfre",
    "FDR.MC": "fluidra",
    "ACX.MC": "acerinox",
    "ROVI.MC": "rovi laboratorios farmaceuticos",
    "SLR.MC": "solaria energia medio ambiente",
    "SCYR.MC": "sacyr",
    "LOG.MC": "logista integral",
    "PUIG.MC": "puig brands",
    "IAG.MC": "iag iberia british airways international airlines",
    "VID.MC": "vidrala",
    "VIS.MC": "viscofan",
    "EBRO.MC": "ebro foods",
    "PHM.MC": "pharmamar pharma mar",
    "CAF.MC": "caf construcciones auxiliar ferrocarriles",
    "GEST.MC": "gestamp automocion",
    "CIE.MC": "cie automotive",
    "ALM.MC": "almirall",
    "FAE.MC": "faes farma",
    "PSG.MC": "prosegur",
    "TUB.MC": "tubacex",
    "TLGO.MC": "talgo",
    "A3M.MC": "atresmedia",
    "GRE.MC": "grenergy renovables",
    "TRE.MC": "tecnicas reunidas",
    "DOM.MC": "global dominion",
    "ENC.MC": "ence energia celulosa",
    "LDA.MC": "linea directa aseguradora",
    "MVC.MC": "metrovacesa",
    "HOME.MC": "neinor homes",
    "AEDAS.MC": "aedas homes",
}

_POS_NEWS_WORDS = {
    "beat", "beats", "surge", "surges", "rally", "rallies", "upgrade", "upgraded",
    "record", "growth", "soar", "soars", "gain", "gains", "bullish", "profit",
    "outperform", "raises", "raised", "boost", "strong", "breakout", "approve",
    "cleared", "clearing", "milestone", "expand", "wins", "partner", "partnership",
    "buyback", "dividend", "top", "tops",
    # Spanish financial news lexicon
    "sube", "subida", "suben", "gana", "ganancias", "beneficio", "beneficios",
    "dividendo", "dividendos", "recompra", "crece", "crecimiento", "supera",
    "mejora", "alcista", "repunta", "repunte", "dispara", "lidera", "expansion",
    "contrato", "adjudica", "aprobado", "objetivo",
}
_NEG_NEWS_WORDS = {
    "miss", "misses", "drop", "drops", "plunge", "plunges", "downgrade", "downgraded",
    "probe", "lawsuit", "cut", "cuts", "slump", "slumps", "warning", "warns",
    "bearish", "loss", "losses", "fall", "falls", "decline", "declines", "investigation",
    "recall", "fine", "fined", "delay", "delays", "layoff", "layoffs", "risk",
    "concern", "concerns", "tumble", "selloff",
    # Spanish financial news lexicon
    "cae", "caida", "caen", "baja", "bajan", "pierde", "perdida", "perdidas",
    "recorte", "recorta", "rebaja", "alerta", "multa", "investigacion", "demanda",
    "desplome", "desploma", "despidos", "huelga", "riesgo", "deuda", "penaliza",
}


def _fold_text(text: str) -> str:
    """Normalize unicode accents so Spanish/European names match cleanly with or without tildes."""
    if not text:
        return ""
    norm = unicodedata.normalize("NFKD", str(text))
    return "".join(ch for ch in norm if not unicodedata.combining(ch)).lower().strip()


def format_exchange_label(raw_exchange: Optional[str], ticker: str = "") -> str:
    """Convert raw Yahoo/exchange codes (MCE, NMS, NYQ, etc.) into readable market names."""
    code = str(raw_exchange or "").upper().strip()
    if code in EXCHANGE_LABELS:
        return EXCHANGE_LABELS[code]
    tk_up = str(ticker or "").upper()
    for suf, (ex_lbl, _) in SUFFIX_CURRENCY_HINTS.items():
        if tk_up.endswith(suf):
            return ex_lbl
    if code:
        return code
    return "Mercado USA" if "." not in tk_up else "Bolsa Global"


def infer_currency_for_ticker(ticker: str, raw_currency: Optional[str] = None) -> str:
    """Infer trading currency code (EUR, USD, GBP, JPY, etc.) from quote metadata or ticker suffix."""
    if raw_currency and str(raw_currency).strip():
        return str(raw_currency).strip()
    tk_up = str(ticker or "").upper()
    for suf, (_, curr) in SUFFIX_CURRENCY_HINTS.items():
        if tk_up.endswith(suf):
            return curr
    return "USD"


def format_currency_price(value: Optional[float], currency: str = "USD") -> str:
    """Format a price float with the appropriate currency symbol (€, $, £, ¥, etc.)."""
    if value is None:
        return "--"
    sym = CURRENCY_SYMBOLS.get(currency, f"{currency} ")
    if currency in {"JPY", "KRW"}:
        return f"{sym}{value:,.0f}"
    if sym.endswith(" "):
        return f"{sym}{value:,.2f}"
    if currency == "GBp":
        return f"{value:,.2f}p"
    return f"{sym}{value:,.2f}"


# ── Curated S&P 500 + IBEX 35 + Global Company Directory ────────────
def get_curated_companies_map() -> Dict[str, Dict[str, str]]:
    """Build dictionary of ticker -> metadata from the 11 GICS S&P 500 sectors & IBEX 35."""
    directory: Dict[str, Dict[str, str]] = {}
    for sector_name, companies in DEFAULT_SECTORS.items():
        if "Big Tech" in sector_name:
            display_sector = "Information Technology / Mega Cap"
        else:
            display_sector = sector_name

        for ticker, info in companies.items():
            if ticker not in directory or "Big Tech" not in sector_name:
                ex_name = info.get("exchange") or format_exchange_label("", ticker)
                curr = info.get("currency") or infer_currency_for_ticker(ticker)
                aliases = MADRID_SEARCH_ALIASES.get(ticker, "")
                directory[ticker] = {
                    "ticker": ticker,
                    "name": info["name"],
                    "cik": info.get("cik", ""),
                    "cap": info.get("cap", ""),
                    "sector": display_sector,
                    "exchange": ex_name,
                    "currency": curr,
                    "aliases": aliases,
                    "curated": "true",
                }
    return directory


@st.cache_data(ttl=86400, show_spinner=False)
def load_madrid_company_directory() -> List[Dict[str, str]]:
    """Load all listed companies on Bolsa de Madrid / BME (MCE) via Yahoo Finance screener."""
    raw_quotes: Optional[List[dict]] = None

    if MADRID_TICKERS_CACHE.exists():
        try:
            age_sec = datetime.now().timestamp() - MADRID_TICKERS_CACHE.stat().st_mtime
            if age_sec < 7 * 86400:
                cached = json.loads(MADRID_TICKERS_CACHE.read_text(encoding="utf-8"))
                if isinstance(cached, list) and cached:
                    raw_quotes = cached
        except Exception:
            raw_quotes = None

    if raw_quotes is None:
        try:
            q = yf.EquityQuery("eq", ["exchange", "MCE"])
            res = yf.screen(q, size=250, offset=0)
            quotes = res.get("quotes", []) if isinstance(res, dict) else []
            # Also fetch page 2 if total ever exceeds 250
            total = int(res.get("total", len(quotes)) or len(quotes)) if isinstance(res, dict) else len(quotes)
            if total > len(quotes):
                res2 = yf.screen(q, size=250, offset=len(quotes))
                quotes.extend(res2.get("quotes", []) if isinstance(res2, dict) else [])

            raw_quotes = [
                {
                    "symbol": str(item.get("symbol", "")).upper().strip(),
                    "longName": item.get("longName") or item.get("shortName") or "",
                    "shortName": item.get("shortName") or "",
                    "sector": item.get("sector") or "Bolsa de Madrid",
                    "marketCap": item.get("marketCap"),
                }
                for item in quotes
                if item.get("symbol")
            ]
            if raw_quotes:
                MADRID_TICKERS_CACHE.parent.mkdir(parents=True, exist_ok=True)
                MADRID_TICKERS_CACHE.write_text(json.dumps(raw_quotes, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not fetch Bolsa de Madrid (MCE) screener: {e}")
            if MADRID_TICKERS_CACHE.exists():
                try:
                    raw_quotes = json.loads(MADRID_TICKERS_CACHE.read_text(encoding="utf-8"))
                except Exception:
                    raw_quotes = []
            else:
                raw_quotes = []

    curated = get_curated_companies_map()
    entries: List[Dict[str, str]] = []
    seen = set()

    # Keep curated IBEX 35 entries first
    for tk, info in curated.items():
        if tk.endswith(".MC"):
            seen.add(tk)
            entries.append(info)

    for item in raw_quotes or []:
        tk = str(item.get("symbol", "")).upper().strip()
        if not tk or tk in seen:
            continue
        seen.add(tk)
        raw_nm = str(item.get("longName") or item.get("shortName") or tk).strip()
        clean_nm = raw_nm.title() if raw_nm.isupper() else raw_nm
        cap_val = item.get("marketCap")
        cap_str = ""
        if isinstance(cap_val, (int, float)) and cap_val > 0:
            cap_str = f"{cap_val / 1e9:.1f}B" if cap_val >= 1e9 else f"{cap_val / 1e6:.0f}M"
        entries.append({
            "ticker": tk,
            "name": clean_nm,
            "cik": "",
            "cap": cap_str,
            "sector": item.get("sector") or "Bolsa de Madrid",
            "exchange": "Bolsa de Madrid",
            "currency": "EUR",
            "aliases": MADRID_SEARCH_ALIASES.get(tk, ""),
            "curated": "false",
        })

    return entries


@st.cache_data(ttl=86400, show_spinner=False)
def load_sec_company_directory() -> List[Dict[str, str]]:
    """Load the full SEC EDGAR public company directory (~10,400 US issuers) + curated S&P 500."""
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

    # Keep curated US companies first with crisp display names & sectors
    for tk, info in curated.items():
        if not tk.endswith(".MC"):
            seen.add(tk)
            entries.append(info)

    if isinstance(raw_data, dict):
        for item in raw_data.values():
            tk = str(item.get("ticker", "")).upper().strip()
            if not tk or tk in seen:
                continue
            seen.add(tk)
            title = str(item.get("title", tk)).strip()
            clean_title = title.title() if title.isupper() else title
            entries.append({
                "ticker": tk,
                "name": clean_title,
                "cik": str(item.get("cik_str", "")).zfill(10),
                "cap": "",
                "sector": "Mercado USA (SEC EDGAR)",
                "exchange": "Mercado USA",
                "currency": "USD",
                "aliases": "",
                "curated": "false",
            })

    return entries


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_yf_company_profile(ticker: str) -> Dict[str, str]:
    """Fetch live metadata (name, exchange, currency, sector, marketCap) for any global public equity."""
    tk_up = ticker.upper().strip()
    try:
        info = yf.Ticker(tk_up).info or {}
        if not info or (not info.get("shortName") and not info.get("longName") and info.get("regularMarketPrice") is None):
            return {}
        raw_nm = str(info.get("longName") or info.get("shortName") or tk_up).strip()
        clean_nm = raw_nm.title() if raw_nm.isupper() else raw_nm
        ex_lbl = format_exchange_label(info.get("exchange") or info.get("fullExchangeName"), tk_up)
        curr = infer_currency_for_ticker(tk_up, info.get("currency"))
        mcap = info.get("marketCap")
        cap_str = ""
        if isinstance(mcap, (int, float)) and mcap > 0:
            cap_str = f"{mcap / 1e9:.1f}B" if mcap >= 1e9 else f"{mcap / 1e6:.0f}M"
        return {
            "ticker": tk_up,
            "name": clean_nm,
            "cik": "",
            "cap": cap_str,
            "sector": info.get("sector") or ex_lbl,
            "exchange": ex_lbl,
            "currency": curr,
            "aliases": "",
            "curated": "false",
        }
    except Exception as e:
        logger.debug(f"Could not fetch live yf profile for {tk_up}: {e}")
        return {}


def get_company_info(ticker: str) -> Dict[str, str]:
    """Return enriched metadata dict for any global ticker (name, cik, sector, exchange, currency)."""
    tk_up = ticker.upper().strip()
    curated = get_curated_companies_map()
    if tk_up in curated:
        return curated[tk_up]

    for item in load_madrid_company_directory():
        if item["ticker"] == tk_up:
            return item

    for item in load_sec_company_directory():
        if item["ticker"] == tk_up:
            return item

    yf_prof = _fetch_yf_company_profile(tk_up)
    if yf_prof:
        return yf_prof

    return {
        "ticker": tk_up,
        "name": tk_up,
        "cik": "",
        "cap": "",
        "sector": "Renta Variable Global",
        "exchange": format_exchange_label("", tk_up),
        "currency": infer_currency_for_ticker(tk_up),
        "aliases": "",
        "curated": "false",
    }


def _yahoo_global_search(query: str, max_results: int = 15) -> List[Dict[str, str]]:
    """Search global equities across all international exchanges via Yahoo Finance."""
    results: List[Dict[str, str]] = []
    seen = set()
    try:
        search = yf.Search(query, max_results=max_results)
        quotes = getattr(search, "quotes", []) or []
        for qt in quotes:
            sym = str(qt.get("symbol", "")).upper().strip()
            qtype = str(qt.get("quoteType", "")).upper()
            if not sym or sym in seen or qtype not in ("EQUITY", "ETF", ""):
                continue
            seen.add(sym)
            raw_ex = str(qt.get("exchange", "")).upper().strip()
            ex_label = format_exchange_label(raw_ex, sym)
            curr = infer_currency_for_ticker(sym)
            raw_nm = str(qt.get("longname") or qt.get("shortname") or sym).strip()
            clean_nm = raw_nm.title() if raw_nm.isupper() else raw_nm
            results.append({
                "ticker": sym,
                "name": clean_nm,
                "cik": "",
                "cap": "",
                "sector": qt.get("sector") or ex_label,
                "exchange": ex_label,
                "raw_exchange": raw_ex,
                "currency": curr,
                "aliases": "",
                "curated": "false",
            })
    except Exception as e:
        logger.debug(f"yf.Search failed for '{query}': {e}")
    return results


def _score_global_candidate(item: Dict[str, str], q_upper: str, q_fold: str) -> Tuple[int, int, str]:
    """Rank search candidates combining text match quality and primary exchange quality."""
    tk = item["ticker"].upper()
    tk_base = tk.split(".", 1)[0]
    nm_fold = _fold_text(item.get("name", ""))
    alias_fold = _fold_text(item.get("aliases", ""))
    raw_ex = item.get("raw_exchange", "").upper()
    is_curated = item.get("curated") == "true"
    is_madrid = tk.endswith(".MC") or item.get("exchange") == "Bolsa de Madrid"

    # Match quality tier
    if tk == q_upper:
        match_rank = 0
    elif tk_base == q_upper and is_madrid:
        match_rank = 1
    elif nm_fold == q_fold or f" {q_fold} " in f" {alias_fold} ":
        match_rank = 2
    elif is_curated and (q_fold in nm_fold or q_fold in alias_fold):
        match_rank = 3
    elif tk_base == q_upper:
        match_rank = 4
    elif tk.startswith(q_upper) or nm_fold.startswith(q_fold) or alias_fold.startswith(q_fold):
        match_rank = 5
    elif re.search(rf"\b{re.escape(q_fold)}\b", nm_fold) or re.search(rf"\b{re.escape(q_fold)}\b", alias_fold):
        match_rank = 6
    else:
        match_rank = 8

    # Prefer primary exchanges over OTC/regional duplicates
    if is_madrid or (raw_ex in PRIMARY_EXCHANGES) or (not raw_ex and "." not in tk):
        ex_penalty = 0
    elif tk.endswith(SECONDARY_EXCHANGE_SUFFIXES) or raw_ex in SECONDARY_EXCHANGES:
        ex_penalty = 5
    else:
        ex_penalty = 2

    # Slight boost for curated flagship companies
    curated_bonus = 0 if is_curated else 1
    return (match_rank + ex_penalty, curated_bonus, len(item.get("name", "")))


def resolve_company_search(query: str) -> Tuple[Optional[Dict[str, str]], List[Dict[str, str]]]:
    """Resolve any US, Spanish (Bolsa de Madrid), or global public equity query."""
    q = (query or "").strip()
    if not q:
        return None, []

    if " - " in q:
        q = q.split(" - ", 1)[0].strip()

    q_upper = q.upper()
    q_fold = _fold_text(q)

    madrid_dir = load_madrid_company_directory()
    sec_dir = load_sec_company_directory()
    combined_local = madrid_dir + sec_dir

    # Fast exact shortcut when user explicitly typed an exchange-suffixed ticker (e.g. ITX.MC, SAP.DE, MC.PA)
    if "." in q_upper:
        for item in combined_local:
            if item["ticker"] == q_upper:
                return item, [item]
        yf_cands = _yahoo_global_search(q_upper, max_results=8)
        for item in yf_cands:
            if item["ticker"] == q_upper:
                return item, [item]
        prof = _fetch_yf_company_profile(q_upper)
        if prof:
            return prof, [prof]
        if yf_cands:
            return yf_cands[0], yf_cands
        return None, []

    # Check local Madrid + SEC matches
    local_matches: List[Dict[str, str]] = []
    seen_tk = set()

    for item in combined_local:
        tk = item["ticker"]
        tk_base = tk.split(".", 1)[0]
        nm_fold = _fold_text(item.get("name", ""))
        al_fold = _fold_text(item.get("aliases", ""))
        if (
            tk == q_upper
            or tk_base == q_upper
            or tk.startswith(q_upper)
            or q_fold in nm_fold
            or q_fold in al_fold
        ):
            if tk not in seen_tk:
                seen_tk.add(tk)
                local_matches.append(item)

    # Check whether query already resolves cleanly without network latency:
    # Fast-path only when exact single curated ticker/name has zero cross-exchange ambiguity
    has_madrid_match = any(m["ticker"].endswith(".MC") for m in local_matches)
    exact_curated_us = [
        m for m in local_matches
        if m.get("curated") == "true"
        and not m["ticker"].endswith(".MC")
        and (m["ticker"] == q_upper or _fold_text(m["name"]) == q_fold)
    ]
    exact_madrid_primary = [
        m for m in local_matches
        if m["ticker"].endswith(".MC")
        and (
            m["ticker"].split(".")[0] == q_upper
            or _fold_text(m["name"]) == q_fold
            or f" {q_fold} " in f" {_fold_text(m.get('aliases', ''))} "
        )
    ]

    if exact_curated_us and not has_madrid_match:
        return exact_curated_us[0], [exact_curated_us[0]]

    if len(exact_madrid_primary) == 1 and len(exact_curated_us) == 0:
        # Check if there is also an exact NYSE/SEC ticker collision (e.g. SAN, BBVA, TEF, FER, IAG)
        sec_exact = [m for m in local_matches if not m["ticker"].endswith(".MC") and m["ticker"] == q_upper]
        if not sec_exact:
            # Verify whether SEC name ALSO explicitly matches (e.g. Santander/BBVA/Telefonica ADRs)
            sec_name_exact = [
                m for m in local_matches
                if not m["ticker"].endswith(".MC") and q_fold in _fold_text(m.get("name", ""))
            ]
            if not sec_name_exact:
                return exact_madrid_primary[0], [exact_madrid_primary[0]]

    # For international / dual-listed / partial queries, merge live Yahoo global search
    merged_pool = list(local_matches)
    should_call_yf = (
        not local_matches
        or has_madrid_match
        or len(q_fold) >= 3
    )
    if should_call_yf:
        for y_item in _yahoo_global_search(q, max_results=15):
            if y_item["ticker"] not in seen_tk:
                seen_tk.add(y_item["ticker"])
                merged_pool.append(y_item)
            else:
                # Propagate raw_exchange tag to local SEC match when helpful
                for existing in merged_pool:
                    if existing["ticker"] == y_item["ticker"] and not existing.get("raw_exchange"):
                        existing["raw_exchange"] = y_item.get("raw_exchange", "")

    if not merged_pool:
        return None, []

    merged_pool.sort(key=lambda item: _score_global_candidate(item, q_upper, q_fold))

    # Filter out obvious secondary/OTC duplicates if strong primary matches exist
    primary_candidates = [
        item for item in merged_pool
        if not (
            item["ticker"].endswith(SECONDARY_EXCHANGE_SUFFIXES)
            or item.get("raw_exchange", "") in SECONDARY_EXCHANGES
        )
    ]
    final_list = primary_candidates[:8] if primary_candidates else merged_pool[:8]
    return final_list[0], final_list


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
    """Classify financial headline sentiment using domain keyword lexicons (EN + ES)."""
    tokens = set(re.findall(r"[a-z]+", _fold_text(title)))
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


def _fetch_google_rss_feed(g_url: str, seen_prefixes: set, headers: dict) -> List[Dict]:
    """Parse a Google News RSS feed while deduplicating seen headlines."""
    out: List[Dict] = []
    try:
        resp = requests.get(g_url, headers=headers, timeout=8)
        if not resp.ok:
            return out
        root = ET.fromstring(resp.content)
        for item in root.findall(".//item"):
            raw_title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub_raw = (item.findtext("pubDate") or "").strip()
            source_el = item.findtext("source") or "Google News"

            if " - " in raw_title and raw_title.endswith(source_el):
                title = raw_title[: -(len(source_el) + 3)].strip()
            else:
                title = raw_title

            if not title:
                continue
            norm_key = re.sub(r"[^a-z0-9]", "", _fold_text(title))[:45]
            if norm_key in seen_prefixes:
                continue
            seen_prefixes.add(norm_key)
            ts, rel_time = _parse_rss_datetime(pub_raw)
            out.append({
                "title": title,
                "link": link,
                "source": source_el,
                "published_ts": ts,
                "published_rel": rel_time,
                "summary": "",
                "sentiment": _classify_headline_sentiment(title),
            })
    except Exception as e:
        logger.debug(f"Google RSS feed error ({g_url}): {e}")
    return out


@st.cache_data(ttl=300, show_spinner=False)
def get_company_news(ticker: str, company_name: str = "", limit: int = 18) -> List[Dict]:
    """Fetch & deduplicate real-time global/Spanish financial news from Yahoo & Google RSS."""
    tk = ticker.upper().strip()
    articles: List[Dict] = []
    seen_prefixes = set()

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ForwardGuidanceAI/2.0"}

    # Source 1: Yahoo Finance RSS (supports both US and international suffixed tickers like ITX.MC)
    try:
        y_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={requests.utils.quote(tk)}&region=US&lang=en-US"
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
                norm_key = re.sub(r"[^a-z0-9]", "", _fold_text(title))[:45]
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

    # Clean corporate suffixes for Google News queries
    clean_name = re.sub(
        r"\b(Inc\.?|Corp\.?|Corporation|plc|Co\.?|Holdings|S\.?A\.?|S\.?L\.?|SE|N\.?V\.?|AG)\b",
        "",
        company_name,
        flags=re.I,
    ).strip(" ,.-")
    tk_base = tk.split(".", 1)[0]
    search_label = clean_name if clean_name and clean_name.upper() != tk_base else tk_base

    # Source 2A: Spanish financial news feed when Bolsa de Madrid (.MC)
    if tk.endswith(".MC"):
        q_es = f'"{search_label}" (bolsa OR acciones OR resultados OR dividendo)'
        g_es_url = (
            f"https://news.google.com/rss/search?q={requests.utils.quote(q_es)}+when:21d"
            "&hl=es&gl=ES&ceid=ES:es"
        )
        articles.extend(_fetch_google_rss_feed(g_es_url, seen_prefixes, headers))

    # Source 2B: Global English financial news feed
    q_en = f"{tk} {search_label} stock".strip() if clean_name else f"{tk} stock"
    g_en_url = (
        f"https://news.google.com/rss/search?q={requests.utils.quote(q_en)}+when:14d"
        "&hl=en-US&gl=US&ceid=US:en"
    )
    articles.extend(_fetch_google_rss_feed(g_en_url, seen_prefixes, headers))

    articles.sort(key=lambda x: x["published_ts"], reverse=True)
    return articles[:limit]


@st.cache_data(ttl=900, show_spinner=False)
def get_recent_sec_events_cached(cik: str, limit: int = 12) -> List[Dict]:
    """Cached fetch of recent 10-K, 10-Q, 8-K SEC EDGAR events for Module A."""
    if not cik:
        return []
    return get_recent_sec_events(cik, limit=limit)


@st.cache_data(ttl=1800, show_spinner=False)
def get_global_corporate_events_cached(ticker: str, limit: int = 12) -> List[Dict]:
    """Fetch earnings calendar / results dates and market links for non-SEC international issuers."""
    tk = ticker.upper().strip()
    events: List[Dict] = []
    now_ts = pd.Timestamp.now()
    try:
        t = yf.Ticker(tk)
        ed = t.get_earnings_dates(limit=max(12, limit))
        if ed is not None and not ed.empty:
            for dt_idx, row in ed.iterrows():
                dt_naive = dt_idx.tz_localize(None) if getattr(dt_idx, "tzinfo", None) is not None else dt_idx
                is_future = dt_naive > now_ts
                eps_est = row.get("EPS Estimate")
                eps_rep = row.get("Reported EPS")
                surp = row.get("Surprise(%)")
                parts = []
                if pd.notna(eps_est):
                    parts.append(f"Est. EPS: {float(eps_est):.2f}")
                if pd.notna(eps_rep):
                    parts.append(f"EPS Real: {float(eps_rep):.2f}")
                if pd.notna(surp):
                    parts.append(f"Sorpresa: {float(surp):+.1f}%")
                events.append({
                    "date": dt_naive.strftime("%Y-%m-%d"),
                    "kind": "Próximos Resultados" if is_future else "Publicación de Resultados",
                    "detail": " · ".join(parts) if parts else "Presentación trimestral / semestral",
                    "url": f"https://finance.yahoo.com/quote/{requests.utils.quote(tk)}/financials",
                })
    except Exception as e:
        logger.debug(f"Global earnings events unavailable for {tk}: {e}")
    return events[:limit]
