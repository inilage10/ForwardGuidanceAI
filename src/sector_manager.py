"""
ForwardGuidanceAI - S&P 500 Sector & Market Cap Manager
=======================================================
Permite seleccionar cualquier sector del S&P 500 y obtiene dinámicamente
las 10 empresas líderes por capitalización bursátil.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

SECTOR_CONFIG_FILE = Path(__file__).resolve().parent.parent / "data" / "active_sector.json"

# Curated Top 10 by Market Cap for all 11 GICS S&P 500 sectors + Big Tech benchmark
DEFAULT_SECTORS: Dict[str, Dict[str, dict]] = {
    "Tecnología (Big Tech - Dataset Base)": {
        "AAPL": {"name": "Apple", "cik": "0000320193", "cap": "3.55T"},
        "MSFT": {"name": "Microsoft", "cik": "0000789019", "cap": "3.15T"},
        "NVDA": {"name": "NVIDIA", "cik": "0001045810", "cap": "3.38T"},
        "GOOGL": {"name": "Alphabet", "cik": "0001652044", "cap": "2.10T"},
        "AMZN": {"name": "Amazon", "cik": "0001018724", "cap": "1.98T"},
        "META": {"name": "Meta Platforms", "cik": "0001326801", "cap": "1.52T"},
        "CRM": {"name": "Salesforce", "cik": "0001108524", "cap": "285B"},
        "ADBE": {"name": "Adobe", "cik": "0000796343", "cap": "215B"},
        "AMD": {"name": "AMD", "cik": "0000002488", "cap": "250B"},
        "INTC": {"name": "Intel", "cik": "0000050863", "cap": "95B"},
    },
    "Information Technology": {
        "AAPL": {"name": "Apple", "cik": "0000320193", "cap": "3.55T"},
        "NVDA": {"name": "NVIDIA", "cik": "0001045810", "cap": "3.38T"},
        "MSFT": {"name": "Microsoft", "cik": "0000789019", "cap": "3.15T"},
        "AVGO": {"name": "Broadcom", "cik": "0001730168", "cap": "830B"},
        "ORCL": {"name": "Oracle", "cik": "0001341439", "cap": "480B"},
        "CRM": {"name": "Salesforce", "cik": "0001108524", "cap": "285B"},
        "AMD": {"name": "AMD", "cik": "0000002488", "cap": "250B"},
        "CSCO": {"name": "Cisco Systems", "cik": "0000858877", "cap": "230B"},
        "ADBE": {"name": "Adobe", "cik": "0000796343", "cap": "215B"},
        "ACN": {"name": "Accenture", "cik": "0001467373", "cap": "220B"},
    },
    "Financials": {
        "JPM": {"name": "JPMorgan Chase", "cik": "0000019617", "cap": "640B"},
        "BAC": {"name": "Bank of America", "cik": "0000070858", "cap": "330B"},
        "WFC": {"name": "Wells Fargo", "cik": "0000072971", "cap": "205B"},
        "MS": {"name": "Morgan Stanley", "cik": "0000895421", "cap": "195B"},
        "GS": {"name": "Goldman Sachs", "cik": "0000886982", "cap": "180B"},
        "BLK": {"name": "BlackRock", "cik": "0001364742", "cap": "150B"},
        "C": {"name": "Citigroup", "cik": "0000831001", "cap": "130B"},
        "SCHW": {"name": "Charles Schwab", "cik": "0000316709", "cap": "140B"},
        "AXP": {"name": "American Express", "cik": "0000004962", "cap": "200B"},
        "PGR": {"name": "Progressive Corp", "cik": "0000080661", "cap": "145B"},
    },
    "Health Care": {
        "LLY": {"name": "Eli Lilly", "cik": "0000059478", "cap": "850B"},
        "UNH": {"name": "UnitedHealth Group", "cik": "0000731766", "cap": "530B"},
        "JNJ": {"name": "Johnson & Johnson", "cik": "0000200406", "cap": "385B"},
        "ABBV": {"name": "AbbVie", "cik": "0001551152", "cap": "330B"},
        "MRK": {"name": "Merck & Co", "cik": "0000310158", "cap": "270B"},
        "TMO": {"name": "Thermo Fisher Scientific", "cik": "0000097745", "cap": "210B"},
        "ABT": {"name": "Abbott Laboratories", "cik": "0000001800", "cap": "200B"},
        "DHR": {"name": "Danaher Corp", "cik": "0000313616", "cap": "185B"},
        "PFE": {"name": "Pfizer", "cik": "0000078003", "cap": "165B"},
        "ISRG": {"name": "Intuitive Surgical", "cik": "0001035267", "cap": "180B"},
    },
    "Consumer Discretionary": {
        "AMZN": {"name": "Amazon", "cik": "0001018724", "cap": "1.98T"},
        "TSLA": {"name": "Tesla", "cik": "0001318605", "cap": "750B"},
        "HD": {"name": "Home Depot", "cik": "0000354950", "cap": "395B"},
        "MCD": {"name": "McDonald's", "cik": "0000063908", "cap": "215B"},
        "NKE": {"name": "Nike", "cik": "0000320187", "cap": "125B"},
        "LOW": {"name": "Lowe's Companies", "cik": "0000060667", "cap": "145B"},
        "BKNG": {"name": "Booking Holdings", "cik": "0001075531", "cap": "160B"},
        "SBUX": {"name": "Starbucks", "cik": "0000829224", "cap": "110B"},
        "TJX": {"name": "TJX Companies", "cik": "0000109198", "cap": "135B"},
        "TGT": {"name": "Target Corp", "cik": "0000027419", "cap": "70B"},
    },
    "Communication Services": {
        "GOOGL": {"name": "Alphabet", "cik": "0001652044", "cap": "2.10T"},
        "META": {"name": "Meta Platforms", "cik": "0001326801", "cap": "1.52T"},
        "NFLX": {"name": "Netflix", "cik": "0001065280", "cap": "310B"},
        "DIS": {"name": "Walt Disney", "cik": "0001744489", "cap": "185B"},
        "CMCSA": {"name": "Comcast", "cik": "0001166691", "cap": "155B"},
        "TMUS": {"name": "T-Mobile US", "cik": "0001283699", "cap": "240B"},
        "VZ": {"name": "Verizon Communications", "cik": "0000732712", "cap": "175B"},
        "T": {"name": "AT&T", "cik": "0000932784", "cap": "150B"},
        "EA": {"name": "Electronic Arts", "cik": "0000712515", "cap": "40B"},
        "WBD": {"name": "Warner Bros Discovery", "cik": "0001437107", "cap": "20B"},
    },
    "Energy": {
        "XOM": {"name": "Exxon Mobil", "cik": "0000034088", "cap": "480B"},
        "CVX": {"name": "Chevron", "cik": "0000093410", "cap": "280B"},
        "COP": {"name": "ConocoPhillips", "cik": "0001163165", "cap": "125B"},
        "EOG": {"name": "EOG Resources", "cik": "0000821189", "cap": "72B"},
        "SLB": {"name": "Schlumberger", "cik": "0000087347", "cap": "65B"},
        "MPC": {"name": "Marathon Petroleum", "cik": "0001510295", "cap": "58B"},
        "PSX": {"name": "Phillips 66", "cik": "0001534701", "cap": "56B"},
        "VLO": {"name": "Valero Energy", "cik": "0001035002", "cap": "45B"},
        "WMB": {"name": "Williams Companies", "cik": "0000107263", "cap": "62B"},
        "OXY": {"name": "Occidental Petroleum", "cik": "0000797468", "cap": "48B"},
    },
    "Industrials": {
        "GE": {"name": "GE Aerospace", "cik": "0000040545", "cap": "205B"},
        "CAT": {"name": "Caterpillar", "cik": "0000018230", "cap": "190B"},
        "UNP": {"name": "Union Pacific", "cik": "0000100885", "cap": "145B"},
        "HON": {"name": "Honeywell", "cik": "0000077384", "cap": "135B"},
        "RTX": {"name": "RTX Corp", "cik": "0000101829", "cap": "165B"},
        "BA": {"name": "Boeing", "cik": "0000012927", "cap": "95B"},
        "DE": {"name": "Deere & Co", "cik": "0000315189", "cap": "115B"},
        "LMT": {"name": "Lockheed Martin", "cik": "0000936468", "cap": "135B"},
        "UPS": {"name": "United Parcel Service", "cik": "0001090727", "cap": "110B"},
        "ETN": {"name": "Eaton Corp", "cik": "0001553350", "cap": "140B"},
    },
    "Consumer Staples": {
        "WMT": {"name": "Walmart", "cik": "0000104169", "cap": "670B"},
        "PG": {"name": "Procter & Gamble", "cik": "0000080424", "cap": "395B"},
        "COST": {"name": "Costco Wholesale", "cik": "0000909832", "cap": "400B"},
        "KO": {"name": "Coca-Cola", "cik": "0000021344", "cap": "290B"},
        "PEP": {"name": "PepsiCo", "cik": "0000077476", "cap": "235B"},
        "PM": {"name": "Philip Morris", "cik": "0001413329", "cap": "200B"},
        "MDLZ": {"name": "Mondelez International", "cik": "0001103982", "cap": "90B"},
        "CL": {"name": "Colgate-Palmolive", "cik": "0000021665", "cap": "80B"},
        "MO": {"name": "Altria Group", "cik": "0000764180", "cap": "95B"},
        "MNST": {"name": "Monster Beverage", "cik": "0000865752", "cap": "50B"},
    },
    "Utilities": {
        "NEE": {"name": "NextEra Energy", "cik": "0000753308", "cap": "165B"},
        "SO": {"name": "Southern Co", "cik": "0000092122", "cap": "98B"},
        "DUK": {"name": "Duke Energy", "cik": "0001326160", "cap": "88B"},
        "CEG": {"name": "Constellation Energy", "cik": "0001868275", "cap": "82B"},
        "SRE": {"name": "Sempra Energy", "cik": "0001032208", "cap": "55B"},
        "AEP": {"name": "American Electric Power", "cik": "0000004904", "cap": "52B"},
        "VST": {"name": "Vistra Corp", "cik": "0001692819", "cap": "45B"},
        "D": {"name": "Dominion Energy", "cik": "0000715957", "cap": "48B"},
        "PEG": {"name": "Public Service Enterprise", "cik": "0000788784", "cap": "44B"},
        "EXC": {"name": "Exelon Corp", "cik": "0001109357", "cap": "40B"},
    },
    "Real Estate": {
        "PLD": {"name": "Prologis", "cik": "0001045609", "cap": "115B"},
        "AMT": {"name": "American Tower", "cik": "0001053507", "cap": "95B"},
        "EQIX": {"name": "Equinix", "cik": "0001101239", "cap": "88B"},
        "WELL": {"name": "Welltower", "cik": "0000766704", "cap": "75B"},
        "SPG": {"name": "Simon Property Group", "cik": "0001063761", "cap": "58B"},
        "DLR": {"name": "Digital Realty Trust", "cik": "0001297996", "cap": "52B"},
        "PSA": {"name": "Public Storage", "cik": "0001393311", "cap": "56B"},
        "O": {"name": "Realty Income", "cik": "0000726728", "cap": "48B"},
        "CCI": {"name": "Crown Castle", "cik": "0001051470", "cap": "46B"},
        "CBRE": {"name": "CBRE Group", "cik": "0001138301", "cap": "38B"},
    },
    "Materials": {
        "LIN": {"name": "Linde plc", "cik": "0001707925", "cap": "215B"},
        "SHW": {"name": "Sherwin-Williams", "cik": "0000089800", "cap": "95B"},
        "FCX": {"name": "Freeport-McMoRan", "cik": "0000831259", "cap": "65B"},
        "APD": {"name": "Air Products & Chemicals", "cik": "0000002969", "cap": "68B"},
        "ECL": {"name": "Ecolab", "cik": "0000031462", "cap": "70B"},
        "NEM": {"name": "Newmont Corp", "cik": "0001164727", "cap": "52B"},
        "CTVA": {"name": "Corteva", "cik": "0001755672", "cap": "40B"},
        "DOW": {"name": "Dow Inc", "cik": "0001751788", "cap": "36B"},
        "DD": {"name": "DuPont de Nemours", "cik": "0001666700", "cap": "34B"},
        "NUE": {"name": "Nucor Corp", "cik": "0000073309", "cap": "35B"},
    },
}


def get_available_sectors() -> List[str]:
    """Return list of available sectors."""
    return list(DEFAULT_SECTORS.keys())


def load_active_sector() -> str:
    """Load the current active sector from disk or session."""
    if SECTOR_CONFIG_FILE.exists():
        try:
            data = json.loads(SECTOR_CONFIG_FILE.read_text(encoding="utf-8"))
            sec = data.get("active_sector")
            if sec in DEFAULT_SECTORS:
                return sec
        except Exception as e:
            logger.warning(f"Error loading active sector: {e}")
    return "Tecnología (Big Tech - Dataset Base)"


def save_active_sector(sector_name: str) -> None:
    """Persist active sector selection to disk."""
    if sector_name not in DEFAULT_SECTORS:
        return
    try:
        SECTOR_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        SECTOR_CONFIG_FILE.write_text(
            json.dumps({"active_sector": sector_name}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as e:
        logger.error(f"Failed to save active sector: {e}")


def get_active_tickers(sector_name: Optional[str] = None) -> Dict[str, str]:
    """Return a mapping of Ticker -> Company Name for the given or active sector."""
    sec = sector_name or load_active_sector()
    companies = DEFAULT_SECTORS.get(sec, DEFAULT_SECTORS["Tecnología (Big Tech - Dataset Base)"])
    return {ticker: info["name"] for ticker, info in companies.items()}


def get_active_sector_details(sector_name: Optional[str] = None) -> List[dict]:
    """Return full detailed list of top 10 companies for sector table display."""
    sec = sector_name or load_active_sector()
    companies = DEFAULT_SECTORS.get(sec, DEFAULT_SECTORS["Tecnología (Big Tech - Dataset Base)"])
    records = []
    for rank, (ticker, info) in enumerate(companies.items(), start=1):
        records.append({
            "Rank": rank,
            "Ticker": ticker,
            "Empresa": info["name"],
            "Market Cap Estimado": info.get("cap", "N/A"),
            "CIK SEC": info.get("cik", ""),
        })
    return records


def render_sector_sidebar() -> str:
    """Render universal sector selector in Streamlit sidebar across all pages.
    
    Returns:
        The currently active sector name.
    """
    sectors = get_available_sectors()
    current = load_active_sector()
    default_idx = sectors.index(current) if current in sectors else 0

    with st.sidebar:
        st.markdown("### 🌐 Sector del S&P 500")
        selected = st.selectbox(
            "Selecciona Sector",
            options=sectors,
            index=default_idx,
            key="global_sp500_sector_selector",
            help="Cambia el sector para analizar sus 10 empresas líderes por capitalización bursátil.",
        )

        if selected != current:
            save_active_sector(selected)
            st.session_state["active_sector"] = selected
            st.rerun()

        # Display badge of active companies
        tickers_map = get_active_tickers(selected)
        badge_text = ", ".join(list(tickers_map.keys())[:5]) + ", ..."
        st.caption(f"Top 10: `{badge_text}`")
        st.divider()

    return selected
