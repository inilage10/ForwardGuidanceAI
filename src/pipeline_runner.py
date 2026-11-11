"""
ForwardGuidanceAI - Live Pipeline Runner & Isolated Backtesting
=================================================================
Orquesta la ejecucion automatica y visible del pipeline SEC EDGAR -> MD&A Parser ->
Motor NLP/LLM -> Scoring -> Validacion Bursatil tanto para empresas individuales
(incluyendo busquedas arbitrarias en tiempo real) como para grupos sectoriales.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.analyzer import analyze_filing
from src.backtester import calculate_ic, calculate_long_short, merge_scores_and_returns
from src.collector import download_filing, get_cik, get_filings_list
from src.market_data import calculate_returns_for_dates, calculate_sector_returns, download_prices
from src.parser import parse_filing
from src.scorer import (
    get_score_evolution,
    load_ticker_guidance,
    score_ticker_filings,
    sync_all_cached_scores,
)
from src.sector_manager import DEFAULT_SECTORS

logger = logging.getLogger(__name__)


def _emit(log_fn: Optional[Callable[[str], None]], message: str) -> None:
    logger.info(message)
    if log_fn is not None:
        try:
            log_fn(message)
        except Exception:
            pass


def find_company_sector_peers(ticker: str) -> Tuple[str, Dict[str, str]]:
    """Find the S&P 500 GICS sector name and peer dictionary for a ticker."""
    tk = ticker.upper().strip()
    for sector_name, companies in DEFAULT_SECTORS.items():
        if "Big Tech" in sector_name:
            continue
        if tk in companies:
            return sector_name, {t: info["name"] for t, info in companies.items()}

    if tk in DEFAULT_SECTORS.get("Tecnología (Big Tech - Dataset Base)", {}):
        comp = DEFAULT_SECTORS["Tecnología (Big Tech - Dataset Base)"]
        return "Tecnología (Big Tech - Dataset Base)", {t: info["name"] for t, info in comp.items()}

    if tk.endswith(".MC") and "IBEX 35 (Bolsa de Madrid)" in DEFAULT_SECTORS:
        ibex = DEFAULT_SECTORS["IBEX 35 (Bolsa de Madrid)"]
        return "IBEX 35 (Bolsa de Madrid)", {t: info["name"] for t, info in ibex.items()}

    return "Universo Global Analizado", {}


def inspect_ticker_pipeline_state(ticker: str) -> Dict:
    """Inspect cached guidance, raw filings, and market prices available for a ticker."""
    tk = ticker.upper().strip()
    guidance_records = load_ticker_guidance(tk)
    has_real_sec = any(
        r.get("source") == "sec_edgar" or "period" not in r
        for r in guidance_records
    )

    raw_dir = config.RAW_DIR / tk
    raw_files = (
        sorted(
            f for f in raw_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in (".htm", ".html", ".txt")
        )
        if raw_dir.exists()
        else []
    )
    prices_file = config.PRICES_DIR / f"{tk}_prices.csv"

    return {
        "ticker": tk,
        "guidance_count": len(guidance_records),
        "has_real_sec": has_real_sec,
        "raw_count": len(raw_files),
        "has_prices": prices_file.exists(),
        "needs_auto_run": len(guidance_records) == 0,
    }


def _load_filings_meta(ticker: str) -> Dict[str, dict]:
    meta_file = config.RAW_DIR / ticker.upper() / "filings_meta.json"
    if meta_file.exists():
        try:
            return json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_filings_meta(ticker: str, meta_dict: Dict[str, dict]) -> None:
    meta_dir = config.RAW_DIR / ticker.upper()
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_file = meta_dir / "filings_meta.json"
    meta_file.write_text(json.dumps(meta_dict, indent=2), encoding="utf-8")


def _infer_filing_date_and_type(filepath: Path, meta_by_acc: Dict[str, dict]) -> Tuple[str, str]:
    """Infer exact SEC filing date and form type (10-K / 10-Q) using metadata or filename."""
    fname = filepath.name
    acc_match = re.match(r"^(\d{10}-\d{2}-\d{6})_", fname)
    if acc_match:
        acc_num = acc_match.group(1)
        if acc_num in meta_by_acc:
            info = meta_by_acc[acc_num]
            return info.get("filing_date", "unknown"), info.get("form_type", "10-Q")

    low = fname.lower()
    form_type = "10-K" if ("10-k" in low or "10k" in low) else "10-Q"

    m1 = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", fname)
    m2 = re.search(r"[-_](20\d{2})(\d{2})(\d{2})", fname)
    m3 = re.search(r"(20\d{2})(\d{2})(\d{2})", fname)
    m4 = re.search(r"\d{10}-(\d{2})-\d{6}", fname)

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

    return filing_date, form_type


def _select_market_benchmark(ticker: str) -> Tuple[str, str]:
    """Select the appropriate regional market benchmark index for Alpha calculation."""
    tk_up = ticker.upper().strip()
    if tk_up.endswith(".MC"):
        return "^IBEX", "IBEX 35"
    if tk_up.endswith((".PA", ".DE", ".AS", ".MI", ".BR", ".LS")):
        return "^STOXX50E", "Euro Stoxx 50"
    if tk_up.endswith(".L"):
        return "^FTSE", "FTSE 100"
    return "^GSPC", "S&P 500"


def _extract_stmt_val(df: Optional[pd.DataFrame], row_names: List[str], col) -> Optional[float]:
    """Safely extract numeric value from yfinance financial statement row candidates."""
    if df is None or df.empty or col not in df.columns:
        return None
    for nm in row_names:
        if nm in df.index:
            val = df.loc[nm, col]
            if pd.notna(val):
                try:
                    return float(val)
                except Exception:
                    continue
    return None


def _match_earnings_release_date(period_dt: pd.Timestamp, past_earnings: List[pd.Timestamp]) -> str:
    """Match fiscal period end date to actual historical earnings release date."""
    p_naive = period_dt.tz_localize(None) if getattr(period_dt, "tzinfo", None) is not None else period_dt
    candidates = [
        ed for ed in past_earnings
        if 7 <= (ed - p_naive).days <= 95
    ]
    if candidates:
        best = min(candidates, key=lambda d: (d - p_naive).days)
        return best.strftime("%Y-%m-%d")
    est = p_naive + pd.Timedelta(days=38)
    today = pd.Timestamp.now().normalize()
    return min(est, today).strftime("%Y-%m-%d")


def _signal_from_delta(delta_val: Optional[float], thresh_strong: float = 0.10, thresh_mod: float = 0.025) -> Tuple[str, str, float]:
    """Convert numeric YoY/sequential financial delta into structured guidance fields."""
    if delta_val is None or np.isnan(delta_val):
        return "maintained", "weak", 0.55
    if delta_val >= thresh_strong:
        return "raised", "strong", 0.85
    if delta_val >= thresh_mod:
        return "positive", "moderate", 0.78
    if delta_val <= -thresh_strong:
        return "lowered", "strong", 0.85
    if delta_val <= -thresh_mod:
        return "negative", "moderate", 0.78
    return "maintained", "moderate", 0.65


def _run_global_quarterly_guidance(
    ticker: str,
    company_name: str,
    engine: str,
    api_key: Optional[str],
    force_reanalyze: bool,
    max_periods: int,
    log_fn: Optional[Callable[[str], None]],
) -> List[dict]:
    """Build quarterly Forward Guidance records for non-SEC global stocks (Bolsa de Madrid, Europe, etc.)."""
    tk = ticker.upper().strip()
    display_name = company_name or tk
    _emit(log_fn, f"[2/4] Descargando estados financieros trimestrales y consenso global de {tk}...")

    try:
        yf_tk = yf.Ticker(tk)
        info = yf_tk.info or {}
        qi = yf_tk.quarterly_income_stmt
        ai = yf_tk.income_stmt
        qc = yf_tk.quarterly_cashflow
        ed_df = yf_tk.get_earnings_dates(limit=16)
    except Exception as e:
        _emit(log_fn, f"      ⚠ Error consultando fundamentales globales para {tk}: {e}")
        return []

    now_ts = pd.Timestamp.now().normalize()
    past_earnings: List[pd.Timestamp] = []
    if ed_df is not None and not ed_df.empty:
        for dt_idx in ed_df.index:
            d_naive = dt_idx.tz_localize(None) if getattr(dt_idx, "tzinfo", None) is not None else dt_idx
            if d_naive <= now_ts:
                past_earnings.append(d_naive)

    # Combine quarterly columns first, supplement with annual fiscal dates if fewer than 4 quarters
    period_entries: List[Tuple[pd.Timestamp, pd.DataFrame, str]] = []
    if qi is not None and not qi.empty:
        for c in sorted(qi.columns):
            c_dt = pd.Timestamp(c)
            if c_dt <= now_ts:
                period_entries.append((c_dt, qi, "QTR"))

    if len(period_entries) < 4 and ai is not None and not ai.empty:
        existing_dates = {p[0].strftime("%Y-%m-%d") for p in period_entries}
        for c in sorted(ai.columns):
            c_dt = pd.Timestamp(c)
            if c_dt <= now_ts and c_dt.strftime("%Y-%m-%d") not in existing_dates:
                period_entries.append((c_dt, ai, "FY"))

    period_entries.sort(key=lambda x: x[0])
    if len(period_entries) > max_periods:
        period_entries = period_entries[-max_periods:]

    if not period_entries:
        _emit(log_fn, f"      ⚠ Sin series trimestrales publicadas en mercado global para {tk}.")
        return []

    _emit(log_fn, f"      • Detectados {len(period_entries)} cierres financieros oficiales ({tk})")
    _emit(log_fn, f"[3/4] Analizando fundamentales, márgenes y narrativa con motor '{engine}'...")

    rec_key = str(info.get("recommendationKey", "") or "").lower()
    target_px = info.get("targetMeanPrice")
    curr_px = info.get("currentPrice") or info.get("regularMarketPrice")
    upside = ((float(target_px) - float(curr_px)) / float(curr_px)) if (target_px and curr_px and float(curr_px) > 0) else None
    biz_summary = str(info.get("longBusinessSummary", "") or "")[:650]
    currency = str(info.get("currency", "EUR" if tk.endswith(".MC") else "USD"))

    analyzed_records: List[dict] = []
    proc_dir = config.PROCESSED_DIR / tk
    proc_dir.mkdir(parents=True, exist_ok=True)

    for idx, (p_dt, stmt_df, f_type) in enumerate(period_entries, start=1):
        rel_date = _match_earnings_release_date(p_dt, past_earnings)
        cache_file = proc_dir / f"{rel_date}_{f_type}_guidance.json"
        if cache_file.exists() and not force_reanalyze:
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                analyzed_records.append(cached)
                _emit(log_fn, f"      • [{idx}/{len(period_entries)}] {rel_date} ({f_type}) cargado desde caché")
                continue
            except Exception:
                pass

        rev = _extract_stmt_val(stmt_df, ["Total Revenue", "Operating Revenue"], p_dt)
        op_inc = _extract_stmt_val(stmt_df, ["Operating Income", "EBIT"], p_dt)
        ebitda = _extract_stmt_val(stmt_df, ["EBITDA", "Normalized EBITDA"], p_dt)
        net_inc = _extract_stmt_val(stmt_df, ["Net Income", "Net Income From Continuing Operation Net Minority Interest"], p_dt)
        gross_p = _extract_stmt_val(stmt_df, ["Gross Profit"], p_dt)
        capex = _extract_stmt_val(qc, ["Capital Expenditure", "Purchase Of PPE"], p_dt) if f_type == "QTR" else None

        # Look back for same-quarter prior year (~1 year ago) or previous period comparison
        prev_col = None
        for prev_dt, prev_df, prev_t in period_entries[: idx - 1]:
            if prev_t == f_type and abs((p_dt - prev_dt).days - 365) <= 45:
                prev_col = (prev_dt, prev_df)
                break
        if prev_col is None and idx >= 2:
            prev_col = (period_entries[idx - 2][0], period_entries[idx - 2][1])

        rev_growth = None
        eps_growth = None
        margin_delta = None
        op_margin = (op_inc / rev) if (op_inc is not None and rev and rev != 0) else None

        if prev_col is not None:
            p_prev, df_prev = prev_col
            prev_rev = _extract_stmt_val(df_prev, ["Total Revenue", "Operating Revenue"], p_prev)
            prev_op = _extract_stmt_val(df_prev, ["Operating Income", "EBIT"], p_prev)
            prev_net = _extract_stmt_val(df_prev, ["Net Income", "Net Income From Continuing Operation Net Minority Interest"], p_prev)
            if rev is not None and prev_rev and abs(prev_rev) > 0:
                rev_growth = (rev - prev_rev) / abs(prev_rev)
            if net_inc is not None and prev_net and abs(prev_net) > 0:
                eps_growth = (net_inc - prev_net) / abs(prev_net)
            if op_margin is not None and prev_op is not None and prev_rev and prev_rev != 0:
                margin_delta = op_margin - (prev_op / prev_rev)

        # Fallback to latest yfinance consensus growth metrics on most recent period
        if idx == len(period_entries):
            if rev_growth is None and info.get("revenueGrowth") is not None:
                rev_growth = float(info["revenueGrowth"])
            if eps_growth is None and info.get("earningsGrowth") is not None:
                eps_growth = float(info["earningsGrowth"])

        rev_dir, rev_mag, rev_conf = _signal_from_delta(rev_growth, thresh_strong=0.10, thresh_mod=0.02)
        mar_dir, mar_mag, mar_conf = _signal_from_delta(margin_delta, thresh_strong=0.025, thresh_mod=0.006)
        earn_dir, earn_mag, earn_conf = _signal_from_delta(eps_growth, thresh_strong=0.12, thresh_mod=0.03)

        capex_ratio = (abs(capex) / rev) if (capex is not None and rev and rev > 0) else None
        cap_dir, cap_mag, cap_conf = (
            ("raised", "moderate", 0.72)
            if (capex_ratio is not None and capex_ratio >= 0.06)
            else ("maintained", "moderate", 0.62)
        )

        # Demand & competitive position incorporate market/consensus strength
        consensus_score = 0.0
        if rec_key in ("strong_buy", "buy"):
            consensus_score += 0.08
        elif rec_key in ("underperform", "sell"):
            consensus_score -= 0.08
        if upside is not None:
            consensus_score += max(-0.08, min(0.08, upside * 0.4))

        demand_val = (rev_growth or 0.0) + consensus_score
        dem_dir, dem_mag, dem_conf = _signal_from_delta(demand_val, thresh_strong=0.12, thresh_mod=0.03)
        comp_val = (op_margin - 0.12 if op_margin is not None else 0.0) + consensus_score
        comp_dir, comp_mag, comp_conf = _signal_from_delta(comp_val, thresh_strong=0.10, thresh_mod=0.02)
        innov_dir, innov_mag, innov_conf = _signal_from_delta(rev_growth, thresh_strong=0.15, thresh_mod=0.04)

        rev_txt = f"{rev / 1e6:,.1f}M {currency}" if rev is not None else "N/D"
        op_txt = f"{op_margin * 100:.1f}%" if op_margin is not None else "N/D"
        gr_txt = f"{rev_growth * 100:+.1f}%" if rev_growth is not None else "N/D"
        eg_txt = f"{eps_growth * 100:+.1f}%" if eps_growth is not None else "N/D"
        md_txt = f"{margin_delta * 100:+.2f} pp" if margin_delta is not None else "estables"

        brief_narrative = (
            f"Resultados financieros de {display_name} ({tk}) correspondientes al cierre {p_dt.strftime('%Y-%m-%d')} "
            f"(publicado {rel_date}, {f_type}). "
            f"Ingresos reportados de {rev_txt} (variación interanual/comparativa {gr_txt}). "
            f"Margen operativo de {op_txt} (variación {md_txt}). "
            f"Beneficio neto variación {eg_txt}. "
            f"Perfil operativo: {biz_summary}"
        )

        sections = {
            "mda": brief_narrative,
            "forward_looking": (
                f"Consensus rating: {rec_key or 'hold'}. "
                + (f"Target upside {upside * 100:+.1f}% vs market price. " if upside is not None else "")
                + f"Revenue trend {gr_txt}, operating margin {op_txt}, net income momentum {eg_txt}."
            ),
            "risk_factors": "",
        }

        nlp_base = analyze_filing(
            ticker=tk,
            filing_date=rel_date,
            filing_type=f_type,
            sections=sections,
            model=engine,
            api_key=api_key,
            force_reanalyze=True,
        )
        if nlp_base is None and "nlp" not in engine.lower():
            nlp_base = analyze_filing(
                ticker=tk,
                filing_date=rel_date,
                filing_type=f_type,
                sections=sections,
                model="nlp_local",
                force_reanalyze=True,
            )

        # Anchor quantitative financial dimensions cleanly using reported figures while keeping LLM/NLP narrative
        guidance_dict = {
            "revenue_growth": {
                "direction": rev_dir,
                "magnitude": rev_mag,
                "confidence": rev_conf,
                "direct_quote": f"Ingresos de {rev_txt} con variación comparativa de {gr_txt} en el cierre {p_dt.strftime('%Y-%m-%d')}.",
            },
            "margin_outlook": {
                "direction": mar_dir,
                "magnitude": mar_mag,
                "confidence": mar_conf,
                "direct_quote": f"Margen operativo situado en {op_txt} (cambio {md_txt}).",
            },
            "capex_plans": {
                "direction": cap_dir,
                "magnitude": cap_mag,
                "confidence": cap_conf,
                "direct_quote": f"Inversión en CapEx / activos del {capex_ratio * 100:.1f}% sobre ingresos." if capex_ratio is not None else "Asignación de capital disciplinada según flujo operativo trimestral.",
            },
            "demand_outlook": {
                "direction": dem_dir,
                "magnitude": dem_mag,
                "confidence": dem_conf,
                "direct_quote": f"Demanda e impulso comercial reflejados en ventas ({gr_txt}) y consenso de analistas ({rec_key or 'neutral'}).",
            },
            "earnings_guidance": {
                "direction": earn_dir,
                "magnitude": earn_mag,
                "confidence": earn_conf,
                "direct_quote": f"Evolución del resultado neto atribuible de {eg_txt} en el periodo {p_dt.strftime('%Y-%m-%d')}.",
            },
            "innovation_pipeline": {
                "direction": innov_dir,
                "magnitude": innov_mag,
                "confidence": innov_conf,
                "direct_quote": (nlp_base or {}).get("guidance", {}).get("innovation_pipeline", {}).get("direct_quote")
                or f"Posicionamiento sectorial en {info.get('industry') or info.get('sector') or 'su segmento principal'}.",
            },
            "competitive_position": {
                "direction": comp_dir,
                "magnitude": comp_mag,
                "confidence": comp_conf,
                "direct_quote": (f"Potencial objetivo de consenso de {upside * 100:+.1f}% ({rec_key.upper()})" if upside is not None else f"Rentabilidad operativa de {op_txt} dentro del sector."),
            },
        }

        dir_vals = []
        map_v = {"raised": 1.0, "positive": 0.6, "maintained": 0.0, "negative": -0.6, "lowered": -1.0}
        for d_val in guidance_dict.values():
            dir_vals.append(map_v.get(d_val["direction"], 0.0))
        quant_sent = float(np.mean(dir_vals)) if dir_vals else 0.0
        nlp_sent = float((nlp_base or {}).get("overall_sentiment", 0.0))
        overall_sent = float(round(0.75 * quant_sent + 0.25 * nlp_sent, 4))

        record = {
            "ticker": tk,
            "filing_date": rel_date,
            "filing_type": f_type,
            "guidance": guidance_dict,
            "overall_sentiment": overall_sent,
            "key_risks": (nlp_base or {}).get("key_risks") or ["Sensibilidad macroeconómica, divisas y evolución de márgenes operativos"],
            "summary": (
                f"{display_name} ({tk}): ingresos de {rev_txt} ({gr_txt}), margen operativo de {op_txt} ({md_txt}) "
                f"y variación de beneficio neto de {eg_txt}."
            ),
            "source": "global_fundamentals",
            "analysis_engine": f"{engine} · Fundamentales Globales",
        }
        cache_file.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        analyzed_records.append(record)
        _emit(log_fn, f"      ✓ [{idx}/{len(period_entries)}] {rel_date} ({f_type}) analizado · Sentiment: {overall_sent:+.2f}")

    return analyzed_records


def run_company_guidance_pipeline(
    ticker: str,
    company_name: str = "",
    cik: str = "",
    engine: str = "nlp_local",
    api_key: Optional[str] = None,
    force_reanalyze: bool = False,
    max_filings: int = 10,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict:
    """Execute visible SEC EDGAR -> MD&A -> NLP/LLM pipeline, with global quarterly fundamentals fallback."""
    tk = ticker.upper().strip()
    display_name = company_name or tk

    # Step 1: Resolve CIK & fetch filing list from SEC EDGAR when applicable
    _emit(log_fn, f"[1/4] Verificando cobertura regulatoria para {tk}...")
    resolved_cik = str(cik).zfill(10) if cik else ""
    if not resolved_cik and "." not in tk:
        try:
            resolved_cik = get_cik(tk)
            _emit(log_fn, f"      CIK resuelto en SEC EDGAR: {resolved_cik}")
        except Exception:
            resolved_cik = ""

    raw_dir = config.RAW_DIR / tk
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta_by_acc = _load_filings_meta(tk)

    existing_raw = sorted(
        f for f in raw_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in (".htm", ".html", ".txt")
    )

    if (len(existing_raw) == 0 or not meta_by_acc or force_reanalyze) and resolved_cik:
        _emit(log_fn, f"[2/4] Consultando informes 10-K / 10-Q / 20-F en SEC EDGAR ({config.ANALYSIS_START_DATE} -> {config.ANALYSIS_END_DATE})...")
        all_filings_meta: List[dict] = []
        for ft in config.FILING_TYPES:
            found = get_filings_list(resolved_cik, ft)
            all_filings_meta.extend(found)
            _emit(log_fn, f"      • Detectados {len(found)} informes {ft} oficiales")

        # Support SEC Foreign Private Issuers (20-F) when 10-K/10-Q absent
        if not all_filings_meta:
            found_20f = get_filings_list(resolved_cik, "20-F")
            if found_20f:
                all_filings_meta.extend(found_20f)
                _emit(log_fn, f"      • Detectados {len(found_20f)} informes anuales 20-F (emisor extranjero SEC)")

        all_filings_meta.sort(key=lambda x: x.get("filing_date", ""), reverse=True)
        selected_meta = all_filings_meta[:max_filings]

        for f_item in selected_meta:
            meta_by_acc[f_item["accession_number"]] = f_item
        _save_filings_meta(tk, meta_by_acc)

        if len(existing_raw) == 0 or force_reanalyze:
            for idx, f_item in enumerate(selected_meta, start=1):
                try:
                    path = download_filing(
                        resolved_cik,
                        f_item["accession_number"],
                        f_item["primary_document"],
                        raw_dir,
                    )
                    _emit(
                        log_fn,
                        f"      ✓ [{idx}/{len(selected_meta)}] Descargado {f_item['form_type']} · {f_item['filing_date']} ({path.name})",
                    )
                except Exception as dl_err:
                    _emit(log_fn, f"      ⚠ Omitido {f_item['filing_date']}: {dl_err}")

        existing_raw = sorted(
            f for f in raw_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in (".htm", ".html", ".txt")
        )
    elif existing_raw:
        _emit(log_fn, f"[2/4] Reutilizando {len(existing_raw)} informes SEC en caché local para {tk}")
    else:
        _emit(log_fn, f"      • {tk} cotiza en bolsa internacional fuera de SEC EDGAR · activando motor global.")

    # Step 3: Parse SEC MD&A OR execute Global Quarterly Fundamentals + Consensus Pipeline
    analyzed_records: List[dict] = []

    if existing_raw:
        _emit(log_fn, f"[3/4] Extrayendo MD&A y ejecutando motor '{engine}'...")
        file_entries = []
        for fp in existing_raw:
            f_date, f_type = _infer_filing_date_and_type(fp, meta_by_acc)
            file_entries.append((f_date, f_type, fp))
        file_entries.sort(key=lambda x: x[0])

        if len(file_entries) > max_filings:
            file_entries = file_entries[-max_filings:]

        for idx, (f_date, f_type, fp) in enumerate(file_entries, start=1):
            cache_file = config.PROCESSED_DIR / tk / f"{f_date}_{f_type}_guidance.json"
            if cache_file.exists() and not force_reanalyze:
                try:
                    cached = json.loads(cache_file.read_text(encoding="utf-8"))
                    analyzed_records.append(cached)
                    _emit(log_fn, f"      • [{idx}/{len(file_entries)}] {f_date} ({f_type}) cargado desde caché")
                    continue
                except Exception:
                    pass

            sections = parse_filing(fp)
            res = analyze_filing(
                ticker=tk,
                filing_date=f_date,
                filing_type=f_type,
                sections=sections,
                model=engine,
                api_key=api_key,
                force_reanalyze=force_reanalyze,
            )

            if res is None and "nlp" not in engine.lower():
                _emit(log_fn, f"      ⚠ Fallback a motor NLP financiero local para {f_date} ({f_type})")
                res = analyze_filing(
                    ticker=tk,
                    filing_date=f_date,
                    filing_type=f_type,
                    sections=sections,
                    model="nlp_local",
                    force_reanalyze=True,
                )

            if res:
                analyzed_records.append(res)
                sent_val = res.get("overall_sentiment", 0.0)
                _emit(log_fn, f"      ✓ [{idx}/{len(file_entries)}] {f_date} ({f_type}) analizado · Sentiment: {sent_val:+.2f}")

            if "groq" in engine.lower():
                time.sleep(0.25)
    else:
        analyzed_records = _run_global_quarterly_guidance(
            ticker=tk,
            company_name=display_name,
            engine=engine,
            api_key=api_key,
            force_reanalyze=force_reanalyze,
            max_periods=min(max_filings, 8),
            log_fn=log_fn,
        )

    # Step 4: Score computation & Market price synchronization
    _emit(log_fn, f"[4/4] Calculando Guidance Score compuesto y sincronizando cotizaciones de {tk}...")
    scores_df = score_ticker_filings(tk, company_name=display_name, filings_guidance=analyzed_records)
    try:
        download_prices(tk)
    except Exception as pe:
        logger.debug(f"Price cache update skipped for {tk}: {pe}")

    _emit(log_fn, f"      ✓ Pipeline completado: {len(analyzed_records)} periodos analizados y puntuados.")

    return {
        "ticker": tk,
        "company_name": display_name,
        "cik": resolved_cik,
        "engine": engine,
        "guidance_records": load_ticker_guidance(tk),
        "scores_df": scores_df,
    }


def run_company_event_backtest(
    ticker: str,
    horizon_days: int = 90,
    score_col: str = "composite_score",
    custom_weights: Optional[dict[str, float]] = None,
) -> Dict:
    """Run an isolated event-study backtest on a single company's SEC filings vs S&P 500."""
    tk = ticker.upper().strip()
    records = load_ticker_guidance(tk)
    if not records:
        return {"error": f"No hay registros de Forward Guidance para {tk}."}

    # Ensure scores DataFrame exists for this ticker (reflecting custom weights if provided)
    ticker_scores = score_ticker_filings(tk, filings_guidance=records, custom_weights=custom_weights)
    if ticker_scores.empty:
        return {"error": f"No se pudieron calcular scores para {tk}."}

    prices_df = download_prices(tk)
    if prices_df.empty:
        return {"error": f"No se pudieron obtener cotizaciones históricas para {tk}."}

    filing_dates = (
        pd.to_datetime(ticker_scores["filing_date"], errors="coerce")
        .dropna()
        .dt.strftime("%Y-%m-%d")
        .unique()
        .tolist()
    )
    ret_df = calculate_returns_for_dates(prices_df, filing_dates, windows=[30, 90, 180])
    if ret_df.empty:
        return {"error": f"No hay ventanas de retorno suficientes posteriores a los filings de {tk}."}

    # Fetch regional market benchmark (^IBEX for Madrid, ^STOXX50E for Eurozone, ^GSPC for US/Global)
    bench_sym, bench_name = _select_market_benchmark(tk)
    try:
        sp_raw = yf.download(bench_sym, start="2023-09-01", progress=False, auto_adjust=False)
        if isinstance(sp_raw.columns, pd.MultiIndex):
            sp_raw.columns = [c[0] for c in sp_raw.columns]
        sp_df = sp_raw.reset_index()
        sp_df["Date"] = pd.to_datetime(sp_df["Date"]).dt.tz_localize(None)
        sp_ret_df = calculate_returns_for_dates(sp_df, filing_dates, windows=[30, 90, 180])
    except Exception:
        sp_ret_df = pd.DataFrame()

    t_copy = ticker_scores.copy()
    t_copy["filing_date"] = pd.to_datetime(t_copy["filing_date"]).dt.strftime("%Y-%m-%d")
    merged = pd.merge(t_copy, ret_df, on="filing_date", how="inner")

    if not sp_ret_df.empty:
        sp_cols = {"filing_date": "filing_date"}
        for w in [30, 90, 180]:
            if f"return_{w}d" in sp_ret_df.columns:
                sp_cols[f"return_{w}d"] = f"sp500_return_{w}d"
        sp_renamed = sp_ret_df[list(sp_cols.keys())].rename(columns=sp_cols)
        merged = pd.merge(merged, sp_renamed, on="filing_date", how="left")
        for w in [30, 90, 180]:
            r_col = f"return_{w}d"
            s_col = f"sp500_return_{w}d"
            if r_col in merged.columns and s_col in merged.columns:
                merged[f"alpha_{w}d"] = merged[r_col] - merged[s_col]

    ret_col = f"return_{horizon_days}d"
    alpha_col = f"alpha_{horizon_days}d"
    eval_col = alpha_col if alpha_col in merged.columns and merged[alpha_col].notna().sum() >= 3 else ret_col

    valid = merged[[score_col, ret_col]].dropna() if ret_col in merged.columns else pd.DataFrame()
    valid_alpha = merged[[score_col, eval_col]].dropna() if eval_col in merged.columns else pd.DataFrame()

    ic, p_val = (0.0, 1.0)
    if len(valid_alpha) >= 4:
        rho, pv = stats.spearmanr(valid_alpha[score_col], valid_alpha[eval_col])
        ic = float(rho) if not np.isnan(rho) else 0.0
        p_val = float(pv) if not np.isnan(pv) else 1.0

    avg_ret = float(valid[ret_col].mean()) if not valid.empty else 0.0
    avg_alpha = float(merged[alpha_col].dropna().mean()) if alpha_col in merged.columns and not merged[alpha_col].dropna().empty else 0.0

    # Win rate when signal >= 0 (or median)
    pos_signals = merged[merged[score_col] >= 0].dropna(subset=[ret_col]) if ret_col in merged.columns else pd.DataFrame()
    win_rate = float((pos_signals[ret_col] > 0).mean()) if not pos_signals.empty else 0.0

    # Spread: top half score vs bottom half score
    spread = 0.0
    if len(valid) >= 4:
        med = valid[score_col].median()
        high_grp = valid[valid[score_col] >= med][ret_col]
        low_grp = valid[valid[score_col] < med][ret_col]
        if not high_grp.empty and not low_grp.empty:
            spread = float(high_grp.mean() - low_grp.mean())

    return {
        "ticker": tk,
        "horizon_days": horizon_days,
        "score_col": score_col,
        "benchmark_symbol": bench_sym,
        "benchmark_name": bench_name,
        "sample_size": len(valid),
        "ic": ic,
        "p_value": p_val,
        "avg_return": avg_ret,
        "avg_alpha": avg_alpha,
        "win_rate": win_rate,
        "high_low_spread": spread,
        "events_df": merged.sort_values("filing_date"),
        "prices_df": prices_df,
    }


def run_sector_cross_backtest(
    ticker: str,
    horizon_days: int = 90,
    score_col: str = "composite_score",
) -> Dict:
    """Run cross-sectional peer backtest for the company's GICS sector (or all analyzed firms)."""
    sector_name, peers = find_company_sector_peers(ticker)

    # Ensure scores.csv contains every analyzed company on disk
    all_scores = sync_all_cached_scores()
    if all_scores.empty:
        return {"error": "No hay scores calculados en el sistema."}

    if peers:
        peer_scores = all_scores[all_scores["ticker"].isin(list(peers.keys()))].copy()
        scope_label = f"Sector {sector_name}"
    else:
        peer_scores = all_scores.copy()
        scope_label = "Universo Completo Analizado"

    if peer_scores.empty:
        peer_scores = all_scores.copy()
        scope_label = "Universo Completo Analizado"

    analyzed_tickers = sorted(peer_scores["ticker"].unique().tolist())

    # Load or download market prices for the peer group
    all_prices: Dict[str, pd.DataFrame] = {}
    for tk in analyzed_tickers:
        df_p = download_prices(tk)
        if not df_p.empty:
            all_prices[tk] = df_p

    returns_df = calculate_sector_returns(all_prices=all_prices, scores_df=peer_scores)
    merged_df = merge_scores_and_returns(peer_scores, returns_df)

    if merged_df.empty:
        return {
            "error": "No se pudieron emparejar filings con retornos de mercado para el grupo.",
            "scope_label": scope_label,
            "analyzed_tickers": analyzed_tickers,
            "missing_peers": [t for t in peers if t not in analyzed_tickers],
        }

    alpha_col = f"alpha_{horizon_days}d"
    target_col = alpha_col if alpha_col in merged_df.columns and merged_df[alpha_col].notna().sum() >= 5 else f"return_{horizon_days}d"

    ic_res = calculate_ic(merged_df, score_col=score_col, return_col=target_col)
    ls_res = calculate_long_short(merged_df, score_col=score_col, return_col=target_col, n_quantiles=3)

    return {
        "scope_label": scope_label,
        "sector_name": sector_name,
        "horizon_days": horizon_days,
        "score_col": score_col,
        "target_col": target_col,
        "analyzed_tickers": analyzed_tickers,
        "all_peers": list(peers.keys()) if peers else analyzed_tickers,
        "missing_peers": [t for t in peers if t not in analyzed_tickers] if peers else [],
        "ic": ic_res,
        "long_short": ls_res,
        "merged_df": merged_df,
    }
