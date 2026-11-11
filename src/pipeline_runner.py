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

    return "Universo S&P 500 Analizado", {}


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
    """Execute visible end-to-end SEC EDGAR -> MD&A Parser -> NLP/LLM -> Scorer pipeline."""
    tk = ticker.upper().strip()
    display_name = company_name or tk

    # Step 1: Resolve CIK & fetch filing list from SEC EDGAR if raw files missing or forcing refresh
    _emit(log_fn, f"[1/4] Verificando identidad regulatoria SEC EDGAR para {tk}...")
    resolved_cik = str(cik).zfill(10) if cik else ""
    if not resolved_cik:
        try:
            resolved_cik = get_cik(tk)
            _emit(log_fn, f"      CIK resuelto en SEC EDGAR: {resolved_cik}")
        except Exception as e:
            _emit(log_fn, f"      No se pudo obtener CIK desde SEC EDGAR ({e}).")
            resolved_cik = ""

    raw_dir = config.RAW_DIR / tk
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta_by_acc = _load_filings_meta(tk)

    existing_raw = sorted(
        f for f in raw_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in (".htm", ".html", ".txt")
    )

    if (len(existing_raw) == 0 or not meta_by_acc or force_reanalyze) and resolved_cik:
        _emit(log_fn, f"[2/4] Consultando informes 10-K y 10-Q en SEC EDGAR ({config.ANALYSIS_START_DATE} -> {config.ANALYSIS_END_DATE})...")
        all_filings_meta: List[dict] = []
        for ft in config.FILING_TYPES:
            found = get_filings_list(resolved_cik, ft)
            all_filings_meta.extend(found)
            _emit(log_fn, f"      • Detectados {len(found)} informes {ft} oficiales")

        # Keep most recent max_filings chronologically
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
    else:
        _emit(log_fn, f"[2/4] Reutilizando {len(existing_raw)} informes SEC en caché local para {tk}")

    # Step 3: Parse MD&A + Forward Looking & Run Sentiment Engine
    _emit(log_fn, f"[3/4] Extrayendo MD&A y ejecutando motor '{engine}'...")
    analyzed_records: List[dict] = []

    if existing_raw:
        # Sort by inferred filing_date ascending
        file_entries = []
        for fp in existing_raw:
            f_date, f_type = _infer_filing_date_and_type(fp, meta_by_acc)
            file_entries.append((f_date, f_type, fp))
        file_entries.sort(key=lambda x: x[0])

        # Cap to most recent max_filings if raw has many historical docs
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

            # Fallback to local NLP engine if Groq/Ollama fails (e.g. missing key / timeout)
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
        _emit(log_fn, f"      ⚠ No se encontraron filings descargables para {tk}; verificando registros procesados...")
        analyzed_records = load_ticker_guidance(tk)

    # Step 4: Score computation & Market price synchronization
    _emit(log_fn, f"[4/4] Calculando Guidance Score compuesto y sincronizando cotizaciones de {tk}...")
    scores_df = score_ticker_filings(tk, company_name=display_name, filings_guidance=analyzed_records)
    try:
        download_prices(tk)
    except Exception as pe:
        logger.debug(f"Price cache update skipped for {tk}: {pe}")

    _emit(log_fn, f"      ✓ Pipeline completado: {len(analyzed_records)} informes analizados y puntuados.")

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

    # Fetch S&P 500 (^GSPC) benchmark over identical period for Alpha vs Market
    try:
        sp_raw = yf.download("^GSPC", start="2023-09-01", progress=False, auto_adjust=False)
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
