"""
ForwardGuidanceAI - Backtesting Engine
======================================
Evalúa si la señal de forward guidance (score de IA) anticipa un
comportamiento bursátil superior dentro del sector.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.market_data import calculate_sector_returns, download_all_prices

logger = logging.getLogger(__name__)


def merge_scores_and_returns(
    scores_df: pd.DataFrame,
    returns_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge guidance scores with future returns on ticker and filing date.

    Args:
        scores_df: DataFrame with ticker, filing_date, scores
        returns_df: DataFrame with ticker, filing_date, returns, alphas

    Returns:
        Merged DataFrame with both guidance scores and stock market returns
    """
    if scores_df.empty or returns_df.empty:
        return pd.DataFrame()

    s_df = scores_df.copy()
    r_df = returns_df.copy()

    s_df["filing_date"] = pd.to_datetime(s_df["filing_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    r_df["filing_date"] = pd.to_datetime(r_df["filing_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    merged = pd.merge(s_df, r_df, on=["ticker", "filing_date"], how="inner")
    logger.info(f"Merged {len(merged)} score-return records")
    return merged


def calculate_ic(
    merged_df: pd.DataFrame,
    score_col: str = "composite_score",
    return_col: str = "alpha_90d",
) -> dict:
    """Calculate Information Coefficient (Spearman rank correlation) between score and subsequent return.

    Args:
        merged_df: Dataframe with score and return columns
        score_col: Name of score column
        return_col: Name of forward return or alpha column

    Returns:
        Dict with overall ic, p_value, and ic_by_period
    """
    valid = merged_df[[score_col, return_col, "filing_date"]].dropna()
    if len(valid) < 5:
        return {"ic": 0.0, "p_value": 1.0, "sample_size": len(valid), "ic_by_period": []}

    # Overall Spearman correlation
    rho, p_val = stats.spearmanr(valid[score_col], valid[return_col])

    # Quarterly IC breakdown
    valid["quarter"] = pd.to_datetime(valid["filing_date"]).dt.to_period("Q").astype(str)
    period_ics = []
    for q, grp in valid.groupby("quarter"):
        if len(grp) >= 3:
            q_rho, q_p = stats.spearmanr(grp[score_col], grp[return_col])
            if not np.isnan(q_rho):
                period_ics.append({"period": q, "ic": float(q_rho), "n": len(grp)})

    return {
        "ic": float(rho) if not np.isnan(rho) else 0.0,
        "p_value": float(p_val) if not np.isnan(p_val) else 1.0,
        "sample_size": len(valid),
        "ic_by_period": period_ics,
    }


def calculate_long_short(
    merged_df: pd.DataFrame,
    score_col: str = "composite_score",
    return_col: str = "alpha_90d",
    n_quantiles: int = 3,
) -> dict:
    """Evaluate performance dividing companies into top/middle/bottom tiers per period.

    Args:
        merged_df: Merged scores and returns dataframe
        score_col: Score column name
        return_col: Return or alpha column name
        n_quantiles: Quantile count (default 3: Top/Middle/Bottom)

    Returns:
        Dict with quantile_returns, long_short_spread, and hit_rate
    """
    valid = merged_df[["ticker", "filing_date", score_col, return_col]].dropna().copy()
    if len(valid) < n_quantiles:
        return {
            "quantile_returns": {},
            "long_short_spread": 0.0,
            "hit_rate": 0.0,
        }

    valid["quarter"] = pd.to_datetime(valid["filing_date"]).dt.to_period("Q").astype(str)

    period_spreads = []
    all_top_returns = []
    all_bottom_returns = []
    all_mid_returns = []

    for q, grp in valid.groupby("quarter"):
        if len(grp) < n_quantiles:
            continue
        try:
            labels = ["Bottom", "Middle", "Top"] if n_quantiles == 3 else [f"Q{i+1}" for i in range(n_quantiles)]
            grp = grp.copy()
            grp["quantile"] = pd.qcut(grp[score_col], q=n_quantiles, labels=labels, duplicates="drop")

            q_means = grp.groupby("quantile", observed=False)[return_col].mean()
            top_ret = q_means.get("Top" if n_quantiles == 3 else labels[-1], np.nan)
            bot_ret = q_means.get("Bottom" if n_quantiles == 3 else labels[0], np.nan)

            if not np.isnan(top_ret) and not np.isnan(bot_ret):
                period_spreads.append(top_ret - bot_ret)
                all_top_returns.append(top_ret)
                all_bottom_returns.append(bot_ret)
                if n_quantiles == 3 and "Middle" in q_means:
                    all_mid_returns.append(q_means["Middle"])
        except Exception as e:
            logger.debug(f"Skipping period {q} in long-short calculation: {e}")

    top_avg = float(np.mean(all_top_returns)) if all_top_returns else 0.0
    bot_avg = float(np.mean(all_bottom_returns)) if all_bottom_returns else 0.0
    mid_avg = float(np.mean(all_mid_returns)) if all_mid_returns else 0.0
    spread = top_avg - bot_avg

    hit_rate = (
        float(np.mean([s > 0 for s in period_spreads]))
        if period_spreads
        else 0.0
    )

    return {
        "quantile_returns": {
            "Top": top_avg,
            "Middle": mid_avg,
            "Bottom": bot_avg,
        },
        "long_short_spread": spread,
        "hit_rate": hit_rate,
        "n_periods": len(period_spreads),
    }


def generate_backtest_summary(results: dict) -> str:
    """Generate a markdown report summarizing backtest results."""
    lines = [
        "### 📌 Resumen Ejecutivo del Backtest",
        "",
        "Se ha analizado la capacidad predictiva del **Forward Guidance Score** "
        "generado por IA frente a los retornos bursátiles posteriores de las 10 empresas tecnológicas.",
        "",
        "| Ventana | Information Coeff (IC) | p-value | Significativo? | Long-Short Spread | Hit Rate |",
        "|---|---|---|---|---|---|",
    ]

    for w in config.RETURN_WINDOWS:
        w_key = f"window_{w}d"
        data = results.get(w_key, {})
        ic_info = data.get("ic", {})
        ls_info = data.get("long_short", {})

        ic_val = ic_info.get("ic", 0.0)
        p_val = ic_info.get("p_value", 1.0)
        sig = "✅ Sí (p<0.05)" if p_val < 0.05 else "❌ No (p≥0.05)"
        spread = ls_info.get("long_short_spread", 0.0)
        hit_rate = ls_info.get("hit_rate", 0.0)

        lines.append(
            f"| **{w} días** | `{ic_val:+.3f}` | `{p_val:.4f}` | {sig} | `{spread:+.2%}` | `{hit_rate:.1%}` |"
        )

    lines.extend([
        "",
        "#### Conclusiones Metodológicas:",
        "- **Information Coefficient (IC):** Mide la correlación ordinal entre el ranking de expectativas y el ranking de retornos relativos.",
        "- **Long-Short Spread:** Representa el exceso de rentabilidad acumulado que habría obtenido una estrategia larga en el tercil con mejor guidance y corta en el peor tercil.",
        "- **Riesgos y límites identificados:** El tamaño muestral histórico y la ambigüedad en el lenguaje directivo pueden introducir ruido en ventanas cortas (30 días), observándose típicamente mayor estabilidad en horizontes trimestrales (90 días).",
    ])

    return "\n".join(lines)


def run_backtest(
    scores_df: Optional[pd.DataFrame] = None,
    returns_df: Optional[pd.DataFrame] = None,
) -> dict:
    """Run full backtest analysis across all windows and save results to disk.

    Args:
        scores_df: Scores DataFrame (loaded from disk if None)
        returns_df: Returns DataFrame (calculated if None)

    Returns:
        Comprehensive results dictionary
    """
    if scores_df is None:
        scores_path = config.SCORES_DIR / "scores.csv"
        if not scores_path.exists():
            raise FileNotFoundError("scores.csv not found. Please run scoring first.")
        scores_df = pd.read_csv(scores_path)

    if returns_df is None:
        logger.info("Calculating sector returns...")
        returns_df = calculate_sector_returns(scores_df=scores_df)

    merged_df = merge_scores_and_returns(scores_df, returns_df)

    # Save merged data for UI exploration
    config.SCORES_DIR.mkdir(parents=True, exist_ok=True)
    merged_path = config.SCORES_DIR / "merged_scores_returns.csv"
    merged_df.to_csv(merged_path, index=False)

    results = {}
    for w in config.RETURN_WINDOWS:
        alpha_col = f"alpha_{w}d"
        if alpha_col not in merged_df.columns:
            continue

        ic_res = calculate_ic(merged_df, score_col="composite_score", return_col=alpha_col)
        ls_res = calculate_long_short(merged_df, score_col="composite_score", return_col=alpha_col, n_quantiles=3)

        results[f"window_{w}d"] = {
            "window_days": w,
            "ic": ic_res,
            "long_short": ls_res,
        }

    results["summary"] = generate_backtest_summary(results)

    # Save results to json
    results_path = config.SCORES_DIR / "backtest_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Backtest completed and saved to {results_path}")
    return results
