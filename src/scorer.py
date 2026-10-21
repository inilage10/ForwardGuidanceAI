"""
ForwardGuidanceAI - Guidance Scoring Engine
=============================================
Convierte el guidance extraído por el LLM en scores numéricos
comparables entre empresas del sector.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)

# ── Mapping Tables ──────────────────────────────────────────────────
DIRECTION_MAP = {
    "positive": 1.0,
    "negative": -1.0,
    "stable": 0.0,
    "unclear": 0.0,
    "increasing": 0.8,
    "decreasing": -0.8,
    "improving": 0.8,
    "declining": -0.8,
    "strong": 1.0,
    "weak": -0.8,
    "neutral": 0.0,
}

MAGNITUDE_MAP = {
    "high": 1.0,
    "moderate": 0.6,
    "low": 0.3,
    "significant": 0.9,
    "slight": 0.2,
    "substantial": 0.85,
    "marginal": 0.15,
}


def score_dimension(dimension_data: dict) -> float:
    """Score a single guidance dimension.

    Score = direction_value * magnitude_value * confidence

    Args:
        dimension_data: Dict with 'direction', 'magnitude', 'confidence'

    Returns:
        Float score in [-1, 1]
    """
    direction = str(dimension_data.get("direction", "unclear")).lower()
    magnitude = str(dimension_data.get("magnitude", "moderate")).lower()
    confidence = dimension_data.get("confidence", 0.5)

    try:
        confidence = float(confidence)
        confidence = max(0.0, min(1.0, confidence))
    except (ValueError, TypeError):
        confidence = 0.5

    d_val = DIRECTION_MAP.get(direction, 0.0)
    m_val = MAGNITUDE_MAP.get(magnitude, 0.5)

    score = d_val * m_val * confidence
    return max(-1.0, min(1.0, score))


def score_company(guidance: dict) -> dict:
    """Score all guidance dimensions for a single filing.

    Args:
        guidance: Full guidance dict from analyzer

    Returns:
        Dict with dimension_scores, composite_score, overall_sentiment
    """
    guidance_data = guidance.get("guidance", {})
    dimension_scores = {}

    for dim in config.GUIDANCE_DIMENSIONS:
        if dim in guidance_data and isinstance(guidance_data[dim], dict):
            dimension_scores[dim] = score_dimension(guidance_data[dim])
        else:
            dimension_scores[dim] = 0.0

    # Composite score: mean of all dimension scores
    scores = list(dimension_scores.values())
    composite = np.mean(scores) if scores else 0.0

    # Overall sentiment from LLM
    overall_sentiment = guidance.get("overall_sentiment", 0.0)
    try:
        overall_sentiment = float(overall_sentiment)
    except (ValueError, TypeError):
        overall_sentiment = 0.0

    return {
        "dimension_scores": dimension_scores,
        "composite_score": float(composite),
        "overall_sentiment": float(overall_sentiment),
    }


def score_all(all_guidance: Optional[dict] = None) -> pd.DataFrame:
    """Score all companies and filings.

    Args:
        all_guidance: Output from analyzer.analyze_all().
                      If None, loads from all_guidance.json.

    Returns:
        DataFrame with scores for all filings
    """
    if all_guidance is None:
        guidance_path = config.PROCESSED_DIR / "all_guidance.json"
        if not guidance_path.exists():
            logger.error("No guidance data found. Run analyzer first.")
            return pd.DataFrame()

        with open(guidance_path, "r", encoding="utf-8") as f:
            all_guidance = json.load(f)

    rows = []

    for ticker, filings in all_guidance.items():
        for filing in filings:
            if not isinstance(filing, dict):
                continue

            scores = score_company(filing)

            row = {
                "ticker": ticker,
                "company_name": config.TICKERS.get(ticker, ticker),
                "filing_date": filing.get("filing_date", "unknown"),
                "filing_type": filing.get("filing_type", "unknown"),
                "composite_score": scores["composite_score"],
                "overall_sentiment": scores["overall_sentiment"],
            }

            # Add individual dimension scores
            for dim, score in scores["dimension_scores"].items():
                row[dim] = score

            rows.append(row)

    if not rows:
        logger.warning("No scores calculated")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")

    # Calculate sector rank and percentile per period
    for period, group in df.groupby(df["filing_date"].dt.to_period("Q")):
        idx = group.index
        df.loc[idx, "sector_rank"] = group["composite_score"].rank(ascending=False)
        df.loc[idx, "sector_percentile"] = group["composite_score"].rank(pct=True)

    # Sort by date and score
    df = df.sort_values(["filing_date", "composite_score"], ascending=[True, False])

    # Save to CSV
    config.SCORES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.SCORES_DIR / "scores.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Scores saved to {output_path} ({len(df)} rows)")

    return df


def get_latest_scores() -> pd.DataFrame:
    """Get the most recent score for each company.

    Returns:
        DataFrame sorted by composite_score descending
    """
    scores_path = config.SCORES_DIR / "scores.csv"
    if not scores_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(scores_path)
    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")

    latest = df.sort_values("filing_date").groupby("ticker").last().reset_index()
    latest = latest.sort_values("composite_score", ascending=False)

    return latest


def get_score_evolution(ticker: str) -> pd.DataFrame:
    """Get the score time series for a specific company.

    Args:
        ticker: Stock ticker

    Returns:
        DataFrame with score evolution
    """
    scores_path = config.SCORES_DIR / "scores.csv"
    if not scores_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(scores_path)
    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")

    ticker_df = df[df["ticker"] == ticker].sort_values("filing_date")
    return ticker_df
