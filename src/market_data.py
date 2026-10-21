"""
ForwardGuidanceAI - Market Data Module
======================================
Descarga de precios históricos con yfinance y cálculo de retornos
y alphas relativos al sector.
"""

import sys
import time
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


def download_prices(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download historical price data for a ticker and cache to disk.

    Args:
        ticker: Stock ticker (e.g. 'AAPL')
        start_date: Start date (YYYY-MM-DD), default config.ANALYSIS_START_DATE
        end_date: End date (YYYY-MM-DD), default config.ANALYSIS_END_DATE + buffer
        force_refresh: If True, ignore cached CSV

    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Adj Close, Volume
    """
    start_date = start_date or config.ANALYSIS_START_DATE

    # Add 180 days buffer to end_date if not specified to allow post-filing return calculation
    if not end_date:
        end_dt = datetime.strptime(config.ANALYSIS_END_DATE, "%Y-%m-%d") + timedelta(days=200)
        end_date = end_dt.strftime("%Y-%m-%d")

    cache_file = config.PRICES_DIR / f"{ticker}_prices.csv"
    if cache_file.exists() and not force_refresh:
        try:
            df = pd.read_csv(cache_file)
            df["Date"] = pd.to_datetime(df["Date"])
            logger.info(f"Loaded cached prices for {ticker} ({len(df)} rows)")
            return df
        except Exception as e:
            logger.warning(f"Error reading cached prices for {ticker}: {e}, re-downloading...")

    logger.info(f"Downloading prices for {ticker} ({start_date} to {end_date})...")
    try:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False,
        )

        if data.empty:
            logger.warning(f"No price data received for {ticker}")
            return pd.DataFrame()

        # Flatten multi-level columns if present (common in newer yfinance versions)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]

        data = data.reset_index()
        data["Date"] = pd.to_datetime(data["Date"])

        # Keep standard columns
        cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in data.columns]
        data = data[cols]

        config.PRICES_DIR.mkdir(parents=True, exist_ok=True)
        data.to_csv(cache_file, index=False)
        logger.info(f"Saved prices for {ticker} to {cache_file} ({len(data)} rows)")
        return data

    except Exception as e:
        logger.error(f"Error downloading prices for {ticker}: {e}")
        return pd.DataFrame()


def download_all_prices(
    tickers: Optional[list[str]] = None,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Download historical prices for all configured tickers.

    Args:
        tickers: List of tickers (defaults to config.TICKERS.keys())
        force_refresh: Force redownload

    Returns:
        Dict mapping ticker -> DataFrame of prices
    """
    tickers = tickers or list(config.TICKERS.keys())
    results = {}

    for i, ticker in enumerate(tickers):
        logger.info(f"Fetching prices [{i+1}/{len(tickers)}]: {ticker}")
        df = download_prices(ticker, force_refresh=force_refresh)
        if not df.empty:
            results[ticker] = df
        time.sleep(0.4)  # Rate limiting for Yahoo

    return results


def calculate_returns_for_dates(
    prices_df: pd.DataFrame,
    filing_dates: list[str],
    windows: Optional[list[int]] = None,
) -> pd.DataFrame:
    """Calculate forward returns from specific filing dates over specified windows.

    Args:
        prices_df: DataFrame containing at least 'Date' and 'Close' (or 'Adj Close')
        filing_dates: List of filing dates (strings YYYY-MM-DD or datetimes)
        windows: List of calendar day windows, defaults to config.RETURN_WINDOWS (e.g. [30, 90, 180])

    Returns:
        DataFrame with columns: filing_date, return_{w}d for each window
    """
    windows = windows or config.RETURN_WINDOWS
    if prices_df.empty:
        return pd.DataFrame()

    df = prices_df.copy()
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.sort_values("Date").reset_index(drop=True)

    records = []
    for f_date_raw in filing_dates:
        f_date = pd.to_datetime(f_date_raw)
        if pd.isna(f_date):
            continue

        # Find closest trading day on or after filing date
        future_prices = df[df["Date"] >= f_date]
        if future_prices.empty:
            continue

        start_row = future_prices.iloc[0]
        start_price = float(start_row[price_col])
        actual_start_date = start_row["Date"]

        row = {
            "filing_date": f_date.strftime("%Y-%m-%d"),
            "trade_start_date": actual_start_date.strftime("%Y-%m-%d"),
            "start_price": start_price,
        }

        for w in windows:
            target_date = actual_start_date + timedelta(days=w)
            window_prices = df[df["Date"] >= target_date]
            if not window_prices.empty:
                end_price = float(window_prices.iloc[0][price_col])
                ret = (end_price - start_price) / start_price
                row[f"return_{w}d"] = ret
            else:
                row[f"return_{w}d"] = np.nan

        records.append(row)

    return pd.DataFrame(records)


def calculate_sector_returns(
    all_prices: Optional[dict[str, pd.DataFrame]] = None,
    scores_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Calculate forward returns and sector-relative excess returns (alphas) for all filings.

    Args:
        all_prices: Dict of ticker -> prices df. If None, loaded from disk.
        scores_df: DataFrame with at least ticker and filing_date. If None, loaded from scores.csv.

    Returns:
        DataFrame containing ticker, filing_date, returns, and alpha metrics.
    """
    if scores_df is None:
        scores_path = config.SCORES_DIR / "scores.csv"
        if not scores_path.exists():
            logger.error("No scores found. Cannot determine filing dates.")
            return pd.DataFrame()
        scores_df = pd.read_csv(scores_path)

    if all_prices is None:
        all_prices = {}
        for ticker in config.TICKERS:
            p_file = config.PRICES_DIR / f"{ticker}_prices.csv"
            if p_file.exists():
                all_prices[ticker] = pd.read_csv(p_file)
            else:
                df = download_prices(ticker)
                if not df.empty:
                    all_prices[ticker] = df

    all_company_returns = []
    for ticker, group in scores_df.groupby("ticker"):
        if ticker not in all_prices or all_prices[ticker].empty:
            continue
        p_df = all_prices[ticker]
        f_dates = group["filing_date"].dropna().unique().tolist()
        ret_df = calculate_returns_for_dates(p_df, f_dates)
        if not ret_df.empty:
            ret_df["ticker"] = ticker
            all_company_returns.append(ret_df)

    if not all_company_returns:
        return pd.DataFrame()

    combined_returns = pd.concat(all_company_returns, ignore_index=True)
    combined_returns["filing_date_dt"] = pd.to_datetime(combined_returns["filing_date"])

    # Calculate sector average returns by period (quarter)
    combined_returns["quarter"] = combined_returns["filing_date_dt"].dt.to_period("Q")

    for w in config.RETURN_WINDOWS:
        ret_col = f"return_{w}d"
        alpha_col = f"alpha_{w}d"
        if ret_col in combined_returns.columns:
            # Quarter average return across sector
            sector_avg = combined_returns.groupby("quarter")[ret_col].transform("mean")
            combined_returns[alpha_col] = combined_returns[ret_col] - sector_avg

    combined_returns = combined_returns.drop(columns=["quarter", "filing_date_dt"], errors="ignore")
    return combined_returns
