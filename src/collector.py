"""
ForwardGuidanceAI - SEC EDGAR Filing Collector
================================================
Descarga filings 10-K y 10-Q de SEC EDGAR para las empresas del sector.
"""

import sys
import logging
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)

# ── SEC EDGAR Constants ─────────────────────────────────────────────
SEC_HEADERS = {
    "User-Agent": f"ForwardGuidanceAI {config.SEC_EDGAR_EMAIL or 'research@university.edu'}",
    "Accept-Encoding": "gzip, deflate",
}
SEC_RATE_LIMIT = 0.15  # seconds between requests (10 req/s max)

# Cache for CIK lookups
_cik_cache: dict[str, str] = {}


def get_cik(ticker: str) -> str:
    """Get the CIK (Central Index Key) for a ticker from SEC EDGAR.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')

    Returns:
        CIK zero-padded to 10 digits
    """
    ticker_upper = ticker.upper()

    if ticker_upper in _cik_cache:
        return _cik_cache[ticker_upper]

    url = "https://www.sec.gov/files/company_tickers.json"
    try:
        resp = requests.get(url, headers=SEC_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker_upper:
                cik = str(entry["cik_str"]).zfill(10)
                _cik_cache[ticker_upper] = cik
                logger.info(f"CIK for {ticker}: {cik}")
                return cik

        raise ValueError(f"Ticker '{ticker}' not found in SEC EDGAR")

    except requests.RequestException as e:
        logger.error(f"Error fetching CIK for {ticker}: {e}")
        raise


def get_filings_list(
    cik: str,
    filing_type: str = "10-K",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> list[dict]:
    """Get list of filings for a company from SEC EDGAR submissions API.

    Args:
        cik: CIK zero-padded to 10 digits
        filing_type: Type of filing ('10-K', '10-Q')
        start_date: Start date filter (YYYY-MM-DD)
        end_date: End date filter (YYYY-MM-DD)

    Returns:
        List of dicts with accession_number, filing_date, primary_document, form_type
    """
    start_date = start_date or config.ANALYSIS_START_DATE
    end_date = end_date or config.ANALYSIS_END_DATE

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"

    try:
        time.sleep(SEC_RATE_LIMIT)
        resp = requests.get(url, headers=SEC_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        recent = data.get("filings", {}).get("recent", {})
        if not recent:
            logger.warning(f"No filings found for CIK {cik}")
            return []

        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        documents = recent.get("primaryDocument", [])

        filings = []
        for i in range(len(forms)):
            form = forms[i]
            date = dates[i]

            # Filter by form type
            if form != filing_type:
                continue

            # Filter by date range
            if start_date and date < start_date:
                continue
            if end_date and date > end_date:
                continue

            filings.append({
                "accession_number": accessions[i],
                "filing_date": date,
                "primary_document": documents[i],
                "form_type": form,
            })

        logger.info(f"Found {len(filings)} {filing_type} filings for CIK {cik}")
        return filings

    except requests.RequestException as e:
        logger.error(f"Error fetching filings for CIK {cik}: {e}")
        return []


def download_filing(
    cik: str,
    accession_number: str,
    primary_document: str,
    save_dir: Path,
) -> Path:
    """Download a single filing document from SEC EDGAR.

    Args:
        cik: CIK (can be zero-padded or numeric)
        accession_number: Accession number (e.g., '0000320193-24-000123')
        primary_document: Primary document filename
        save_dir: Directory to save the file

    Returns:
        Path to the downloaded file
    """
    # Clean up accession number for URL
    acc_no_dashes = accession_number.replace("-", "")
    cik_numeric = cik.lstrip("0") or "0"

    # Build filename from accession number and document
    safe_filename = f"{accession_number}_{primary_document}".replace("/", "_")
    filepath = save_dir / safe_filename

    # Skip if already downloaded
    if filepath.exists() and filepath.stat().st_size > 0:
        logger.debug(f"Already downloaded: {filepath.name}")
        return filepath

    url = f"https://www.sec.gov/Archives/edgar/data/{cik_numeric}/{acc_no_dashes}/{primary_document}"

    try:
        time.sleep(SEC_RATE_LIMIT)
        resp = requests.get(url, headers=SEC_HEADERS, timeout=60)
        resp.raise_for_status()

        save_dir.mkdir(parents=True, exist_ok=True)
        filepath.write_bytes(resp.content)
        logger.info(f"Downloaded: {filepath.name} ({len(resp.content)} bytes)")
        return filepath

    except requests.RequestException as e:
        logger.error(f"Error downloading {url}: {e}")
        raise


def collect_all(
    tickers: Optional[list[str]] = None,
    filing_types: Optional[list[str]] = None,
) -> dict[str, list[str]]:
    """Collect all filings for all tickers.

    Args:
        tickers: List of tickers (defaults to config.TICKERS)
        filing_types: List of filing types (defaults to config.FILING_TYPES)

    Returns:
        Dict mapping ticker -> list of downloaded file paths (as strings)
    """
    tickers = tickers or list(config.TICKERS.keys())
    filing_types = filing_types or config.FILING_TYPES

    results: dict[str, list[str]] = {}

    for ticker in tickers:
        logger.info(f"Collecting filings for {ticker}...")

        try:
            cik = get_cik(ticker)
        except Exception as e:
            logger.error(f"Failed to get CIK for {ticker}: {e}")
            results[ticker] = []
            continue

        ticker_files = []

        for ft in filing_types:
            filings = get_filings_list(cik, ft)

            for filing in filings:
                save_dir = config.RAW_DIR / ticker
                try:
                    path = download_filing(
                        cik,
                        filing["accession_number"],
                        filing["primary_document"],
                        save_dir,
                    )
                    ticker_files.append(str(path))
                    logger.info(
                        f"  ✅ {ticker} | {filing['form_type']} | {filing['filing_date']}"
                    )
                except Exception as e:
                    logger.warning(
                        f"  ⚠️ {ticker} | {filing['form_type']} | {filing['filing_date']}: {e}"
                    )

        results[ticker] = ticker_files
        logger.info(f"Collected {len(ticker_files)} filings for {ticker}")

    # Save manifest
    manifest_path = config.RAW_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    total = sum(len(v) for v in results.values())
    logger.info(f"Total filings collected: {total} across {len(results)} companies")

    return results
