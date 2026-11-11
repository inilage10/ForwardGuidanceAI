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


def _extract_matching_filings(
    block: dict,
    target_forms: set[str],
    start_date: Optional[str],
    end_date: Optional[str],
) -> list[dict]:
    """Extract matching SEC filings from a submissions block."""
    forms = block.get("form", [])
    dates = block.get("filingDate", [])
    accessions = block.get("accessionNumber", [])
    documents = block.get("primaryDocument", [])
    descriptions = block.get("primaryDocDescription", [])
    items = block.get("items", [])

    matched = []
    for i in range(len(forms)):
        form = forms[i]
        date = dates[i]

        if form not in target_forms:
            continue
        if start_date and date < start_date:
            continue
        if end_date and date > end_date:
            continue

        matched.append({
            "accession_number": accessions[i],
            "filing_date": date,
            "primary_document": documents[i] if i < len(documents) else "",
            "form_type": form,
            "description": descriptions[i] if i < len(descriptions) else "",
            "items": items[i] if i < len(items) else "",
        })
    return matched


def get_filings_list(
    cik: str,
    filing_type: str = "10-K",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_archive_pages: int = 6,
) -> list[dict]:
    """Get list of filings for a company from SEC EDGAR submissions API.

    Handles paginated submission archives for frequent filers (e.g., JPM, GS).

    Args:
        cik: CIK zero-padded to 10 digits
        filing_type: Type of filing ('10-K', '10-Q', etc.)
        start_date: Start date filter (YYYY-MM-DD)
        end_date: End date filter (YYYY-MM-DD)
        max_archive_pages: Maximum paginated archive JSON files to inspect if needed

    Returns:
        List of dicts with accession_number, filing_date, primary_document, form_type
    """
    cik_padded = str(cik).zfill(10)
    start_date = start_date or config.ANALYSIS_START_DATE
    end_date = end_date or config.ANALYSIS_END_DATE
    target_forms = {filing_type}

    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"

    try:
        time.sleep(SEC_RATE_LIMIT)
        resp = requests.get(url, headers=SEC_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        recent = data.get("filings", {}).get("recent", {})
        if not recent:
            logger.warning(f"No filings found for CIK {cik_padded}")
            return []

        filings = _extract_matching_filings(recent, target_forms, start_date, end_date)

        # Follow paginated archives if the issuer files high volumes (e.g. structured notes)
        recent_dates = recent.get("filingDate", [])
        oldest_recent = min(recent_dates) if recent_dates else ""
        archive_files = data.get("filings", {}).get("files", [])

        if archive_files and (not start_date or oldest_recent > start_date):
            pages_checked = 0
            for arch in archive_files:
                if pages_checked >= max_archive_pages:
                    break
                f_from = arch.get("filingFrom", "")
                f_to = arch.get("filingTo", "")
                if start_date and f_to and f_to < start_date:
                    continue
                if end_date and f_from and f_from > end_date:
                    continue

                arch_name = arch.get("name", "")
                if not arch_name:
                    continue

                pages_checked += 1
                time.sleep(SEC_RATE_LIMIT)
                arch_resp = requests.get(
                    f"https://data.sec.gov/submissions/{arch_name}",
                    headers=SEC_HEADERS,
                    timeout=30,
                )
                if arch_resp.ok:
                    arch_data = arch_resp.json()
                    filings.extend(
                        _extract_matching_filings(arch_data, target_forms, start_date, end_date)
                    )

        # Deduplicate preserving order
        seen = set()
        unique_filings = []
        for f in filings:
            acc = f["accession_number"]
            if acc not in seen:
                seen.add(acc)
                unique_filings.append(f)

        logger.info(f"Found {len(unique_filings)} {filing_type} filings for CIK {cik_padded}")
        return unique_filings

    except requests.RequestException as e:
        logger.error(f"Error fetching filings for CIK {cik_padded}: {e}")
        return []


def get_recent_sec_events(cik: str, limit: int = 15) -> list[dict]:
    """Fetch most recent 10-K, 10-Q, and 8-K filings with direct SEC EDGAR URLs."""
    cik_padded = str(cik).zfill(10)
    cik_numeric = cik_padded.lstrip("0") or "0"
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    try:
        time.sleep(SEC_RATE_LIMIT)
        resp = requests.get(url, headers=SEC_HEADERS, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        recent = data.get("filings", {}).get("recent", {})
        if not recent:
            return []

        events = _extract_matching_filings(recent, {"10-K", "10-Q", "8-K"}, None, None)
        for ev in events:
            acc_no_dashes = ev["accession_number"].replace("-", "")
            doc = ev["primary_document"]
            ev["sec_url"] = f"https://www.sec.gov/Archives/edgar/data/{cik_numeric}/{acc_no_dashes}/{doc}"
        return events[:limit]
    except Exception as e:
        logger.warning(f"Error fetching recent SEC events for CIK {cik_padded}: {e}")
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
