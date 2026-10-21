"""
ForwardGuidanceAI - Filing Document Parser
=============================================
Parsea filings HTML de la SEC y extrae secciones relevantes
(MD&A, Risk Factors, Forward-Looking Statements).
"""

import re
import logging
import json
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


def clean_html(html_content: str) -> str:
    """Parse HTML filing and extract clean text.

    Args:
        html_content: Raw HTML content of a filing

    Returns:
        Cleaned text content
    """
    try:
        soup = BeautifulSoup(html_content, "lxml")
    except Exception:
        soup = BeautifulSoup(html_content, "html.parser")

    # Remove non-content tags
    for tag in soup.find_all(["script", "style", "meta", "link", "head"]):
        tag.decompose()

    # Get text
    text = soup.get_text(separator="\n")

    # Clean up whitespace
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if line:
            lines.append(line)

    return "\n".join(lines)


def extract_mda_section(text: str) -> Optional[str]:
    """Extract Management's Discussion and Analysis section.

    Looks for Item 7 (10-K) or Item 2 (10-Q) sections.

    Args:
        text: Clean text from filing

    Returns:
        Extracted MD&A section text, or None if not found
    """
    # Patterns for start of MD&A section
    start_patterns = [
        r"(?i)item\s+7[\.\s]*[\-—]*\s*management.s\s+discussion\s+and\s+analysis",
        r"(?i)item\s+2[\.\s]*[\-—]*\s*management.s\s+discussion\s+and\s+analysis",
        r"(?i)management.s\s+discussion\s+and\s+analysis\s+of\s+financial\s+condition",
        r"(?i)ITEM\s+7\b",
        r"(?i)ITEM\s+2\b.*management",
    ]

    # Patterns for end of MD&A section
    end_patterns = [
        r"(?i)item\s+7a[\.\s]*[\-—]*\s*quantitative",
        r"(?i)item\s+8[\.\s]*[\-—]*\s*financial\s+statements",
        r"(?i)item\s+3[\.\s]*[\-—]*\s*quantitative",
        r"(?i)item\s+4[\.\s]*[\-—]*\s*controls",
        r"(?i)ITEM\s+7A\b",
        r"(?i)ITEM\s+8\b",
        r"(?i)ITEM\s+3\b(?!.*legal)",
    ]

    for start_pat in start_patterns:
        start_match = re.search(start_pat, text)
        if start_match:
            start_pos = start_match.start()
            remaining = text[start_pos:]

            # Find end
            best_end = len(remaining)
            for end_pat in end_patterns:
                end_match = re.search(end_pat, remaining[500:])  # Skip at least 500 chars
                if end_match:
                    end_pos = end_match.start() + 500
                    if end_pos < best_end:
                        best_end = end_pos

            section = remaining[:best_end].strip()
            if len(section) > 200:  # Must be substantial
                logger.info(f"Found MD&A section: {len(section)} chars")
                return section

    logger.warning("MD&A section not found")
    return None


def extract_forward_looking(text: str) -> Optional[str]:
    """Extract forward-looking statements and outlook sections.

    Looks for paragraphs containing forward-looking keywords
    with surrounding context.

    Args:
        text: Clean text from filing

    Returns:
        Extracted forward-looking text, or None
    """
    keywords = [
        "forward-looking",
        "outlook",
        "guidance",
        "expects",
        "anticipates",
        "projects",
        "forecasts",
        "estimates that",
        "believes that",
        "plans to",
        "intends to",
        "future",
        "next fiscal year",
        "next quarter",
        "going forward",
        "we expect",
        "we anticipate",
        "we project",
    ]

    paragraphs = text.split("\n")
    relevant_indices = set()

    for i, para in enumerate(paragraphs):
        para_lower = para.lower()
        if any(kw in para_lower for kw in keywords):
            # Include surrounding paragraphs for context
            for j in range(max(0, i - 1), min(len(paragraphs), i + 2)):
                relevant_indices.add(j)

    if relevant_indices:
        sorted_indices = sorted(relevant_indices)
        sections = [paragraphs[i] for i in sorted_indices if paragraphs[i].strip()]
        result = "\n".join(sections)
        if len(result) > 100:
            logger.info(f"Found forward-looking content: {len(result)} chars")
            return result

    logger.warning("Forward-looking statements not found")
    return None


def extract_risk_factors(text: str) -> Optional[str]:
    """Extract Risk Factors section (Item 1A).

    Args:
        text: Clean text from filing

    Returns:
        Extracted risk factors text, or None
    """
    start_patterns = [
        r"(?i)item\s+1a[\.\s]*[\-—]*\s*risk\s+factors",
        r"(?i)ITEM\s+1A\b",
        r"(?i)risk\s+factors",
    ]

    end_patterns = [
        r"(?i)item\s+1b[\.\s]*[\-—]*\s*unresolved",
        r"(?i)item\s+2[\.\s]*[\-—]*\s*properties",
        r"(?i)ITEM\s+1B\b",
        r"(?i)ITEM\s+2\b",
    ]

    for start_pat in start_patterns:
        start_match = re.search(start_pat, text)
        if start_match:
            start_pos = start_match.start()
            remaining = text[start_pos:]

            best_end = min(len(remaining), 20000)  # Cap risk factors
            for end_pat in end_patterns:
                end_match = re.search(end_pat, remaining[200:])
                if end_match:
                    end_pos = end_match.start() + 200
                    if end_pos < best_end:
                        best_end = end_pos

            section = remaining[:best_end].strip()
            if len(section) > 100:
                logger.info(f"Found Risk Factors section: {len(section)} chars")
                return section

    logger.warning("Risk Factors section not found")
    return None


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to approximately max_tokens (rough estimate: 4 chars/token).

    Args:
        text: Text to truncate
        max_tokens: Maximum token count

    Returns:
        Truncated text
    """
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[... truncated ...]"


def parse_filing(filepath: Path, max_tokens: int = 6000) -> dict:
    """Parse a filing and extract all relevant sections.

    Args:
        filepath: Path to the filing document
        max_tokens: Max tokens per section (approx 4 chars/token)

    Returns:
        Dict with keys: mda, forward_looking, risk_factors, full_text_excerpt
    """
    logger.info(f"Parsing filing: {filepath.name}")

    # Read file with encoding fallback
    content = None
    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            content = filepath.read_text(encoding=encoding)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue

    if content is None:
        logger.error(f"Could not read file: {filepath}")
        return {"mda": None, "forward_looking": None, "risk_factors": None, "full_text_excerpt": None}

    # Detect if HTML or plain text
    if "<html" in content.lower() or "<body" in content.lower() or "<!doctype" in content.lower():
        text = clean_html(content)
    else:
        text = content

    # Extract sections
    mda = extract_mda_section(text)
    forward_looking = extract_forward_looking(text)
    risk_factors = extract_risk_factors(text)

    # Truncate sections
    sections = {
        "mda": _truncate_to_tokens(mda, max_tokens) if mda else None,
        "forward_looking": _truncate_to_tokens(forward_looking, max_tokens) if forward_looking else None,
        "risk_factors": _truncate_to_tokens(risk_factors, max_tokens // 2) if risk_factors else None,
        "full_text_excerpt": _truncate_to_tokens(text, max_tokens) if not mda else None,
    }

    found = [k for k, v in sections.items() if v is not None]
    logger.info(f"Sections found: {found}")

    return sections


def parse_all_filings(filing_paths: Optional[dict] = None) -> dict:
    """Parse all downloaded filings.

    Args:
        filing_paths: Dict from collector.collect_all() mapping ticker -> list of paths.
                      If None, scans RAW_DIR.

    Returns:
        Dict mapping ticker -> list of {filing_date, filing_type, sections: {...}}
    """
    if filing_paths is None:
        # Scan RAW_DIR for files
        filing_paths = {}
        for ticker in config.TICKERS:
            ticker_dir = config.RAW_DIR / ticker
            if ticker_dir.exists():
                files = [
                    str(f) for f in ticker_dir.rglob("*")
                    if f.is_file() and f.suffix in (".htm", ".html", ".txt")
                ]
                if files:
                    filing_paths[ticker] = files

    results = {}

    for ticker, paths in filing_paths.items():
        logger.info(f"Parsing filings for {ticker} ({len(paths)} files)...")
        ticker_results = []

        for path_str in paths:
            filepath = Path(path_str)
            if not filepath.exists():
                logger.warning(f"File not found: {filepath}")
                continue

            # Extract filing info from filename
            name = filepath.stem
            # Try to extract date from accession number pattern
            filing_date = "unknown"
            filing_type = "10-K"

            # Accession number format: XXXXXXXXXX-YY-ZZZZZZ
            date_match = re.search(r"(\d{4}-\d{2}-\d{2})", name)
            if date_match:
                filing_date = date_match.group(1)

            if "10-Q" in str(filepath) or "10q" in str(filepath).lower():
                filing_type = "10-Q"
            elif "10-K" in str(filepath) or "10k" in str(filepath).lower():
                filing_type = "10-K"

            sections = parse_filing(filepath)

            ticker_results.append({
                "filing_date": filing_date,
                "filing_type": filing_type,
                "filepath": str(filepath),
                "sections": sections,
            })

        results[ticker] = ticker_results

    # Save parsed results
    output_path = config.PROCESSED_DIR / "parsed_filings.json"
    # Only save metadata, not full text
    meta = {}
    for ticker, filings in results.items():
        meta[ticker] = [
            {
                "filing_date": f["filing_date"],
                "filing_type": f["filing_type"],
                "filepath": f["filepath"],
                "sections_found": [k for k, v in f["sections"].items() if v is not None],
            }
            for f in filings
        ]

    with open(output_path, "w", encoding="utf-8") as fout:
        json.dump(meta, fout, indent=2, ensure_ascii=False)

    total = sum(len(v) for v in results.values())
    logger.info(f"Parsed {total} filings across {len(results)} companies")

    return results
