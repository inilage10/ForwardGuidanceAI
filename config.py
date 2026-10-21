"""Configuration settings for the ForwardGuidanceAI application.

This module centralizes directory paths, ticker definitions, filing parameters,
LLM configuration, and guidance analysis dimensions used across the project.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List

# Load environment variables (with fallback if python-dotenv is not installed)
try:
    from dotenv import load_dotenv
    _has_dotenv = True
except ImportError:
    _has_dotenv = False

# Set up module logger
logger = logging.getLogger(__name__)

# Base directory definitions
PROJECT_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
SCORES_DIR: Path = DATA_DIR / "scores"
PRICES_DIR: Path = DATA_DIR / "prices"

# List of directories managed by the project
ALL_DIRS: List[Path] = [
    DATA_DIR,
    RAW_DIR,
    PROCESSED_DIR,
    SCORES_DIR,
    PRICES_DIR,
]


def ensure_directories_exist(dirs: List[Path] | None = None) -> None:
    """Ensure that all required project directories exist on disk.

    Args:
        dirs: Optional list of directory Paths. Defaults to ALL_DIRS if omitted.
    """
    target_dirs = dirs if dirs is not None else ALL_DIRS
    for dir_path in target_dirs:
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error("Failed to create directory %s: %s", dir_path, e)
            raise


# Initialize directory structure
ensure_directories_exist()

# Load environment variables from .env if present
ENV_FILE: Path = PROJECT_DIR / ".env"
if _has_dotenv:
    if ENV_FILE.is_file():
        load_dotenv(dotenv_path=ENV_FILE)
    else:
        load_dotenv()
elif ENV_FILE.is_file():
    try:
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    except Exception:
        pass

# Sector and target companies (dynamically loaded from sector_manager)
def get_active_sector() -> str:
    try:
        from src.sector_manager import load_active_sector
        return load_active_sector()
    except Exception:
        return "Tecnología (Big Tech - Dataset Base)"

def get_active_tickers() -> Dict[str, str]:
    try:
        from src.sector_manager import get_active_tickers
        return get_active_tickers()
    except Exception:
        return {
            "AAPL": "Apple",
            "MSFT": "Microsoft",
            "GOOGL": "Alphabet",
            "AMZN": "Amazon",
            "NVDA": "NVIDIA",
            "META": "Meta Platforms",
            "CRM": "Salesforce",
            "ADBE": "Adobe",
            "INTC": "Intel",
            "AMD": "AMD",
        }

def __getattr__(name: str):
    if name == "SECTOR":
        return get_active_sector()
    if name == "TICKERS":
        return get_active_tickers()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Filing and date range settings
FILING_TYPES: List[str] = ["10-K", "10-Q"]
ANALYSIS_START_DATE: str = "2023-10-01"
ANALYSIS_END_DATE: str = "2026-10-01"

# LLM and external service configurations
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
SEC_EDGAR_EMAIL: str = os.getenv("SEC_EDGAR_EMAIL", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# Guidance dimensions evaluated by LLM
GUIDANCE_DIMENSIONS: List[str] = [
    "revenue_growth",
    "margin_outlook",
    "capex_plans",
    "demand_outlook",
    "earnings_guidance",
    "innovation_pipeline",
    "competitive_position",
]

# Market price return evaluation horizons (in calendar/trading days)
RETURN_WINDOWS: List[int] = [30, 90, 180]
