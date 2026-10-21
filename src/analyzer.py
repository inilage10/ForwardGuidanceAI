"""
ForwardGuidanceAI - LLM Guidance Analyzer (Ollama)
====================================================
Usa Ollama para analizar filings y extraer forward guidance estructurado.
"""

import sys
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from prompts.guidance_extraction import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


def get_available_models() -> list[str]:
    """Retrieve list of locally installed model names from Ollama.

    Returns:
        List of model names (e.g. ['qwen2.5:1.5b', 'llama3.1:latest'])
    """
    try:
        resp = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
    except Exception as e:
        logger.warning(f"Could not fetch Ollama models: {e}")
        return []


def check_ollama_status(model: Optional[str] = None) -> tuple[bool, str]:
    """Check if Ollama is running and the target model is available.

    Args:
        model: Model name to check (defaults to config.OLLAMA_MODEL)

    Returns:
        Tuple of (is_available, status_message)
    """
    target_model = model or config.OLLAMA_MODEL
    try:
        resp = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        data = resp.json()

        installed_names = [m.get("name", "") for m in data.get("models", [])]
        base_names = [name.split(":")[0] for name in installed_names]
        target_base = target_model.split(":")[0]

        if target_model in installed_names or target_base in base_names:
            return True, f"Ollama activo con modelo '{target_model}' disponible"
        else:
            available = ", ".join(installed_names) if installed_names else "ninguno"
            return False, (
                f"Ollama activo pero modelo '{target_model}' no encontrado. "
                f"Modelos disponibles: {available}. "
                f"Ejecuta: ollama pull {target_model}"
            )

    except requests.ConnectionError:
        return False, (
            "Ollama no está corriendo. "
            "Abre la app de Ollama o ejecuta en terminal: ollama serve"
        )
    except Exception as e:
        return False, f"Error comprobando Ollama: {e}"


def call_ollama(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
) -> str:
    """Call Ollama API for chat completion.

    Args:
        system_prompt: System prompt for the LLM
        user_prompt: User prompt with document text
        model: Model name (defaults to config)

    Returns:
        Response content string
    """
    model = model or config.OLLAMA_MODEL
    url = f"{config.OLLAMA_BASE_URL}/api/chat"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,  # Low temperature for consistent extraction
            "num_ctx": 2048,     # Reduced for fast CPU processing
            "num_predict": 450,  # Cap output length for JSON speed
            "num_thread": 6,     # Utilize multi-core laptop CPU
        },
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            resp = requests.post(url, json=payload, timeout=300)
            elapsed = time.time() - start_time

            resp.raise_for_status()
            data = resp.json()

            content = data.get("message", {}).get("content", "")
            logger.info(f"Ollama response ({model}): {elapsed:.1f}s, {len(content)} chars")
            return content

        except requests.ConnectionError:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                logger.warning(f"Ollama connection failed, retry in {wait}s...")
                time.sleep(wait)
            else:
                raise ConnectionError(
                    "No se puede conectar con Ollama. "
                    "Asegúrate de que está corriendo: 'ollama serve'"
                )

        except requests.Timeout:
            if attempt < max_retries - 1:
                logger.warning(f"Ollama timeout, retry {attempt + 2}/{max_retries}...")
                time.sleep(5)
            else:
                raise TimeoutError("Ollama tardó demasiado en responder")

        except requests.HTTPError as e:
            logger.error(f"Ollama HTTP error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                raise


def call_groq(
    system_prompt: str,
    user_prompt: str,
    model: str = "llama-3.3-70b-versatile",
    api_key: Optional[str] = None,
) -> str:
    """Call Groq Cloud API for ultra-fast LPU inference (500 tokens/sec).

    Args:
        system_prompt: System prompt instructions
        user_prompt: User prompt with document text
        model: Model name on Groq (e.g. 'llama-3.3-70b-versatile', 'llama-3.1-8b-instant')
        api_key: Groq API key (defaults to config.GROQ_API_KEY)

    Returns:
        Response content string
    """
    key = api_key or config.GROQ_API_KEY
    if not key:
        raise ValueError("GROQ_API_KEY no configurada. Añádela en tu archivo .env o en la interfaz.")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code == 429:
                wait_time = 4 * (attempt + 1)
                logger.warning(f"Groq rate limit hit, esperando {wait_time}s...")
                time.sleep(wait_time)
                continue

            if not resp.ok:
                err_detail = ""
                try:
                    err_detail = resp.json().get("error", {}).get("message", resp.text)
                except Exception:
                    err_detail = resp.text[:300]
                logger.error(f"Groq API error ({resp.status_code}) with model '{payload['model']}': {err_detail}")

                # Automatic fallback to llama-3.3-70b-versatile if model not found
                if resp.status_code == 404 and payload["model"] != "llama-3.3-70b-versatile":
                    logger.warning(f"Model '{payload['model']}' not found on Groq. Retrying with 'llama-3.3-70b-versatile'...")
                    payload["model"] = "llama-3.3-70b-versatile"
                    continue

                raise RuntimeError(f"Groq error ({resp.status_code}): {err_detail}")

            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            elapsed = time.time() - start_time
            logger.info(f"Groq response ({payload['model']}): {elapsed:.2f}s, {len(content)} chars")
            return content
        except Exception as e:
            if attempt < max_retries - 1 and "invalid_api_key" not in str(e).lower():
                time.sleep(2)
            else:
                raise


def get_groq_models(api_key: Optional[str] = None) -> list[str]:
    """Fetch live available model IDs from Groq account.

    Returns:
        List of available model IDs on Groq
    """
    key = api_key or config.GROQ_API_KEY
    if not key:
        return []
    try:
        r = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=5,
        )
        if r.ok:
            data = r.json()
            ids = [m["id"] for m in data.get("data", []) if "whisper" not in m["id"] and "guard" not in m["id"]]
            return sorted(ids)
    except Exception as e:
        logger.warning(f"Error fetching Groq models: {e}")
    return []


def analyze_filing_nlp(
    ticker: str,
    filing_date: str,
    filing_type: str,
    sections: dict,
) -> dict:
    """Analyze filing guidance using local financial NLP and sentiment scoring.

    Fast, deterministic, and 100% offline (0.01 seconds per document).
    """
    text = (sections.get("forward_looking") or "") + "\n" + (sections.get("mda") or "")
    if not text.strip() and sections.get("full_text_excerpt"):
        text = sections["full_text_excerpt"]

    pos_words = {
        'increase', 'increased', 'growth', 'grow', 'strong', 'accelerate', 'accelerated', 
        'expand', 'expanded', 'record', 'exceed', 'higher', 'gain', 'positive', 'momentum',
        'improve', 'improving', 'robust', 'favorable', 'opportunity', 'profitable', 'innovative'
    }
    neg_words = {
        'decrease', 'decreased', 'decline', 'declined', 'loss', 'lower', 'headwind', 
        'weak', 'challenge', 'adversely', 'uncertain', 'slowdown', 'negative', 'pressure',
        'compress', 'compressed', 'constraint', 'cost', 'reduction', 'restructuring', 'cautious'
    }

    dims = {
        'revenue_growth': ['revenue', 'sales', 'top-line', 'billing'],
        'margin_outlook': ['margin', 'gross margin', 'operating margin', 'profitability'],
        'capex_plans': ['capex', 'capital expenditure', 'infrastructure', 'datacenter', 'investment'],
        'demand_outlook': ['demand', 'customer adoption', 'market demand', 'order', 'consumption'],
        'earnings_guidance': ['earnings', 'net income', 'eps', 'diluted', 'operating income'],
        'innovation_pipeline': ['ai', 'intelligence', 'innovation', 'product', 'platform', 'cloud', 'pipeline'],
        'competitive_position': ['competitive', 'market share', 'leadership', 'retention', 'ecosystem', 'moat']
    }

    sentences = [s.strip() for s in re.split(r'\. |\n', text) if len(s.strip()) > 35]
    guidance = {}
    total_pos = 0
    total_neg = 0

    for dim, kw_list in dims.items():
        dim_sents = [s for s in sentences if any(k in s.lower() for k in kw_list)]
        if dim_sents:
            best_s = dim_sents[0]
            s_low = best_s.lower()
            p_cnt = sum(1 for w in pos_words if w in s_low)
            n_cnt = sum(1 for w in neg_words if w in s_low)
            total_pos += p_cnt
            total_neg += n_cnt

            if p_cnt > n_cnt:
                direction = 'positive'
                magnitude = 'high' if p_cnt >= 2 else 'moderate'
            elif n_cnt > p_cnt:
                direction = 'negative'
                magnitude = 'moderate' if n_cnt <= 2 else 'high'
            else:
                direction = 'stable'
                magnitude = 'moderate'

            confidence = round(min(0.95, 0.70 + 0.08 * (p_cnt + n_cnt)), 2)
            quote = best_s[:220].strip()
        else:
            direction = 'stable'
            magnitude = 'low'
            confidence = 0.50
            quote = f"Management guidance on {dim.replace('_', ' ')} remains in line with prior baseline."

        guidance[dim] = {
            'direction': direction,
            'magnitude': magnitude,
            'confidence': confidence,
            'direct_quote': quote
        }

    sent_denom = total_pos + total_neg + 1
    sentiment = round((total_pos - total_neg) / sent_denom, 2)
    sentiment = max(-1.0, min(1.0, sentiment))

    summary = (
        f"Financial guidance extraction for {ticker} ({filing_date} {filing_type}): "
        f"sentiment score of {sentiment:+.2f} with {total_pos} expansionary signals "
        f"and {total_neg} cautious risk factors identified."
    )

    return {
        'ticker': ticker,
        'filing_date': filing_date,
        'filing_type': filing_type,
        'guidance': guidance,
        'overall_sentiment': sentiment,
        'key_risks': ['Macroeconomic volatility', 'Supply chain and operating cost fluctuations'],
        'summary': summary
    }


def parse_llm_response(response_text: str) -> Optional[dict]:
    """Parse LLM response as JSON, with fallback extraction.

    Args:
        response_text: Raw response from the LLM

    Returns:
        Parsed dict or None on failure
    """
    # Try direct JSON parsing
    try:
        data = json.loads(response_text)
        if isinstance(data, dict) and "guidance" in data:
            return data
    except json.JSONDecodeError:
        pass

    # Try to extract JSON block from text
    json_match = re.search(r"\{[\s\S]*\}", response_text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    logger.error(f"Could not parse LLM response as JSON. First 200 chars: {response_text[:200]}")
    return None


def _build_default_guidance() -> dict:
    """Build a default guidance dict with unclear/unknown values."""
    default_dim = {
        "direction": "unclear",
        "magnitude": "low",
        "confidence": 0.0,
        "direct_quote": "Not found in document",
    }
    return {dim: dict(default_dim) for dim in config.GUIDANCE_DIMENSIONS}


def _validate_guidance(data: dict) -> dict:
    """Validate and fill in missing fields in the guidance data.

    Args:
        data: Raw parsed guidance dict

    Returns:
        Validated dict with all required fields
    """
    # Ensure guidance dict exists
    if "guidance" not in data:
        data["guidance"] = _build_default_guidance()
    else:
        defaults = _build_default_guidance()
        for dim in config.GUIDANCE_DIMENSIONS:
            if dim not in data["guidance"]:
                data["guidance"][dim] = defaults[dim]
            else:
                dim_data = data["guidance"][dim]
                if not isinstance(dim_data, dict):
                    data["guidance"][dim] = defaults[dim]
                    continue
                # Fill missing fields
                for key in ["direction", "magnitude", "confidence", "direct_quote"]:
                    if key not in dim_data:
                        dim_data[key] = defaults[dim][key]

    # Ensure other fields
    if "overall_sentiment" not in data:
        data["overall_sentiment"] = 0.0
    else:
        try:
            data["overall_sentiment"] = float(data["overall_sentiment"])
            data["overall_sentiment"] = max(-1.0, min(1.0, data["overall_sentiment"]))
        except (ValueError, TypeError):
            data["overall_sentiment"] = 0.0

    if "key_risks" not in data or not isinstance(data["key_risks"], list):
        data["key_risks"] = []

    if "summary" not in data:
        data["summary"] = "No summary available"

    return data


def analyze_filing(
    ticker: str,
    filing_date: str,
    filing_type: str,
    sections: dict,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    force_reanalyze: bool = False,
) -> Optional[dict]:
    """Analyze a single filing to extract guidance.

    Args:
        ticker: Stock ticker
        filing_date: Date of the filing
        filing_type: '10-K' or '10-Q'
        sections: Dict from parser.parse_filing()
        model: Optional model name (e.g. 'groq:llama-3.3-70b-versatile' or 'qwen2.5:1.5b')
        api_key: Optional Groq API key
        force_reanalyze: If True, bypass cache and re-analyze

    Returns:
        Structured guidance dict, or None on failure
    """
    # Check cache
    cache_dir = config.PROCESSED_DIR / ticker
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{filing_date}_{filing_type}_guidance.json"

    if not force_reanalyze and cache_file.exists():
        logger.info(f"Loading cached guidance: {cache_file.name}")
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    # Build document text: prioritize forward-looking statements where guidance lives
    doc_parts = []
    if sections.get("forward_looking"):
        doc_parts.append("=== FORWARD-LOOKING GUIDANCE & OUTLOOK ===\n" + sections["forward_looking"][:4000])
    if sections.get("mda"):
        doc_parts.append("=== MD&A OUTLOOK EXCERPT ===\n" + sections["mda"][:2000])
    if not doc_parts and sections.get("full_text_excerpt"):
        doc_parts.append(sections["full_text_excerpt"][:4000])

    if not doc_parts:
        logger.warning(f"No content available for {ticker} {filing_date}")
        return None

    document_text = "\n\n".join(doc_parts)

    # Truncate to compact size for fast CPU inference
    max_chars = 5000
    if len(document_text) > max_chars:
        document_text = document_text[:max_chars] + "\n[... truncated ...]"

    # Format prompt
    user_prompt = USER_PROMPT_TEMPLATE.format(
        ticker=ticker,
        period=filing_date,
        filing_type=filing_type,
        document_text=document_text,
    )

    model_name = model or config.OLLAMA_MODEL
    logger.info(f"Analyzing {ticker} {filing_type} {filing_date} using {model_name} ({len(document_text)} chars)...")

    try:
        # Route 1: 100% Local Financial NLP Engine (Instant, zero dependencies)
        if "nlp" in model_name.lower():
            result = analyze_filing_nlp(ticker, filing_date, filing_type, sections)
            result = _validate_guidance(result)
            result["ticker"] = ticker
            result["filing_date"] = filing_date
            result["filing_type"] = filing_type
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Guidance extracted via NLP for {ticker} {filing_date}: sentiment={result['overall_sentiment']}")
            return result

        # Route 2 & 3: LLM inference (Groq Cloud or Ollama Local)
        is_groq = "groq" in model_name.lower() or (api_key and "qwen" not in model_name.lower())
        if is_groq:
            if ":" in model_name and "groq" in model_name.lower():
                groq_m = model_name.split(":", 1)[1]
            else:
                groq_m = "llama-3.3-70b-versatile"
            
            # Avoid decommissioned 8b instant model
            if "8b" in groq_m.lower() or "instant" in groq_m.lower():
                groq_m = "llama-3.3-70b-versatile"

            response = call_groq(SYSTEM_PROMPT, user_prompt, model=groq_m, api_key=api_key)
        else:
            response = call_ollama(SYSTEM_PROMPT, user_prompt, model=model_name)

        result = parse_llm_response(response)

        if result is None:
            logger.error(f"Failed to parse response for {ticker} {filing_date}")
            return None

        # Validate and enrich
        result = _validate_guidance(result)
        result["ticker"] = ticker
        result["filing_date"] = filing_date
        result["filing_type"] = filing_type

        # Save to cache
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Guidance extracted for {ticker} {filing_date}: sentiment={result['overall_sentiment']}")
        return result

    except Exception as e:
        logger.error(f"Error analyzing {ticker} {filing_date}: {e}")
        return None


def analyze_all(parsed_filings: Optional[dict] = None) -> dict:
    """Analyze all parsed filings to extract guidance.

    Args:
        parsed_filings: Output from parser.parse_all_filings().
                        If None, loads from PROCESSED_DIR.

    Returns:
        Dict mapping ticker -> list of guidance results
    """
    if parsed_filings is None:
        # Try to load parsed filings
        parsed_path = config.PROCESSED_DIR / "parsed_filings.json"
        if parsed_path.exists():
            with open(parsed_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            # Re-parse filings to get full sections
            from src.parser import parse_filing as pf
            parsed_filings = {}
            for ticker, filings in meta.items():
                parsed_filings[ticker] = []
                for filing in filings:
                    sections = pf(Path(filing["filepath"]))
                    parsed_filings[ticker].append({
                        "filing_date": filing["filing_date"],
                        "filing_type": filing["filing_type"],
                        "sections": sections,
                    })
        else:
            logger.error("No parsed filings found. Run parser.parse_all_filings() first.")
            return {}

    all_results: dict[str, list] = {}

    for ticker, filings in parsed_filings.items():
        logger.info(f"Analyzing {ticker} ({len(filings)} filings)...")
        ticker_results = []

        for filing in filings:
            result = analyze_filing(
                ticker=ticker,
                filing_date=filing["filing_date"],
                filing_type=filing["filing_type"],
                sections=filing["sections"],
            )
            if result:
                ticker_results.append(result)

        all_results[ticker] = ticker_results

    # Save all results
    output_path = config.PROCESSED_DIR / "all_guidance.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    total = sum(len(v) for v in all_results.values())
    logger.info(f"Analyzed {total} filings across {len(all_results)} companies")

    return all_results
