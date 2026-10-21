"""
ForwardGuidanceAI - Prompts para extracción de guidance
========================================================
Prompts estructurados para que el LLM extraiga forward guidance
de filings de la SEC.
"""

SYSTEM_PROMPT = """You are a senior financial analyst specializing in equity research and corporate guidance analysis.

Your task is to extract forward-looking guidance and management outlook from SEC filing excerpts (10-K and 10-Q documents).

Forward guidance includes any statements about future expectations, projections, targets, or outlook provided by management regarding:

1. **revenue_growth**: Expected revenue trends, sales growth, demand projections
2. **margin_outlook**: Expected profit margins, cost trends, operating efficiency
3. **capex_plans**: Capital expenditure plans, investment intentions, R&D spending
4. **demand_outlook**: Market demand expectations, order pipeline, customer trends
5. **earnings_guidance**: EPS guidance, profit expectations, bottom-line targets
6. **innovation_pipeline**: New products, technologies, services in development
7. **competitive_position**: Market share expectations, competitive advantages, strategic positioning

For each dimension, provide:
- **direction**: One of "positive", "negative", "stable", or "unclear"
- **magnitude**: One of "high", "moderate", or "low"  
- **confidence**: A float from 0.0 to 1.0 indicating how clearly the guidance was stated
- **direct_quote**: The exact sentence or phrase from the text that supports this assessment

Also provide:
- **overall_sentiment**: A float from -1.0 (very negative) to 1.0 (very positive)
- **key_risks**: A list of the main risks mentioned
- **summary**: A 2-3 sentence summary of the company's forward outlook

You MUST respond with valid JSON only. No additional text before or after the JSON."""

USER_PROMPT_TEMPLATE = """Analyze the following SEC filing excerpt and extract all forward-looking guidance.

**Company:** {ticker}
**Filing Period:** {period}
**Filing Type:** {filing_type}

---
DOCUMENT TEXT:
{document_text}
---

Extract the forward guidance as a JSON object with this exact structure:
{{
  "ticker": "{ticker}",
  "period": "{period}",
  "filing_type": "{filing_type}",
  "guidance": {{
    "revenue_growth": {{"direction": "...", "magnitude": "...", "confidence": 0.0, "direct_quote": "..."}},
    "margin_outlook": {{"direction": "...", "magnitude": "...", "confidence": 0.0, "direct_quote": "..."}},
    "capex_plans": {{"direction": "...", "magnitude": "...", "confidence": 0.0, "direct_quote": "..."}},
    "demand_outlook": {{"direction": "...", "magnitude": "...", "confidence": 0.0, "direct_quote": "..."}},
    "earnings_guidance": {{"direction": "...", "magnitude": "...", "confidence": 0.0, "direct_quote": "..."}},
    "innovation_pipeline": {{"direction": "...", "magnitude": "...", "confidence": 0.0, "direct_quote": "..."}},
    "competitive_position": {{"direction": "...", "magnitude": "...", "confidence": 0.0, "direct_quote": "..."}}
  }},
  "overall_sentiment": 0.0,
  "key_risks": ["risk1", "risk2"],
  "summary": "Brief 2-3 sentence summary..."
}}

Respond ONLY with the JSON object. No other text."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "ticker": {"type": "string"},
        "period": {"type": "string"},
        "filing_type": {"type": "string"},
        "guidance": {
            "type": "object",
            "properties": {
                dim: {
                    "type": "object",
                    "properties": {
                        "direction": {"type": "string", "enum": ["positive", "negative", "stable", "unclear"]},
                        "magnitude": {"type": "string", "enum": ["high", "moderate", "low"]},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "direct_quote": {"type": "string"},
                    },
                    "required": ["direction", "magnitude", "confidence", "direct_quote"],
                }
                for dim in [
                    "revenue_growth", "margin_outlook", "capex_plans",
                    "demand_outlook", "earnings_guidance",
                    "innovation_pipeline", "competitive_position",
                ]
            },
        },
        "overall_sentiment": {"type": "number", "minimum": -1.0, "maximum": 1.0},
        "key_risks": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["ticker", "guidance", "overall_sentiment", "key_risks", "summary"],
}
