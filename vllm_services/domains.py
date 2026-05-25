"""
Domain classification utilities for real-time call dashboards.

Canonical domains (exact labels expected by UI / downstream):
  - Bullying
  - Sexual Harassment
  - Hate Speech
  - Banking
  - Healthcare
  - Utilities
  - Education
  - Account Verification
  - Neutral
  - N/A
"""

from __future__ import annotations

from typing import Optional, Any


def _safe_str(x: Any) -> str:
    try:
        return str(x)
    except Exception:
        return ""


CANONICAL_DOMAINS = (
    "Bullying",
    "Sexual Harassment",
    "Hate Speech",
    "Banking",
    "Healthcare",
    "Utilities",
    "Education",
    "Account Verification",
    "Neutral",
    "N/A",
)


def normalize_domain(value: Optional[str]) -> str:
    """
    Normalize free-text domain outputs (Greek/English variants) to CANONICAL_DOMAINS.
    """
    if not value:
        return "N/A"
    s = str(value).strip()
    if not s:
        return "N/A"

    low = s.lower()

    # Direct matches
    for d in CANONICAL_DOMAINS:
        if low == d.lower():
            return d

    # N/A
    if low in {"n/a", "na", "n a"}:
        return "N/A"

    # Sexual harassment
    if "sexual" in low or "harass" in low or "παρενόχλη" in low or "σεξου" in low:
        return "Sexual Harassment"

    # Hate speech
    if "hate" in low or "μίσ" in low or "ρατσ" in low or "ξενοφο" in low:
        return "Hate Speech"

    # Bullying
    if "bully" in low or "εκφοβ" in low or "απειλ" in low:
        return "Bullying"

    # Banking
    if (
        "bank" in low
        or "banking" in low
        or "τράπεζ" in low
        or "κάρτα" in low
        or "λογαριασ" in low
        or "δάνει" in low
        or "επένδυ" in low
    ):
        return "Banking"

    # Utilities
    if (
        "utilit" in low
        or "ρεύμα" in low
        or "νερό" in low
        or "αέριο" in low
        or "δεη" in low
        or "πάροχ" in low
        or "τηλεπικοινων" in low
    ):
        return "Utilities"

    # Healthcare
    if (
        "health" in low
        or "healthcare" in low
        or "ιατρ" in low
        or "γιατρ" in low
        or "φαρμακ" in low
        or "κλιν" in low
    ):
        return "Healthcare"

    # Education
    if "educat" in low or "εκπαίδευ" in low or "σχολ" in low or "μαθή" in low or "φοιτη" in low:
        return "Education"

    # Account verification
    if (
        "verification" in low
        or "verify" in low
        or "account verification" in low
        or "επιβεβαίω" in low
        or "επαλήθευ" in low
        or "επικαιροποίη" in low
    ):
        return "Account Verification"

    # Neutral
    if "neutral" in low or "κανον" in low:
        return "Neutral"

    return "Neutral"


def normalize_domain_from_json(obj: Any) -> str:
    """
    Normalize a domain from an LLM JSON response.

    Expected LLM JSON: {"domain": "..."}.
    Returns a canonical domain string.
    """
    if isinstance(obj, dict):
        return normalize_domain(obj.get("domain"))
    return normalize_domain(_safe_str(obj))


async def semantic_domain_via_vllm(
    *,
    text: str,
    client: Any,
) -> str:
    """
    Semantic domain classification using the vLLM-backed LLM (JSON output).

    This is the preferred path for **semantic meaning** beyond keywords.
    It requires:
      - `client` to have an async method `classify_json(prompt, max_tokens=...)`
        (compatible with `vllm_services_dev.vllm_client.VLLMClient`)
      - `vllm_services_dev.prompts.build_domain_prompt` for prompt construction

    Returns a canonical domain label from CANONICAL_DOMAINS.

    Notes:
    - If the model returns invalid JSON or an unknown value, we fall back to `normalize_domain(text)`.
    - This keeps real-time robustness even when the LLM output is malformed.
    """
    from . import prompts  # local import to avoid import-time coupling

    prompt = prompts.build_domain_prompt(text)
    try:
        parsed, _lat_ms = await client.classify_json(prompt, max_tokens=64)
        dom = normalize_domain_from_json(parsed)
        if dom in CANONICAL_DOMAINS:
            return dom
    except Exception:
        pass

    # Fallback: keyword-based normalization from text (best-effort)
    return normalize_domain(text)

