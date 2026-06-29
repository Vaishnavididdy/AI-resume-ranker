"""JD Intelligence Layer — calls Claude API once to parse the job description
into structured features: must_have, preferred, red_flags, behavioral_requirements,
hidden_intent, min_years, and key_skills list.

Falls back gracefully to a keyword-based extractor if the API call fails.
"""
from __future__ import annotations
import json
import re
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """\
You are an expert technical recruiter assistant.
Given a raw job description, extract a structured JSON object with exactly these keys:

{
  "title": "<job title string>",
  "min_years": <integer, minimum years of experience required, 0 if not stated>,
  "key_skills": ["<skill>", ...],
  "must_have": ["<requirement>", ...],
  "preferred": ["<nice-to-have>", ...],
  "red_flags": ["<explicit rejection criteria>", ...],
  "behavioral_requirements": ["<soft skill or work-style requirement>", ...],
  "hidden_intent": ["<inferred recruiter intent not explicitly stated>", ...]
}

Rules:
- Return ONLY valid JSON, no markdown fences, no extra text.
- key_skills: canonical lowercase skill names (e.g. "python", "embeddings", "rag", "mlops").
- hidden_intent: what the recruiter really cares about beneath the surface (e.g. "wants someone who has shipped ranking systems at scale, not just academic work").
- red_flags: e.g. "no production experience", "consulting-only background".
- Be concise; each list item is one short phrase or skill.
"""

_USER_TEMPLATE = "Job Description:\n\n{jd_text}"


# ---------------------------------------------------------------------------
# Claude API helper (no key needed — handled by proxy)
# ---------------------------------------------------------------------------
def _call_claude(jd_text: str) -> Dict[str, Any]:
    import urllib.request

    payload = json.dumps({
        "model": "claude-sonnet-4-6",
        "max_tokens": 1000,
        # FIX: temperature=0 makes the response deterministic (greedy decoding).
        # Without this, Claude samples stochastically → key_skills, must_have,
        # and min_years differ across runs → skill_overlap and exp_fit scores vary.
        "temperature": 0,
        "messages": [{"role": "user", "content": _USER_TEMPLATE.format(jd_text=jd_text[:4000])}],
        "system": _SYSTEM_PROMPT,
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())

    raw = "".join(
        block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"
    )
    # strip possible fences
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Keyword fallback
# ---------------------------------------------------------------------------
_FALLBACK_SKILLS = [
    "python", "embeddings", "rag", "retrieval", "ranking", "llm",
    "mlops", "production", "vector search", "transformers", "pytorch",
    "tensorflow", "recommendation", "ndcg", "evaluation",
]

def _keyword_fallback(jd_text: str) -> Dict[str, Any]:
    lower = jd_text.lower()
    found = [s for s in _FALLBACK_SKILLS if s in lower]
    years_match = re.search(r"(\d+)\+?\s*years?", lower)
    min_years = int(years_match.group(1)) if years_match else 0
    # naive title extraction
    first_line = jd_text.strip().splitlines()[0][:80] if jd_text.strip() else "Engineer"
    return {
        "title": first_line,
        "min_years": min_years,
        "key_skills": found,
        "must_have": found[:5],
        "preferred": found[5:],
        "red_flags": ["no production ML experience", "consulting-only background"],
        "behavioral_requirements": ["self-starter", "collaborative"],
        "hidden_intent": ["wants production ML experience over academic credentials"],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
# FIX: In-process cache keyed on JD text hash.
# Even with temperature=0, network retries can return different tokens in
# edge cases. Caching pins the parse result for identical inputs within
# the same process lifecycle.
import hashlib as _hashlib
_JD_PARSE_CACHE: dict = {}


def parse_jd(jd_raw: Dict[str, Any]) -> Dict[str, Any]:
    """Return enriched JD features dict (superset of what extract_jd_features returned)."""
    text = jd_raw.get("raw_text", "")

    # FIX: Cache hit — identical JD text always returns identical features.
    cache_key = _hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()
    if cache_key in _JD_PARSE_CACHE:
        print("[jd_intelligence] Cache hit — returning cached JD parse")
        return _JD_PARSE_CACHE[cache_key]

    try:
        structured = _call_claude(text)
        print("[jd_intelligence] LLM parse succeeded")
    except Exception as exc:
        print(f"[jd_intelligence] LLM parse failed ({exc}), using keyword fallback")
        structured = _keyword_fallback(text)

    # Normalise
    structured.setdefault("title", "Engineer")
    structured.setdefault("min_years", 0)
    structured.setdefault("key_skills", [])
    structured.setdefault("must_have", [])
    structured.setdefault("preferred", [])
    structured.setdefault("red_flags", [])
    structured.setdefault("behavioral_requirements", [])
    structured.setdefault("hidden_intent", [])

    # Lower-case all skills for matching
    structured["key_skills"] = [s.lower().strip() for s in structured["key_skills"]]
    structured["must_have_lower"] = [s.lower().strip() for s in structured["must_have"]]
    structured["text"] = text

    # FIX: Store in cache
    _JD_PARSE_CACHE[cache_key] = structured
    return structured