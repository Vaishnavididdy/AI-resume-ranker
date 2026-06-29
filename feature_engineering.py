"""
Candidate ranking pipeline:
Stage 1 scores candidates using semantic match, career evidence,
production experience, company relevance, and behavioral signals.
Stage 2 refines ranking using pairwise comparison.
"""
from __future__ import annotations
import re
import math
import json
from typing import Any, Dict, List, Set, Tuple, Union, Iterable
from datetime import datetime

# FIX: All keyword collections that are ITERATED (not just membership-tested)
# must be sorted Lists, not Sets. Sets in CPython use hash-randomised ordering
# controlled by PYTHONHASHSEED, which is re-seeded every process start.
# Iterating a Set in scoring order produces different matched_evidence orderings
# across runs, which corrupts the dedup count and the reasoning text.
# Membership tests (kw in set) are unaffected by order — those Sets are fine.
_SORTED_LIST = sorted  # alias for readability at definition sites

# ---------------------------------------------------------------------------
# Keyword sets — retrieval/ranking domain
# ---------------------------------------------------------------------------
# These must appear in CAREER/PROJECT text to count as retrieval evidence
_RETRIEVAL_CAREER_KEYWORDS: List[str] = _SORTED_LIST([
    # Search & retrieval systems
    "search engine", "search system", "information retrieval", "retrieval pipeline",
    "retrieval system", "query understanding", "query expansion", "query rewriting",
    "document retrieval", "passage retrieval", "dense retrieval", "sparse retrieval",
    "hybrid retrieval", "semantic search", "lexical search",
    # Ranking systems
    "ranking system", "ranking model", "ranking pipeline", "re-ranking", "reranking",
    "learning to rank", "ltr", "pointwise", "pairwise", "listwise", "lambdarank",
    "relevance ranking", "neural ranking", "passage ranking",
    # Recommendation systems
    "recommendation engine", "recommendation system", "recommender system",
    "collaborative filtering", "content-based filtering", "matrix factorization",
    "item-to-item", "user-to-item", "candidate generation", "two-tower",
    "personalization", "personaliz", "recsys",
    # Embeddings in context
    "embedding model", "embedding pipeline", "sentence embedding", "dense embedding",
    "semantic embedding", "learned embedding", "text embedding",
    # Vector infrastructure
    "vector search", "vector index", "approximate nearest neighbor", "ann index",
    "faiss", "hnsw", "annoy", "qdrant", "weaviate", "pinecone", "milvus",
    "pgvector", "elasticsearch vector", "opensearch vector",
    # Classic IR
    "bm25", "tf-idf", "inverted index", "solr", "elasticsearch", "opensearch",
    # Evaluation metrics in context (must appear in career/project descriptions)
    "ndcg", "mrr", "map@", "precision@", "recall@", "normalized discounted",
    "mean average precision", "mean reciprocal", "click-through rate",
    "relevance judgment", "offline evaluation", "online evaluation",
    "a/b test.*rank", "search quality",
    # Matching systems
    "job matching", "candidate matching", "entity matching", "semantic matching",
    "cross-encoder", "bi-encoder", "dual-encoder",
])

# Patterns that indicate retrieval work in free text (regex)
_RETRIEVAL_CAREER_PATTERNS: List[str] = [
    r"built\s+(?:a\s+)?(?:search|retrieval|recommendation|ranking)",
    r"(?:designed|developed|implemented|created)\s+(?:a\s+)?(?:search|retrieval|recommendation|ranking|recsys)",
    r"(?:improved|optimized)\s+(?:search|ranking|retrieval|relevance|recall|precision)",
    r"(?:search|retrieval|ranking)\s+(?:system|pipeline|engine|infrastructure|service|api)",
    r"recommendation\s+(?:system|engine|pipeline|model|service)",
    r"vector\s+(?:search|store|database|index)",
    r"semantic\s+(?:search|similarity|matching|retrieval)",
    r"(?:deployed|served|scaled)\s+(?:embedding|retrieval|ranking|search)",
    r"(?:ndcg|mrr|map@\d|precision@\d|recall@\d)",
    r"(?:faiss|hnsw|annoy|qdrant|weaviate|pinecone|milvus)",
    r"learning.to.rank",
    r"(?:two.tower|dual.encoder|cross.encoder|bi.encoder)",
    r"(?:item|user).embedding",
    r"candidate\s+(?:retrieval|generation|ranking)",
    r"search\s+(?:relevance|quality|evaluation|recall|precision)",
]

# Title keywords that strongly indicate retrieval/ranking role
_RETRIEVAL_TITLE_KEYWORDS: Set[str] = {
    "search", "retrieval", "ranking", "recommendation", "recsys",
    "relevance", "information retrieval", "matching",
}

# ML/AI title keywords (weaker signal — need supporting career evidence)
_ML_TITLE_KEYWORDS: Set[str] = {
    "machine learning", "ml engineer", "ai engineer", "data scientist",
    "nlp engineer", "applied scientist", "research scientist",
}

# ---------------------------------------------------------------------------
# Company classification
# ---------------------------------------------------------------------------
_PRODUCT_COMPANIES: Set[str] = {
    "google", "meta", "facebook", "amazon", "apple", "microsoft", "netflix",
    "uber", "lyft", "airbnb", "linkedin", "twitter", "x corp", "spotify",
    "stripe", "openai", "anthropic", "databricks", "snowflake", "palantir",
    "salesforce", "adobe", "doordash", "instacart", "pinterest", "snapchat",
    "dropbox", "slack", "atlassian", "shopify", "square", "block", "figma",
    "notion", "airtable", "coupang", "grab", "flipkart", "swiggy", "zomato",
    "meesho", "razorpay", "phonepe",
}

_CONSULTING_FIRMS: Set[str] = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mphasis", "hexaware", "lti",
    "mindtree", "genpact", "virtusa", "cgi", "epam", "ness", "persistent",
    "zensar", "mtwenty", "niit tech", "mastech",
}

# Roles that strongly indicate NO retrieval work (unless overridden by career evidence)
_DISQUALIFYING_ROLES: Set[str] = {
    "full stack developer", "frontend developer", "backend developer",
    "web developer", "mobile developer", "ios developer", "android developer",
    "qa engineer", "quality assurance", "test engineer", "automation engineer",
    "devops engineer", "site reliability", "sre", "infrastructure engineer",
    "network engineer", "database administrator", "dba", "system administrator",
    "technical support", "it support", "helpdesk", "business analyst",
    "project manager", "scrum master", "product manager",
}

# STRONG DISQUALIFIER ROLES - These should heavily penalize candidates
_STRONG_DISQUALIFIER_ROLES: Set[str] = {
    "sales", "sales executive", "sales manager", "business development",
    "graphic designer", "ui designer", "ux designer", "visual designer",
    "operations manager", "operations analyst", "operations executive",
    "marketing", "marketing manager", "digital marketing", "seo specialist",
    "human resources", "hr", "hr manager", "talent acquisition", "recruiter",
    "project coordinator", "program manager", "scrum master",
    "financial analyst", "finance", "accounting", "accountant",
    "customer support", "customer service", "support engineer",
    "data analyst", "business intelligence", "bi analyst",
}

# Production ML keywords (must appear in career/project text)
_PROD_ML_CAREER_KEYWORDS: frozenset = frozenset({
    "production", "serving", "latency", "throughput", "scalab", "deployed at scale",
    "millions of", "billions of", "real-time", "low latency", "high availability",
    "kubernetes", "docker", "mlops", "feature store", "model monitoring",
    "a/b test", "shadow deploy", "canary deploy", "online serving",
})

# Generic AI buzzwords without depth (flag if present alone)
_AI_BUZZWORD_SET: frozenset = frozenset({
    "ai", "artificial intelligence", "chatgpt", "gpt", "llm", "generative ai",
    "prompt engineering", "langchain", "openai api", "chatbot", "ai integration",
})

# ============================================================================
# ROBUST TEXT EXTRACTION - RECURSIVE FIELD DISCOVERY
# ============================================================================

# Fields to skip during text extraction (metadata, IDs, etc.)
_SKIP_KEYS: Set[str] = {
    "id", "ids", "candidate_id", "user_id", "person_id", "employee_id",
    "timestamp", "created_at", "updated_at", "deleted_at",
    "is_active", "is_deleted", "is_verified", "is_available",
    "version", "sequence", "position", "index",
    "type", "category", "classification", "label", "tag",
    "score", "rating", "rank", "priority",
    "count", "total", "size", "length",
}

# Fields that are likely to contain text content
_TEXT_FIELDS: Set[str] = {
    "description", "descriptions", "summary", "summaries", 
    "responsibilities", "achievements", "accomplishments", 
    "details", "detail", "bio", "biography", "about", "overview",
    "experience", "experiences", "work", "career", "employment",
    "education", "educations", "degree", "major", "field_of_study",
    "project", "projects", "portfolio", "publication", "publications",
    "patent", "patents", "certification", "certifications", 
    "award", "awards", "honor", "honors", "recognition",
    "role", "position", "title", "job_title", "job_title_description",
    "company", "organization", "employer", "team", "department",
    "skills", "skill", "competencies", "technologies", "tools",
    "languages", "language", "framework", "framework", "library",
    "resume", "cv", "profile", "profile_summary",
    "headline", "tagline", "objective", "career_objective",
    "interest", "interests", "hobby", "hobbies",
    "volunteer", "volunteering", "community", "leadership",
    "achievement", "accomplishment", "impact", "outcome",
    "results", "deliverables", "projects", "project_summary",
    "work_description", "role_description", "position_description",
    "job_summary", "role_summary", "position_summary",
    "responsibility", "key_responsibilities", "main_responsibilities",
    "duties", "task", "tasks", "activities", "daily_tasks",
    "technical_skills", "soft_skills", "domain_skills",
}


def _recursive_text_extractor(
    obj: Any,
    path: str = "",
    max_depth: int = 10,
    current_depth: int = 0,
) -> List[str]:
    """
    Recursively extract all text strings from a nested Python object.
    
    Args:
        obj: The object to extract text from (dict, list, str, etc.)
        path: Current path in the object (for debugging)
        max_depth: Maximum recursion depth
        current_depth: Current recursion depth
    
    Returns:
        List of extracted text strings
    """
    if current_depth > max_depth:
        return []
    
    extracted: List[str] = []
    
    # Base case: string
    if isinstance(obj, str):
        if obj.strip():
            extracted.append(obj.strip())
        return extracted
    
    # Base case: numbers, booleans, None - skip
    if isinstance(obj, (int, float, bool, type(None))):
        return extracted
    
    # Handle lists
    if isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, str) and item.strip():
                extracted.append(item.strip())
            else:
                extracted.extend(_recursive_text_extractor(
                    item, f"{path}[{i}]", max_depth, current_depth + 1
                ))
        return extracted
    
    # Handle dictionaries
    if isinstance(obj, dict):
        # First, check for common text fields
        for key, value in obj.items():
            # Skip metadata keys
            if key.lower() in _SKIP_KEYS:
                continue
            
            new_path = f"{path}.{key}" if path else key
            
            # If value is a string, add it
            if isinstance(value, str) and value.strip():
                extracted.append(value.strip())
            # If value is a list of strings, add all
            elif isinstance(value, list) and all(isinstance(v, str) for v in value):
                extracted.extend([v.strip() for v in value if v and v.strip()])
            # Recurse for nested structures
            elif isinstance(value, (dict, list)):
                extracted.extend(_recursive_text_extractor(
                    value, new_path, max_depth, current_depth + 1
                ))
        
        # Special handling for common nested structures
        # Work experience arrays
        for exp_key in ["work_experience", "career_history", "experience", "employment_history"]:
            if exp_key in obj and isinstance(obj[exp_key], list):
                for exp in obj[exp_key]:
                    if isinstance(exp, dict):
                        # Extract description fields
                        for desc_key in ["description", "responsibilities", "achievements", "summary", "details"]:
                            if desc_key in exp and isinstance(exp[desc_key], str) and exp[desc_key].strip():
                                extracted.append(exp[desc_key].strip())
                        # Extract title and company
                        for title_key in ["title", "position", "role", "job_title"]:
                            if title_key in exp and isinstance(exp[title_key], str) and exp[title_key].strip():
                                extracted.append(exp[title_key].strip())
                        for co_key in ["company", "employer", "organization"]:
                            if co_key in exp and isinstance(exp[co_key], str) and exp[co_key].strip():
                                extracted.append(exp[co_key].strip())
        
        # Projects arrays
        for proj_key in ["projects", "project", "portfolio"]:
            if proj_key in obj and isinstance(obj[proj_key], list):
                for proj in obj[proj_key]:
                    if isinstance(proj, dict):
                        for desc_key in ["description", "summary", "details", "impact", "outcome"]:
                            if desc_key in proj and isinstance(proj[desc_key], str) and proj[desc_key].strip():
                                extracted.append(proj[desc_key].strip())
                        if "name" in proj and isinstance(proj["name"], str) and proj["name"].strip():
                            extracted.append(proj["name"].strip())
        
        # Skills arrays
        for skills_key in ["skills", "skill", "technical_skills", "core_competencies"]:
            if skills_key in obj:
                skills_data = obj[skills_key]
                if isinstance(skills_data, list):
                    for skill in skills_data:
                        if isinstance(skill, str) and skill.strip():
                            extracted.append(skill.strip())
                        elif isinstance(skill, dict) and "name" in skill:
                            if isinstance(skill["name"], str) and skill["name"].strip():
                                extracted.append(skill["name"].strip())
        
        return extracted
    
    # Handle other types (not expected)
    return extracted


def _discover_keys(obj: Any, path: str = "", max_depth: int = 5, current_depth: int = 0) -> Set[str]:
    """
    Discover all keys in a nested object structure.
    Useful for debugging and understanding the data schema.
    """
    keys: Set[str] = set()
    
    if current_depth > max_depth:
        return keys
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            full_path = f"{path}.{key}" if path else key
            keys.add(full_path)
            if isinstance(value, (dict, list)):
                keys.update(_discover_keys(value, full_path, max_depth, current_depth + 1))
    elif isinstance(obj, list) and obj:
        # Sample first item to discover keys
        sample = obj[0]
        if isinstance(sample, (dict, list)):
            keys.update(_discover_keys(sample, f"{path}[0]", max_depth, current_depth + 1))
    
    return keys


def _extract_all_text_from_candidate(cand: Dict[str, Any]) -> str:
    """
    Extract ALL text from a candidate object using recursive extraction.
    Returns a single concatenated string of all text content.
    """
    # Extract text recursively
    text_parts = _recursive_text_extractor(cand)
    
    # Also extract from raw if present
    if "raw" in cand and isinstance(cand["raw"], dict):
        raw_text = _recursive_text_extractor(cand["raw"])
        text_parts.extend(raw_text)
    
    # Remove duplicates and join
    seen = set()
    unique_parts = []
    for part in text_parts:
        if part not in seen:
            seen.add(part)
            unique_parts.append(part)
    
    combined_text = " ".join(unique_parts).lower()
    return combined_text


# ============================================================================
# END OF ROBUST TEXT EXTRACTION
# ============================================================================

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_career_project_text(cand: Dict[str, Any]) -> str:
    """
    Extract text from candidate with special focus on career/project descriptions.
    This is the high-trust corpus for retrieval evidence scoring.
    """
    # Use the comprehensive text extractor
    return _extract_all_text_from_candidate(cand)


def _extract_titles_companies(cand: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    titles: List[str] = []
    companies: List[str] = []
    
    # Try multiple possible paths for work experience
    work_exp_paths = ["career_history", "work_experience", "experience", "employment_history", "work"]
    
    for path in work_exp_paths:
        work_exp = cand.get(path, [])
        if work_exp and isinstance(work_exp, list):
            for exp in work_exp:
                if not isinstance(exp, dict):
                    continue
                t = exp.get("title") or exp.get("position") or exp.get("role") or exp.get("job_title")
                if t and isinstance(t, str):
                    titles.append(str(t).strip())
                c = exp.get("company") or exp.get("employer") or exp.get("organization")
                if c and isinstance(c, str):
                    companies.append(str(c).strip())
            break
    
    # Also check profile for current title
    profile = cand.get("profile", {})
    if isinstance(profile, dict):
        t = profile.get("title") or profile.get("job_title") or profile.get("position")
        if t and isinstance(t, str):
            titles.append(str(t).strip())
        c = profile.get("company") or profile.get("current_company")
        if c and isinstance(c, str):
            companies.append(str(c).strip())
    
    # Clean up
    titles = [t for t in titles if t]
    companies = [c for c in companies if c]
    return titles, companies


def _extract_skills_list(cand: Dict[str, Any]) -> List[str]:
    skills: List[str] = []
    
    # Try multiple possible paths for skills
    skills_paths = ["skills", "skill", "technical_skills", "core_competencies", "competencies"]
    
    for path in skills_paths:
        skills_raw = cand.get(path, [])
        if skills_raw:
            if isinstance(skills_raw, list):
                for s in skills_raw:
                    if isinstance(s, dict):
                        name = s.get("name", "").strip().lower()
                        if name:
                            skills.append(name)
                    elif isinstance(s, str):
                        cleaned = s.strip().lower()
                        if cleaned:
                            skills.append(cleaned)
            elif isinstance(skills_raw, str):
                # Comma-separated skills
                for s in skills_raw.split(","):
                    cleaned = s.strip().lower()
                    if cleaned:
                        skills.append(cleaned)
            break
    
    # Also check profile
    profile = cand.get("profile", {})
    if isinstance(profile, dict):
        profile_skills = profile.get("skills", [])
        if profile_skills:
            if isinstance(profile_skills, list):
                for s in profile_skills:
                    if isinstance(s, dict):
                        name = s.get("name", "").strip().lower()
                        if name:
                            skills.append(name)
                    elif isinstance(s, str):
                        cleaned = s.strip().lower()
                        if cleaned:
                            skills.append(cleaned)
            elif isinstance(profile_skills, str):
                for s in profile_skills.split(","):
                    cleaned = s.strip().lower()
                    if cleaned:
                        skills.append(cleaned)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_skills = []
    for s in skills:
        if s not in seen:
            seen.add(s)
            unique_skills.append(s)
    
    return unique_skills


def _score_career_retrieval_evidence(career_text: str) -> Tuple[float, List[str]]:
    """
    Score retrieval evidence from career/project text only.
    Returns (score 0-1, list of matched evidence phrases).
    """
    if not career_text:
        return 0.0, []

    matched_evidence: List[str] = []

    # Keyword hits - check for exact matches and substrings
    for kw in _RETRIEVAL_CAREER_KEYWORDS:
        if kw in career_text:
            matched_evidence.append(kw)
    
    # Also check for implicit evidence (shorter phrases that might appear)
    implicit_retrieval = [
        "search", "ranking", "retrieval", "recommend", "personalization",
        "relevance", "vector", "embedding", "ann", "ltr", "rerank"
    ]
    for kw in implicit_retrieval:
        if kw in career_text:
            # Only add if not already matched
            if kw not in matched_evidence and len(kw) >= 4:
                matched_evidence.append(f"implicit:{kw}")

    # Regex pattern hits
    for pattern in _RETRIEVAL_CAREER_PATTERNS:
        m = re.search(pattern, career_text, re.IGNORECASE)
        if m:
            snippet = career_text[max(0, m.start()-10):m.end()+20].strip()
            matched_evidence.append(f"~{snippet[:60]}")

    # Deduplicate
    matched_evidence = list(dict.fromkeys(matched_evidence))

    if not matched_evidence:
        return 0.0, []

    # Score: log-compressed based on number of distinct evidence signals
    n = len(matched_evidence)
    # 1 signal → ~0.25, 3 signals → ~0.55, 6 signals → ~0.80, 10+ → ~0.95
    score = 1.0 - math.exp(-0.25 * n)
    return min(score, 1.0), matched_evidence


def _score_prod_evidence(career_text: str) -> float:
    if not career_text:
        return 0.0
    hits = sum(1 for kw in _PROD_ML_CAREER_KEYWORDS if kw in career_text)
    return min(hits / 4.0, 1.0)  # 4+ hits = full score


def _has_retrieval_title(titles: List[str]) -> bool:
    for t in titles:
        tl = t.lower()
        if any(kw in tl for kw in _RETRIEVAL_TITLE_KEYWORDS):
            return True
    return False


def _has_ml_title(titles: List[str]) -> bool:
    for t in titles:
        tl = t.lower()
        if any(kw in tl for kw in _ML_TITLE_KEYWORDS):
            return True
    return False


def _is_disqualifying_role(titles: List[str]) -> bool:
    """Returns True only if ALL titles are disqualifying (career evidence can override)."""
    if not titles:
        return False
    disq_count = 0
    for t in titles:
        tl = t.lower()
        if any(role in tl for role in _DISQUALIFYING_ROLES):
            disq_count += 1
    return disq_count == len(titles)


def _has_strong_disqualifier(titles: List[str]) -> float:
    """
    Check if candidate has strongly disqualifying roles.
    Returns penalty multiplier (0.0 to 1.0) based on worst role.
    """
    if not titles:
        return 1.0
    
    max_penalty = 1.0
    for title in titles:
        tl = title.lower()
        for bad_role in _STRONG_DISQUALIFIER_ROLES:
            if bad_role in tl:
                max_penalty = min(max_penalty, 0.40)
    
    return max_penalty


def _classify_companies(companies: List[str]) -> Tuple[bool, bool, float]:
    """Returns (has_product_company, is_consulting_only, product_company_score)."""
    all_text = " ".join(companies).lower()
    has_product = any(c in all_text for c in _PRODUCT_COMPANIES)
    has_consulting = any(c in all_text for c in _CONSULTING_FIRMS)
    is_consulting_only = has_consulting and not has_product
    
    product_score = 0.0
    for company in companies:
        cl = company.lower()
        if any(pc in cl for pc in _PRODUCT_COMPANIES):
            product_score += 0.3
    product_score = min(product_score, 1.0)
    
    return has_product, is_consulting_only, product_score


def _extract_years(cand: Dict[str, Any]) -> float:
    for key in ("years_of_experience", "yoe", "total_experience_years", "experience_years"):
        v = cand.get(key)
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            num = "".join(c for c in v if c.isdigit() or c == ".")
            if num:
                try:
                    return float(num)
                except ValueError:
                    pass
    
    profile = cand.get("profile", {})
    if isinstance(profile, dict):
        v = profile.get("years_of_experience")
        if isinstance(v, (int, float)):
            return float(v)
    
    # Infer from work experience durations
    work_exp_paths = ["career_history", "work_experience", "experience", "employment_history"]
    total = 0.0
    for path in work_exp_paths:
        work_exp = cand.get(path, [])
        if work_exp and isinstance(work_exp, list):
            for exp in work_exp:
                if not isinstance(exp, dict):
                    continue
                dur = exp.get("duration_months") or exp.get("duration") or exp.get("months")
                if isinstance(dur, (int, float)):
                    total += float(dur) / 12.0
            break
    
    if total > 0:
        return total
    
    # Try to extract from summary
    summary = cand.get("summary", "") or cand.get("bio", "") or ""
    if summary:
        m = re.search(r"(\d+)\+?\s*(?:years?|yrs?)", summary.lower())
        if m:
            return float(m.group(1))
    
    return 5.0


def _extract_behavioral(cand: Dict[str, Any]) -> Tuple[float, bool, int, float, float]:
    """Returns (behavioral_score, open_to_work, notice_days, recruiter_rate, inactive_days)."""
    signals = cand.get("resume_signals", {}) or {}
    open_to_work = bool(signals.get("open_to_work_flag") or cand.get("open_to_work"))
    recruiter_rate = float(signals.get("recruiter_response_rate", 0.5))

    # Notice period
    raw_notice = (
        cand.get("notice_period") or cand.get("availability") or
        (cand.get("profile", {}) or {}).get("notice_period") or ""
    )
    notice_days = -1
    if raw_notice:
        s = str(raw_notice).lower()
        if any(x in s for x in ["immediate", "0 day", "now", "instantly"]):
            notice_days = 0
        else:
            m = re.search(r"(\d+)\s*(day|week|month)", s)
            if m:
                n, unit = int(m.group(1)), m.group(2)
                notice_days = n * (7 if unit.startswith("week") else 30 if unit.startswith("month") else 1)
            else:
                m2 = re.search(r"(\d+)", s)
                if m2:
                    notice_days = int(m2.group(1))

    # Inactivity
    _REFERENCE_DATE = datetime(2025, 1, 1)

    last_active = signals.get("last_active_days") or cand.get("last_active")
    inactive_days = 90.0
    if isinstance(last_active, (int, float)):
        inactive_days = float(last_active)
    elif isinstance(last_active, str):
        try:
            from dateutil import parser as dateparser
            dt = dateparser.parse(last_active)
            inactive_days = float((_REFERENCE_DATE - dt).days)
        except Exception:
            inactive_days = float(signals.get("inactivity_days", 90))

    is_active = inactive_days < 60
    notice_score = 1.0 if notice_days == 0 else (0.7 if notice_days <= 30 else 0.4 if notice_days <= 60 else 0.1)
    behavioral_score = (
        0.30 * (1.0 if open_to_work else 0.2) +
        0.25 * (1.0 if is_active else max(0.0, 1.0 - inactive_days / 365)) +
        0.25 * min(recruiter_rate, 1.0) +
        0.20 * notice_score
    )
    return min(behavioral_score, 1.0), open_to_work, notice_days, recruiter_rate, inactive_days


def _python_score(career_text: str, skills: List[str]) -> float:
    """Python score: requires both Python AND at least one ML library in career/project text."""
    has_python = "python" in career_text or "python" in skills
    ml_libs: frozenset = frozenset({
        "numpy", "pandas", "scikit-learn", "sklearn", "pytorch", "tensorflow",
        "jax", "transformers", "huggingface", "keras", "scipy", "xgboost", "lightgbm",
    })
    has_ml_lib = any(lib in career_text for lib in ml_libs) or any(lib in skills for lib in ml_libs)
    if has_python and has_ml_lib:
        return 1.0
    if has_python:
        return 0.5
    return 0.0


def _detect_honeypot(
    career_evidence_score: float,
    skills: List[str],
    career_text: str,
    is_disq: bool,
    is_consulting_only: bool,
) -> Tuple[bool, float, List[str]]:
    """
    Detect inflated/fake profiles and return a SOFT penalty factor.
    Returns: (is_honeypot, penalty_factor, reasons)
    """
    reasons: List[str] = []
    penalty = 1.0
    
    if len(skills) > 50 and career_evidence_score < 0.15:
        reasons.append("skill_stuffing_no_career_proof")
        penalty *= 0.85
    
    if "langchain" in career_text and career_evidence_score < 0.2:
        if not any(kw in career_text for kw in ("production", "deploy", "serving", "scale")):
            reasons.append("langchain_only_no_prod")
            penalty *= 0.80
    
    buzz_hits = sum(1 for bw in _AI_BUZZWORD_SET if bw in career_text)
    if buzz_hits >= 4 and career_evidence_score < 0.1:
        reasons.append("ai_buzzwords_no_substance")
        penalty *= 0.75
    
    if is_consulting_only:
        reasons.append("consulting_only_career")
    
    is_honeypot = penalty < 0.9 or bool(reasons)
    return is_honeypot, penalty, reasons


def _compute_retrieval_gate(
    career_evidence_score: float,
    has_ret_title: bool,
    has_ml_title: bool,
    matched_evidence: List[str],
    years: float,
    is_disq: bool,
    is_consulting_only: bool,
    skills: List[str],
    product_company_score: float,
) -> Tuple[bool, float]:
    """
    Soft retrieval gate that gradually penalizes candidates.
    """
    n_evidence = len(matched_evidence)
    gate_penalty = 0.68
    
    if is_disq and career_evidence_score < 0.15:
        gate_penalty = 0.60
    elif is_consulting_only and career_evidence_score < 0.15:
        gate_penalty = 0.62
    
    if career_evidence_score >= 0.40 and n_evidence >= 4:
        gate_penalty = 1.00
    elif career_evidence_score >= 0.30 and n_evidence >= 3:
        gate_penalty = 0.95
    elif career_evidence_score >= 0.25 and n_evidence >= 2:
        gate_penalty = 0.88
    elif career_evidence_score >= 0.18 and n_evidence >= 1:
        gate_penalty = 0.80
    elif career_evidence_score >= 0.10:
        gate_penalty = 0.74
    
    if has_ret_title and gate_penalty < 0.85:
        gate_penalty = min(gate_penalty + 0.10, 0.90)
    
    if has_ml_title and gate_penalty < 0.80:
        gate_penalty = min(gate_penalty + 0.08, 0.85)
    
    if product_company_score >= 0.5 and gate_penalty < 0.85:
        gate_penalty = min(gate_penalty + 0.10, 0.92)
    elif product_company_score >= 0.3 and gate_penalty < 0.80:
        gate_penalty = min(gate_penalty + 0.05, 0.85)
    
    if years >= 5 and gate_penalty < 0.85:
        gate_penalty = min(gate_penalty + 0.05, 0.90)
    elif years >= 8 and gate_penalty < 0.90:
        gate_penalty = min(gate_penalty + 0.03, 0.93)
    
    gate_penalty = max(gate_penalty, 0.60)
    gate_passed = gate_penalty >= 0.75
    
    return gate_passed, round(gate_penalty, 3)


def extract_candidate_features(cand: Dict[str, Any]) -> Dict[str, Any]:
    """Extract all features from a candidate with robust text extraction."""
    
    # ============================================================
    #  DEBUG VALIDATION - Log first 10 candidates
    # ============================================================
    # Use a global counter to track candidate processing
    if not hasattr(extract_candidate_features, "_counter"):
        extract_candidate_features._counter = 0
    extract_candidate_features._counter += 1
    
    # --- Basic extraction ---
    cid = str(cand.get("candidate_id") or cand.get("id") or "")
    
    # ============================================================
    # Robust text extraction from ALL fields
    # ============================================================
    career_text = _extract_all_text_from_candidate(cand)
    
    # Extract titles and companies using robust extraction
    titles, companies = _extract_titles_companies(cand)
    skills = _extract_skills_list(cand)
    years = _extract_years(cand)
    
    # ============================================================
    # Debug logging for first 10 candidates
    # ============================================================
    if extract_candidate_features._counter <= 10:
        # Discover keys in the candidate for debugging
        discovered_keys = _discover_keys(cand)
        key_sample = sorted(list(discovered_keys))[:20]  # Show first 20 keys
        
        print(f"\n{'='*60}")
        print(f"[DEBUG] Candidate #{extract_candidate_features._counter}")
        print(f"  ID: {cid}")
        print(f"  Text length: {len(career_text)} characters")
        print(f"  Number of skills: {len(skills)}")
        print(f"  Titles: {titles}")
        print(f"  Companies: {companies}")
        print(f"  Years experience: {years}")
        print(f"  Sample keys found: {key_sample}")
        print(f"{'='*60}\n")
    
    # --- TIER 2: Full text blob (for semantic embedding only) ---
    full_text = " ".join(filter(None, [
        " ".join(titles), " ".join(companies), " ".join(skills), career_text,
    ])).lower()

    # --- Retrieval evidence — scored from career/project text ONLY ---
    career_evidence_score, matched_evidence = _score_career_retrieval_evidence(career_text)

    # --- Supplementary domain scores ---
    prod_evidence = _score_prod_evidence(career_text)
    py_score = _python_score(career_text, skills)

    # --- Title / company classification ---
    has_ret_title = _has_retrieval_title(titles)
    has_ml_title_flag = _has_ml_title(titles)
    is_disq_role = _is_disqualifying_role(titles)
    has_product_co, is_consulting_only, product_company_score = _classify_companies(companies)
    strong_disq_penalty = _has_strong_disqualifier(titles)

    # --- Career relevance composite ---
    career_relevance = 0.0
    if has_ret_title:
        career_relevance += 0.50
    if has_ml_title_flag and not has_ret_title:
        career_relevance += 0.20
    if career_evidence_score >= 0.30:
        career_relevance += 0.30
    elif career_evidence_score >= 0.15:
        career_relevance += 0.15
    if product_company_score >= 0.5:
        career_relevance += 0.20
    elif product_company_score >= 0.3:
        career_relevance += 0.10
    
    if is_disq_role and career_evidence_score < 0.20:
        career_relevance = max(career_relevance - 0.15, 0.0)
    
    if strong_disq_penalty < 0.5:
        career_relevance = career_relevance * strong_disq_penalty
    
    career_relevance = min(career_relevance, 1.0)

    # --- Hard retrieval gate ---
    retrieval_gate_passed, gate_penalty = _compute_retrieval_gate(
        career_evidence_score, 
        has_ret_title, 
        has_ml_title_flag,
        matched_evidence,
        years, 
        is_disq_role, 
        is_consulting_only, 
        skills,
        product_company_score,
    )

    # --- Honeypot detection ---
    is_honeypot, honey_penalty, honey_reasons = _detect_honeypot(
        career_evidence_score, skills, career_text, is_disq_role, is_consulting_only
    )
    
    if is_honeypot:
        gate_penalty = gate_penalty * honey_penalty
        gate_penalty = max(gate_penalty, 0.60)

    # --- Behavioral ---
    behavioral_score, open_to_work, notice_days, recruiter_rate, inactive_days = _extract_behavioral(cand)
    is_active = inactive_days < 60

    # --- Education ---
    has_ml_degree = False
    education_data = cand.get("education", [])
    if education_data and isinstance(education_data, list):
        for edu in education_data:
            if not isinstance(edu, dict):
                continue
            field = (edu.get("field_of_study") or edu.get("major") or edu.get("degree") or "").lower()
            if any(kw in field for kw in ("machine learning", "computer science", "statistics", "data science", "ai")):
                has_ml_degree = True
                break

    # ============================================================
    # Debug logging for retrieval evidence (first 10 candidates)
    # ============================================================
    if extract_candidate_features._counter <= 10:
        print(f"[DEBUG] Candidate #{extract_candidate_features._counter} results:")
        print(f"  career_evidence_score: {career_evidence_score:.4f}")
        print(f"  matched_evidence count: {len(matched_evidence)}")
        print(f"  matched_evidence sample: {matched_evidence[:3]}")
        print(f"  retrieval_gate_passed: {retrieval_gate_passed}")
        print(f"  gate_penalty: {gate_penalty}")
        print(f"  prod_evidence: {prod_evidence:.4f}")
        print(f"  career_relevance: {career_relevance:.4f}")
        print(f"{'='*60}\n")

    return {
        # Identity
        "candidate_id": cid,
        "name": cand.get("name", "") or (cand.get("profile", {}).get("anonymized_name", "")),
        "skills": skills,
        "titles": titles,
        "companies": companies,
        "years_experience": years,
        "has_ml_degree": has_ml_degree,
        # Core evidence signals
        "career_evidence_score": career_evidence_score,
        "matched_evidence": matched_evidence[:8],
        "retrieval_gate_passed": retrieval_gate_passed,
        "gate_penalty": gate_penalty,
        "prod_evidence": prod_evidence,
        "python_score": py_score,
        # Career / company signals
        "career_relevance": career_relevance,
        "has_retrieval_title": has_ret_title,
        "has_ml_title": has_ml_title_flag,
        "has_product_company": has_product_co,
        "product_company_score": product_company_score,
        "is_disq_role": is_disq_role,
        "is_consulting_only": is_consulting_only,
        "strong_disq_penalty": strong_disq_penalty,
        # Aliases expected by ranker.py
        "retrieval_ranking_score": career_evidence_score,
        "retrieval_evidence": career_evidence_score,
        "vector_evidence": 1.0 if any(kw in career_text for kw in (
            "faiss", "hnsw", "qdrant", "weaviate", "pinecone", "milvus",
            "vector search", "vector index", "pgvector",
        )) else 0.0,
        "eval_evidence": 1.0 if any(kw in career_text for kw in (
            "ndcg", "mrr", "map@", "precision@", "recall@", "mean average precision",
        )) else 0.0,
        "has_fake_ai": "ai_buzzwords_no_substance" in honey_reasons,
        "has_langchain_only": "langchain_only_no_prod" in honey_reasons,
        "is_generic_tech": is_disq_role,
        "has_real_prod": prod_evidence >= 0.5,
        "recent_relevant": has_ret_title or career_evidence_score >= 0.20,
        # Behavioral
        "behavioral_score": behavioral_score,
        "is_active": is_active,
        "inactive_days": inactive_days,
        "open_to_work": open_to_work,
        "notice_days": notice_days,
        "recruiter_response_rate": recruiter_rate,
        # Honeypot
        "is_honeypot": is_honeypot,
        "honeypot_reasons": honey_reasons,
        # Text for semantic embedding (full corpus)
        "text": full_text,
        "raw": cand,
    }


def skill_overlap(candidate_skills: List[str], jd_features: Dict[str, Any]) -> float:
    jd_skills = set(jd_features.get("key_skills", []))
    must_have = set(jd_features.get("must_have_lower", []))
    if not jd_skills and not must_have:
        return 0.3
    cand_set = set(s.lower().strip() for s in candidate_skills if s)
    matched_must = {m for m in must_have if any(m in cs or cs in m for cs in cand_set)}
    matched_skills = {js for js in jd_skills if any(js in cs or cs in js for cs in cand_set)}
    must_recall = len(matched_must) / len(must_have) if must_have else 0.0
    skill_recall = len(matched_skills) / len(jd_skills) if jd_skills else 0.0
    union = len(jd_skills | cand_set)
    jaccard = len(matched_skills) / union if union else 0.0
    return min(0.50 * must_recall + 0.30 * skill_recall + 0.20 * jaccard, 1.0)