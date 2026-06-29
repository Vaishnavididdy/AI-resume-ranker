"""
Stage 2 pairwise reranker.

Compares top candidates across multiple signals to refine
ordering within close score ranges.
"""
from __future__ import annotations
import math
import re
from typing import Any, Dict, List, Set, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ELITE_POOL_SIZE: int = 25   # candidates entering Stage 2
ALPHA: float = 0.08         # max score shift ±

# Axis weights — must sum to 1.0
# REBALANCED: retrieval & semantic dominance, behavioral minimized
_W: Dict[str, float] = {
    "retrieval_depth":  0.40,  # Primary: retrieval/ranking evidence
    "semantic_match":   0.25,  # JD semantic similarity (from Stage 1)
    "prod_quality":     0.15,  # Production ML deployment
    "ml_engineering":   0.10,  # ML engineering depth
    "seniority":        0.05,  # Experience (minimal)
    "availability":     0.05,  # Behavioral (minimal)
}
assert abs(sum(_W.values()) - 1.0) < 1e-9, "Axis weights must sum to 1.0"

# Elite keywords used for axis-wise bonus scoring
_RETRIEVAL_ELITE: frozenset = frozenset({
    # Learning-to-rank
    "learning to rank", "ltr", "lambdarank", "listwise ranking", "pointwise ranking",
    "pairwise ranking", "ranknet", "xgboost rank", "lightgbm rank",
    # Classic IR
    "bm25", "tf-idf ranking", "inverted index", "solr", "elasticsearch ranking",
    "opensearch ranking",
    # Hybrid / multi-stage
    "hybrid search", "hybrid retrieval", "multi-stage ranking", "cascade ranking",
    "first-stage retrieval", "second-stage ranking", "recall then rank",
    # Dense retrieval
    "dense retrieval", "bi-encoder retrieval", "dual-encoder retrieval",
    "dense passage retrieval", "dpr",
    # Re-ranking
    "cross-encoder rerank", "reranking pipeline", "neural reranker",
    # Query understanding
    "query understanding", "query rewriting", "query expansion", "query reformulation",
    # Relevance
    "search relevance", "relevance optimization", "relevance tuning",
    "search quality improvement", "recall optimization", "precision optimization",
    # Candidate/job matching
    "candidate matching", "job matching", "semantic matching pipeline",
    # Two-tower / siamese
    "two-tower model", "two tower", "siamese network retrieval",
    # Vector DBs (moved from vector_infra)
    "faiss", "hnsw", "annoy", "qdrant", "weaviate", "pinecone", "milvus",
    "pgvector", "chroma", "redis vector", "elastic vector", "opensearch vector",
    "vespa", "marqo",
    # ANN / indexing
    "approximate nearest neighbor", "ann index", "ivf index", "product quantization",
    "scalar quantization", "vector quantization", "embedding index",
    # Vector infra concepts
    "dense vector store", "vector similarity search", "embedding lookup",
    "semantic index", "embedding cache",
    # Evaluation metrics (moved from eval_framework)
    "ndcg", "ndcg@", "mrr", "mrr@", "map@", "mean average precision",
    "mean reciprocal rank", "normalized discounted cumulative gain",
    "precision@", "recall@", "hit rate@", "expected reciprocal rank",
    # Evaluation process
    "offline evaluation", "online evaluation", "a/b test",
    "interleaved experiment", "counterfactual evaluation",
    "relevance judgment", "editorial judgment", "human evaluation",
    "annotation pipeline", "qrel", "trec eval",
    # LTR evaluation
    "ranking metric", "ranking evaluation", "pairwise accuracy",
    "normalized kendall tau",
    # Click models
    "click-through rate", "ctr model", "dwell time", "implicit feedback",
    "click model", "position bias",
})

_PROD_ELITE: frozenset = frozenset({
    # Scale indicators
    "at scale", "millions of queries", "billions of", "million users",
    "high throughput", "low latency", "p99 latency", "p95 latency",
    "sub-millisecond", "real-time serving", "online serving",
    # Deployment patterns
    "production deployment", "production serving", "model serving",
    "inference pipeline", "feature store", "mlops pipeline",
    "canary deploy", "shadow mode", "traffic split", "blue-green",
    # Infrastructure
    "kubernetes", "k8s", "docker", "ray serve", "triton inference",
    "torchserve", "sagemaker endpoint", "vertex ai endpoint",
    # Reliability
    "model monitoring", "data drift", "concept drift", "retraining pipeline",
    "continuous training", "ci/cd ml", "model registry",
})

_ML_ENGINEERING_ELITE: frozenset = frozenset({
    # ML frameworks
    "pytorch", "tensorflow", "jax", "keras",
    # NLP / transformer
    "transformers", "huggingface", "sentence-transformers", "spacy",
    # Classic ML
    "scikit-learn", "sklearn", "xgboost", "lightgbm", "catboost",
    # Data
    "numpy", "pandas", "scipy", "polars",
    # Big data / distributed
    "ray", "spark ml", "pyspark", "dask",
    # Infra
    "fastapi", "grpc", "asyncio",
})

# DISQUALIFYING ROLES - Strong penalty for irrelevant roles
_IRRELEVANT_ROLES: frozenset = frozenset({
    "sales", "graphic designer", "operations", "support", "marketing",
    "business analyst", "project manager", "scrum master", "product manager",
    "data analyst", "analytics", "hr", "human resources", "recruiter",
    "finance", "accounting", "legal", "compliance", "admin", "administrative",
    "customer success", "customer service", "qa", "quality assurance",
    "manual tester", "ui/ux", "designer", "art director", "creative director",
})


# ---------------------------------------------------------------------------
# Axis scorers — each returns float [0, 1] for a single candidate
# ---------------------------------------------------------------------------

def _hit_score(text: str, keywords: Set[str], per_hit: float = 0.06) -> float:
    """Count keyword hits and compress to [0, 1] with diminishing returns."""
    hits = sum(1 for kw in keywords if kw in text)
    if hits == 0:
        return 0.0
    return min(1.0 - math.exp(-per_hit * hits * 12), 1.0)


def _ax_retrieval_depth(c: Dict[str, Any]) -> float:
    """Primary axis: retrieval/ranking evidence from career text."""
    base = float(c.get("career_evidence_score", 0.0))
    boost = _hit_score(c.get("text", ""), _RETRIEVAL_ELITE, per_hit=0.07)
    # Base drives most of the score, boost adds separation in the elite tier
    return min(0.65 * base + 0.35 * boost, 1.0)


def _ax_semantic_match(c: Dict[str, Any]) -> float:
    """JD semantic similarity (from Stage 1)."""
    return float(c.get("semantic_score", 0.0))


def _ax_prod_quality(c: Dict[str, Any]) -> float:
    """Production ML deployment evidence."""
    base = float(c.get("prod_evidence", 0.0))
    boost = _hit_score(c.get("text", ""), _PROD_ELITE, per_hit=0.07)
    return min(0.60 * base + 0.40 * boost, 1.0)


def _ax_ml_engineering(c: Dict[str, Any]) -> float:
    """ML engineering depth (Python, frameworks, etc.)."""
    base = float(c.get("python_score", 0.0))          # 0, 0.5, or 1.0
    breadth = _hit_score(c.get("text", ""), _ML_ENGINEERING_ELITE, per_hit=0.06)
    return min(0.55 * base + 0.45 * breadth, 1.0)


def _ax_seniority(c: Dict[str, Any], jd_min_years: int) -> float:
    """Experience relative to JD - MINIMAL WEIGHT."""
    yoe = float(c.get("years_experience", 0.0))
    if jd_min_years <= 0:
        return min(yoe / 15.0, 1.0)
    return 1.0 / (1.0 + math.exp(-0.5 * (yoe - jd_min_years)))


def _ax_availability(c: Dict[str, Any]) -> float:
    """Behavioral signals - MINIMAL WEIGHT."""
    return float(c.get("behavioral_score", 0.3))


def _get_irrelevant_penalty(c: Dict[str, Any]) -> float:
    """
    Check if candidate has irrelevant role titles.
    Returns penalty multiplier (0.0 to 1.0).
    Strong penalty for irrelevant roles with low retrieval evidence.
    """
    titles = c.get("titles", [])
    if not titles:
        return 1.0
    
    career_ev = float(c.get("career_evidence_score", 0.0))
    
    # Check each title for irrelevant keywords
    for title in titles:
        title_lower = title.lower()
        for bad_role in _IRRELEVANT_ROLES:
            if bad_role in title_lower:
                # If strong retrieval evidence, less penalty
                if career_ev >= 0.35:
                    return 0.70  # Mild penalty even with evidence
                elif career_ev >= 0.20:
                    return 0.50  # Moderate penalty
                else:
                    return 0.20  # Severe penalty - effectively removes from top 20
    
    return 1.0


# ---------------------------------------------------------------------------
# Per-axis human labels for reasoning output
# ---------------------------------------------------------------------------
_AXIS_LABELS: Dict[str, str] = {
    "retrieval_depth":  "retrieval/ranking system depth (BM25, LTR, hybrid search, vector DB)",
    "semantic_match":   "JD semantic similarity",
    "prod_quality":     "production ML deployment at scale",
    "ml_engineering":   "ML engineering depth (PyTorch, HuggingFace, etc.)",
    "seniority":        "experience relative to role requirements",
    "availability":     "recruiter availability and responsiveness",
}


def _axis_label(axis: str, diff: float) -> str:
    intensity = "significantly stronger" if abs(diff) >= 0.35 else "stronger"
    return f"{intensity} {_AXIS_LABELS[axis]}"


# ---------------------------------------------------------------------------
# Pairwise comparison: A vs B → preference for A ∈ [-1, +1]
# ---------------------------------------------------------------------------
def _compute_all_axes(
    c: Dict[str, Any],
    jd_min_years: int,
) -> Dict[str, float]:
    return {
        "retrieval_depth":  _ax_retrieval_depth(c),
        "semantic_match":   _ax_semantic_match(c),
        "prod_quality":     _ax_prod_quality(c),
        "ml_engineering":   _ax_ml_engineering(c),
        "seniority":        _ax_seniority(c, jd_min_years),
        "availability":     _ax_availability(c),
    }


def _compare_pair(
    axes_a: Dict[str, float],
    axes_b: Dict[str, float],
) -> Tuple[float, List[str]]:
    """
    Returns:
      preference ∈ [-1, +1]  (positive = A better, negative = B better)
      advantages: axes where A has a meaningful edge (diff >= 0.15)
    """
    weighted_sum = 0.0
    advantages: List[str] = []

    for axis, w in _W.items():
        diff = axes_a[axis] - axes_b[axis]   # positive = A is better on this axis

        # Soft-clip: suppress noise below 0.05
        if abs(diff) < 0.05:
            eff_diff = 0.0
        elif abs(diff) < 0.15:
            eff_diff = diff * 0.5             # half-credit for small differences
        else:
            eff_diff = diff                   # full credit

        weighted_sum += w * eff_diff

        # Record advantages for reasoning (only significant positives)
        if diff >= 0.15:
            advantages.append(_axis_label(axis, diff))

    # Normalise by max possible weighted sum (= sum of all weights = 1.0)
    preference = max(-1.0, min(1.0, weighted_sum))
    return preference, advantages


# ---------------------------------------------------------------------------
# Tournament
# ---------------------------------------------------------------------------
def _run_tournament(
    elite: List[Dict[str, Any]],
    jd_min_years: int,
) -> List[Dict[str, Any]]:
    n = len(elite)

    # Pre-compute all axis scores (avoids redundant keyword scans)
    all_axes = [_compute_all_axes(c, jd_min_years) for c in elite]
    
    # Pre-compute irrelevance penalties
    irrelevance_penalties = [_get_irrelevant_penalty(c) for c in elite]

    borda: List[float] = [0.0] * n
    win_axes: List[List[str]] = [[] for _ in range(n)]  # accumulated advantages per candidate

    for i in range(n):
        for j in range(i + 1, n):
            pref_ij, adv_ij = _compare_pair(all_axes[i], all_axes[j])
            
            # Apply irrelevance penalty to preference
            # If candidate i is irrelevant, reduce their preference
            # If candidate j is irrelevant, increase preference for i
            penalty_factor = irrelevance_penalties[i] / max(irrelevance_penalties[j], 0.01)
            # Clamp to avoid extreme values
            penalty_factor = max(0.3, min(3.0, penalty_factor))
            
            # Apply penalty: if candidate i is irrelevant, pref_ij is reduced
            # If candidate j is irrelevant, pref_ij is increased
            adjusted_pref = pref_ij * min(penalty_factor, 1.5)
            
            borda[i] += adjusted_pref
            borda[j] -= adjusted_pref
            
            if adjusted_pref > 0.05:
                win_axes[i].extend(adv_ij)
            elif adjusted_pref < -0.05:
                # B beats A — get B's advantages (reverse comparison)
                _, adv_ji = _compare_pair(all_axes[j], all_axes[i])
                win_axes[j].extend(adv_ji)

    # Normalise borda to [-1, +1]
    max_possible = float(n - 1)   # max borda if beat everyone with preference = 1.0
    if max_possible <= 0:
        return elite

    for i, cand in enumerate(elite):
        # Apply irrelevance penalty to the final delta as well
        base_norm = borda[i] / max_possible
        delta = base_norm * ALPHA
        
        # Additional cap: irrelevant candidates get at most ±0.01 delta
        if irrelevance_penalties[i] < 0.5:
            delta = max(-0.01, min(0.01, delta))
        elif irrelevance_penalties[i] < 0.8:
            delta = max(-0.03, min(0.03, delta))
        
        stage1 = float(cand.get("stage1_score", cand.get("score", 0.0)))
        new_score = max(0.0001, min(1.0, stage1 + delta))

        # Deduplicate and cap advantages list for reasoning
        seen: set = set()
        unique_adv: List[str] = []
        for adv in win_axes[i]:
            if adv not in seen:
                seen.add(adv)
                unique_adv.append(adv)

        elite[i] = {
            **cand,
            "borda_score":         round(base_norm, 4),
            "borda_delta":         round(delta, 6),
            "pairwise_advantages": unique_adv[:4],   # top 4 for readability
            "pairwise_run":        True,
            "score":               round(new_score, 6),
            "irrelevance_penalty": irrelevance_penalties[i],
        }

    return elite


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def pairwise_rerank(
    ranked: List[Dict[str, Any]],
    top_n: int = 100,
    jd_min_years: int = 5,
) -> List[Dict[str, Any]]:
    """
    Called by ranker.py after Stage 1 sort.

    Args:
        ranked:        full candidate list, sorted by stage1_score descending
        top_n:         how many to return
        jd_min_years:  minimum years from JD (for seniority axis)

    Returns:
        full list; elite candidates rescored, rest untouched
    """
    if len(ranked) < 2:
        return ranked

    elite_size = min(ELITE_POOL_SIZE, len(ranked))
    elite = ranked[:elite_size]
    rest  = ranked[elite_size:]

    # Stamp stage1_score before any modification
    for c in elite:
        c.setdefault("stage1_score", c.get("score", 0.0))
        c.setdefault("pairwise_run", False)
        c.setdefault("borda_score", 0.0)
        c.setdefault("borda_delta", 0.0)
        c.setdefault("pairwise_advantages", [])
        c.setdefault("irrelevance_penalty", 1.0)

    # Need at least 2 gate-passed candidates to run a meaningful tournament
    gate_passed_count = sum(
        1 for c in elite if c.get("retrieval_gate_passed", False)
    )
    if gate_passed_count < 2:
        print(f"[pairwise] Only {gate_passed_count} gate-passed candidates in elite pool "
              f"— skipping tournament")
        return ranked

    n_pairs = elite_size * (elite_size - 1) // 2
    print(f"[pairwise] Tournament: {elite_size} candidates × {n_pairs} comparisons "
          f"(jd_min_years={jd_min_years}) …")

    elite = _run_tournament(elite, jd_min_years)

    print(f"[pairwise] Done. Score range: "
          f"{min(c['score'] for c in elite):.4f} – {max(c['score'] for c in elite):.4f}")

    return elite + rest