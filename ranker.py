"""Ranker — Stage 1 independent scoring + Stage 2 pairwise tournament.

Stage 1 weights (per-candidate, independent) - REBALANCED:
  25%  semantic_similarity     — JD semantic match
  25%  prod_evidence           — Production ML deployment (NEW)
  30%  career_evidence_score   — retrieval/ranking proof from career text
  15%  product_company_score   — Product-company background (NEW)
   5%  behavioral_score        — availability / engagement (MINIMAL)

  + up to 5% experience bonus (sigmoid vs JD min_years)

  Strong disqualifiers apply meaningful penalties:
  - Strong disqualifier role: 0.4x to 0.6x penalty
  - Consulting-only: 0.7x penalty
  - Generic tech without retrieval evidence: 0.6x penalty
"""
from __future__ import annotations
import math
from typing import Any, Dict, List

import numpy as np
from embeddings import compute_similarities
from feature_engineering import skill_overlap
from pairwise_ranker import pairwise_rerank, ELITE_POOL_SIZE


def _exp_fit(years: float, min_years: int) -> float:
    if min_years <= 0:
        return min(years / 15.0, 1.0)
    return 1.0 / (1.0 + math.exp(-0.6 * (years - min_years)))


def _calibrate(score: float) -> float:
    """Calibrate scores to spread out the top candidates."""
    if score >= 0.70:
        return 0.80 + 0.19 * (1.0 - math.exp(-5.0 * (score - 0.70)))
    if score >= 0.45:
        return 0.50 + 0.30 * ((score - 0.45) / 0.25)
    if score >= 0.20:
        return 0.15 + 0.35 * ((score - 0.20) / 0.25)
    return score * 0.5


def _apply_disqualifier_penalties(cand: Dict[str, Any], base_score: float) -> float:
    """Apply meaningful penalties for disqualifying factors."""
    score = base_score
    
    # Strong disqualifier role - severe penalty
    strong_penalty = cand.get("strong_disq_penalty", 1.0)
    if strong_penalty < 0.5:
        # Severe penalty for clearly irrelevant roles
        score = score * strong_penalty
        # Additional penalty if no retrieval evidence
        if cand.get("career_evidence_score", 0.0) < 0.15:
            score = score * 0.6
    
    # Consulting-only with no product experience
    if cand.get("is_consulting_only", False):
        if cand.get("product_company_score", 0.0) < 0.3:
            score = score * 0.7
    
    # Generic tech role without retrieval evidence
    if cand.get("is_disq_role", False) and cand.get("career_evidence_score", 0.0) < 0.15:
        score = score * 0.6
    
    # Honeypot detection
    if cand.get("is_honeypot", False):
        score = score * 0.7
    
    return score


def rank_candidates(
    jd_features: Dict[str, Any],
    candidate_features: List[Dict[str, Any]],
    top_n: int = 100,
) -> List[Dict[str, Any]]:
    if not candidate_features:
        return []

    jd_text     = jd_features.get("text", "")
    min_years   = int(jd_features.get("min_years", 5))
    cand_texts  = [c.get("text", "") for c in candidate_features]

    print(f"[ranker] Stage 1 — scoring {len(cand_texts)} candidates …")
    sims = compute_similarities(jd_text, cand_texts)

    results: List[Dict[str, Any]] = []
    for i, cand in enumerate(candidate_features):
        semantic     = float(sims[i]) if i < len(sims) else 0.0
        career_ev    = float(cand.get("career_evidence_score", 0.0))
        prod_ev      = float(cand.get("prod_evidence", 0.0))
        product_score = float(cand.get("product_company_score", 0.0))
        behavioral   = float(cand.get("behavioral_score", 0.3))
        sk_overlap   = skill_overlap(cand.get("skills", []), jd_features)
        years        = float(cand.get("years_experience", 5.0))
        gate_penalty = float(cand.get("gate_penalty", 0.68))

        # ── Weighted base (REBALANCED) ──
        base = (
            0.25 * semantic      # JD semantic match
            + 0.25 * prod_ev      # Production ML deployment (NEW)
            + 0.30 * career_ev    # Retrieval/ranking evidence
            + 0.15 * product_score # Product-company background (NEW)
            + 0.05 * behavioral   # Behavioral (MINIMAL)
        )
        # Experience bonus (minor)
        base = min(base + 0.05 * _exp_fit(years, min_years), 1.0)

        # ── Apply disqualifier penalties ──
        base = _apply_disqualifier_penalties(cand, base)

        # ── Combine with gate ──
        stage1 = _calibrate(base * 0.85 + gate_penalty * 0.15)

        results.append({
            **cand,
            "semantic_score":           round(semantic, 4),
            "career_evidence_score":    round(career_ev, 4),
            "prod_evidence":            round(prod_ev, 4),
            "product_company_score":    round(product_score, 4),
            "skill_overlap_score":      round(sk_overlap, 4),
            "gate_penalty":             round(gate_penalty, 3),
            "behavioral_score":         round(behavioral, 4),
            "raw_score":                round(base, 6),
            "stage1_score":             round(stage1, 6),
            "score":                    round(stage1, 6),
            "retrieval_ranking_score":  round(career_ev, 4),
            "python_score":             cand.get("python_score", 0.0),
            "penalty_multiplier":       round(gate_penalty, 3),
            "borda_score":              0.0,
            "borda_delta":              0.0,
            "pairwise_advantages":      [],
            "pairwise_run":             False,
            "strong_disq_penalty":      cand.get("strong_disq_penalty", 1.0),
        })

    # ── Stage 1 sort ──
    results.sort(
        key=lambda x: (
            -round(x["score"], 6),
            -round(x.get("career_evidence_score", 0.0), 6),
            -round(x.get("prod_evidence", 0.0), 6),
            str(x.get("candidate_id", "")),
        )
    )
    top_s1 = results[0]["score"] if results else 0
    print(f"[ranker] Stage 1 complete. Top score: {top_s1:.4f}")

    # ── Stage 2: pairwise tournament ──
    print("[ranker] Stage 2 — pairwise tournament …")
    results = pairwise_rerank(results, top_n=top_n, jd_min_years=min_years)
    results.sort(
        key=lambda x: (
            -round(x["score"], 6),
            -round(x.get("career_evidence_score", 0.0), 6),
            -round(x.get("prod_evidence", 0.0), 6),
            str(x.get("candidate_id", "")),
        )
    )

    top_s2 = results[0]["score"] if results else 0
    print(f"[ranker] Stage 2 complete. Final top score: {top_s2:.4f}")

    return results[:top_n]