"""
Builds candidate reasoning summary using evidence, strengths,
concerns, and final ranking scores.
"""
from __future__ import annotations
from typing import Any, Dict, List


def _tier(score: float, gate_passed: bool, is_honeypot: bool) -> str:
    if is_honeypot:
        return "SUSPICIOUS — likely fake or inflated profile"
    if score >= 0.85:
        return "ELITE — top-tier retrieval/ranking engineer; immediate interview"
    if score >= 0.70:
        return "STRONG FIT — clear retrieval/ranking evidence; recommend interview"
    if score >= 0.55:
        return "GOOD FIT — solid evidence; worth screening"
    if score >= 0.40:
        return "MODERATE — some retrieval signal; verify depth"
    return "WEAK — limited retrieval evidence; low priority"


def _recommendation(score: float, gate_passed: bool, is_honeypot: bool) -> str:
    if is_honeypot:
        return "Do not proceed"
    if score >= 0.70:
        return "Proceed with technical interview"
    if score >= 0.50:
        return "Consider — verify retrieval depth in screen call"
    if score >= 0.35:
        return "Pipeline — low priority"
    return "Not recommended"


def _evidence_narrative(matched: List[str]) -> str:
    if not matched:
        return "No retrieval/ranking evidence found in career or project descriptions"
    clean: List[str] = []
    for e in matched[:5]:
        e = e.strip("~").strip()
        if len(e) > 80:
            e = e[:80] + "…"
        clean.append(e)
    return "; ".join(clean)


def _strengths(cand: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    ev = cand.get("career_evidence_score", 0.0)
    if ev >= 0.60:
        out.append("strong retrieval/ranking systems evidence in career descriptions")
    elif ev >= 0.35:
        out.append("solid retrieval/ranking experience")
    elif ev >= 0.15:
        out.append("some retrieval/ranking exposure")

    # Production ML evidence
    prod_ev = cand.get("prod_evidence", 0.0)
    if prod_ev >= 0.6:
        out.append("strong production ML deployment experience at scale")
    elif prod_ev >= 0.3:
        out.append("some production ML deployment experience")

    # Product company background
    product_score = cand.get("product_company_score", 0.0)
    if product_score >= 0.5:
        out.append("product-company engineering background")

    if cand.get("vector_evidence", 0.0) >= 0.5:
        out.append("vector DB / embedding pipeline experience")
    if cand.get("eval_evidence", 0.0) >= 0.5:
        out.append("ranking evaluation expertise (NDCG / MRR / MAP)")
    if cand.get("python_score", 0.0) >= 1.0:
        out.append("Python with ML libraries")
    if cand.get("has_retrieval_title", False):
        out.append("job title confirms retrieval/ranking/recommendation role")
    
    semantic = cand.get("semantic_score", 0.0)
    if semantic >= 0.70:
        out.append("strong JD semantic match")
    elif semantic >= 0.50:
        out.append("good JD semantic match")
    
    pairwise_adv = cand.get("pairwise_advantages", [])
    if pairwise_adv:
        for adv in pairwise_adv[:2]:
            out.append(adv)
    
    return out


def _concerns(cand: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    
    # Strong disqualifier roles
    strong_penalty = cand.get("strong_disq_penalty", 1.0)
    if strong_penalty < 0.5:
        out.append("role does not align with retrieval/ranking requirements (sales, operations, design, etc.)")
    elif strong_penalty < 0.8:
        out.append("role has limited alignment with retrieval/ranking engineering")
    
    if cand.get("is_consulting_only", False):
        out.append("consulting-only background without product-company retrieval experience")
    if cand.get("is_disq_role", False) and cand.get("career_evidence_score", 0.0) < 0.20:
        out.append("generic tech role without strong retrieval evidence")
    if cand.get("has_fake_ai", False):
        out.append("AI buzzword inflation with limited career-level proof")
    if cand.get("has_langchain_only", False):
        out.append("LangChain-only without production evidence")
    if cand.get("years_experience", 0.0) < 4:
        out.append(f"only {cand['years_experience']:.0f}y experience (minimum 4y preferred)")
    if cand.get("is_honeypot", False):
        reasons = cand.get("honeypot_reasons", [])
        if reasons:
            out.append("flagged: " + ", ".join(reasons))
    
    return out


def _availability(cand: Dict[str, Any]) -> str:
    parts: List[str] = []
    if cand.get("open_to_work", False):
        parts.append("open to work")
    if cand.get("is_active", False):
        parts.append("recently active")
    else:
        d = cand.get("inactive_days", 0)
        parts.append(f"inactive {d:.0f}d")
    nd = cand.get("notice_days", -1)
    if nd == 0:
        parts.append("immediate availability")
    elif nd > 0:
        parts.append(f"{nd}d notice")
    rr = cand.get("recruiter_response_rate", 0.5)
    if rr >= 0.75:
        parts.append("highly responsive to recruiters")
    elif rr <= 0.30:
        parts.append("low recruiter responsiveness")
    return " | ".join(parts) if parts else "unknown"


def build_reasoning(cand: Dict[str, Any], jd_features: Dict[str, Any]) -> str:
    name = cand.get("name", "Candidate") or "Candidate"
    yoe = cand.get("years_experience", 0)
    titles = cand.get("titles", [])
    companies = cand.get("companies", [])
    title_str = titles[0] if titles else "unknown role"
    co_str = companies[0] if companies else ""
    header = f"{name} | {yoe:.0f}y exp | {title_str}" + (f" @ {co_str}" if co_str else "")

    matched = cand.get("matched_evidence", [])
    evidence = _evidence_narrative(matched)
    strengths = _strengths(cand)
    concerns = _concerns(cand)
    avail = _availability(cand)

    score = cand.get("score", 0.0)
    stage1 = cand.get("stage1_score", score)
    gate_passed = cand.get("retrieval_gate_passed", False)
    is_honeypot = cand.get("is_honeypot", False)
    borda_delta = cand.get("borda_delta", 0.0)
    pairwise_advantages = cand.get("pairwise_advantages", [])

    score_str = (
        f"career_ev={cand.get('career_evidence_score', 0):.2f} "
        f"prod_ev={cand.get('prod_evidence', 0):.2f} "
        f"product_score={cand.get('product_company_score', 0):.2f} "
        f"semantic={cand.get('semantic_score', 0):.2f} "
        f"gate={cand.get('gate_penalty', 0):.2f} "
        f"s1={stage1:.4f}"
    )
    if cand.get("pairwise_run", False):
        direction = "+" if borda_delta >= 0 else ""
        score_str += f" pairwise={direction}{borda_delta:.4f} final={score:.4f}"
    else:
        score_str += f" final={score:.4f}"

    assessment = _tier(score, gate_passed, is_honeypot)

    if pairwise_advantages and not is_honeypot:
        adv_str = "; ".join(pairwise_advantages[:3])
        assessment += f" | OUTRANKS nearby candidates due to: {adv_str}"
    
    # Add disqualifier note
    strong_penalty = cand.get("strong_disq_penalty", 1.0)
    if strong_penalty < 0.5:
        assessment += " | ROLE CONCERN: candidate role not aligned with retrieval/ranking engineering"

    recommendation = _recommendation(score, gate_passed, is_honeypot)

    parts = [header]
    parts.append(f"EVIDENCE: {evidence}")
    parts.append("STRENGTHS: " + ("; ".join(strengths) if strengths else "none identified"))
    if concerns:
        parts.append("CONCERNS: " + "; ".join(concerns))
    parts.append(f"AVAILABILITY: {avail}")
    parts.append(f"SCORES: [{score_str}]")
    parts.append(f"ASSESSMENT: {assessment}")
    parts.append(f"RECOMMENDATION: {recommendation}")

    return " | ".join(parts)