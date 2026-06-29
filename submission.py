from __future__ import annotations
from typing import Any, Dict, List
import pandas as pd
from reasoning import build_reasoning

def build_submission_df(ranked: List[Dict], jd_features: Dict, target_count: int = 100) -> pd.DataFrame:
    rows = []
    for i, cand in enumerate(ranked[:target_count], start=1):
        cand_id = cand.get("candidate_id", "")
        if not cand_id:
            raw = cand.get("raw", {})
            cand_id = raw.get("candidate_id") or raw.get("id") or f"cand_{i:04d}"
        reasoning = build_reasoning(cand, jd_features)
        rows.append({
            "candidate_id": cand_id,
            "rank": i,
            "score": round(float(cand.get("score", 0.0)), 6),
            "reasoning": reasoning
        })
    while len(rows) < target_count:
        rows.append({
            "candidate_id": "",
            "rank": len(rows) + 1,
            "score": 0.0,
            "reasoning": "No candidate available."
        })
    return pd.DataFrame(rows)[["candidate_id", "rank", "score", "reasoning"]]

def write_submission_csv(df: pd.DataFrame, path: str) -> str:
    df.to_csv(path, index=False)
    return path