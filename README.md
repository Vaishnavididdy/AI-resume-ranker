# AI Resume Ranker

AI-powered candidate ranking for the Data & AI Challenge.
Ranks candidates from any JSONL/JSON file against any job description and
exports the top 100 to a submission CSV.

## Architecture

```
JD file ──► jd_intelligence.py (Claude API, one-shot)
               └──► structured JSON: must_have, preferred,
                    red_flags, hidden_intent, key_skills, min_years

Candidates ──► feature_engineering.py
                 └──► per-candidate structured features:
                      ai_depth, retrieval, production, ranking_eval,
                      python, product_score, consulting_multiplier,
                      availability_score, years_experience

Both ──► embeddings.py (sentence-transformers all-MiniLM-L6-v2 / TF-IDF fallback)
           └──► cosine similarity per candidate

All signals ──► ranker.py (hybrid score)
                  └──► 0.35 semantic + 0.25 domain + 0.20 skill_overlap
                       + 0.20 experience_fit + 0.05 availability
                  └──► penalties: honeypot ×0.5, consulting ×0.5–1.0

──► reasoning.py (evidence-based, no hallucination)
──► submission.py → submission.csv (always 100 rows)
```

## Constraints honoured
- CPU only · no external/hosted LLM APIs at ranking time
- LLM (Claude Sonnet via API) used **once** for JD parsing only, before ranking
- Runtime ≤ 5 min · RAM ≤ 16 GB
- Output: exactly 100 ranked rows — `candidate_id, rank, score, reasoning`

## Run locally

```bash
pip install -r requirements.txt
python app.py
# open http://localhost:5000
```

Upload a JD file and a candidates JSONL/JSON file, click **Rank Candidates**,
then **Download CSV**.

## Modules

| File | Purpose |
|------|---------|
| `app.py` | Flask server + route wiring |
| `data_loader.py` | Load JD and candidates from JSON / JSONL / text / docx |
| `jd_intelligence.py` | LLM-based JD parser → structured features (with keyword fallback) |
| `feature_engineering.py` | Rich candidate features: domain scores, company type, availability |
| `embeddings.py` | Semantic similarity via sentence-transformers or TF-IDF fallback |
| `ranker.py` | Hybrid ranker combining all signals |
| `reasoning.py` | Evidence-based reasoning — no hallucination |
| `submission.py` | Build final submission.csv (always 100 rows) |
| `templates/index.html` | Minimal, clean frontend |

## Scoring formula

```
score = 0.35 × semantic_similarity    (embedding cosine, all-MiniLM-L6-v2)
      + 0.25 × domain_fit             (retrieval/prod/AI/eval/python weighted)
      + 0.20 × skill_overlap          (0.7 recall + 0.3 jaccard vs JD skills)
      + 0.20 × experience_fit         (sigmoid centred on JD min_years)
      + 0.05 × availability_score     (open-to-work, notice period, response rate)

score × 0.5   if honeypot / suspicious
score × 0.5–1.0  based on consulting vs product background
```

## Reasoning format

```
{Name} | {N}y exp | {most-recent-title} at {company}
| Strengths: retrieval/ranking systems, production ML, …
| Concerns: consulting-only background
| Availability: actively looking; 30d notice
| Scores [semantic=0.72 domain=0.65 skills=0.58 exp=0.80 total=0.6841]
```