# AI Resume Ranker

AI-powered candidate ranking system built for the Redrob Data & AI Challenge.

The system ranks candidates from structured profile datasets (JSON / JSONL) against a given job description and generates a submission-ready CSV containing the top 100 ranked candidates with reasoning.

---

## Overview

This project uses a hybrid ranking pipeline combining:

- Semantic similarity matching
- Feature engineering from candidate profiles
- Weighted scoring across multiple ranking signals
- Pairwise reranking for close candidates
- Explicit trap / honeypot detection

The goal is to identify genuinely relevant AI / ML candidates while reducing false positives caused by keyword stuffing, fake AI profiles, or irrelevant career backgrounds.

---

## Architecture

```text
Job Description
    │
    ├──► jd_intelligence.py
    │      Parses job description into structured features:
    │      - title
    │      - minimum experience
    │      - key skills
    │      - must-have requirements
    │      - red flags
    │
    │      Supports optional LLM-assisted parsing
    │      with offline keyword fallback.
    │
Candidates
    │
    ├──► feature_engineering.py
    │      Extracts candidate signals:
    │      - retrieval / ranking evidence
    │      - production ML evidence
    │      - company relevance
    │      - behavioral signals
    │      - disqualification penalties
    │
    ├──► embeddings.py
    │      Computes semantic similarity using:
    │      - sentence-transformers
    │      - TF-IDF fallback
    │
    ├──► ranker.py
    │
    │      Stage 1 Scoring
    │      • 25% semantic similarity
    │      • 25% production ML evidence
    │      • 30% retrieval/ranking evidence
    │      • 15% product-company relevance
    │      •  5% behavioral signals
    │      • + experience bonus
    │
    ├──► pairwise_ranker.py
    │      Stage 2 Pairwise Reranking
    │      Refines ordering among top candidates
    │
    ├──► reasoning.py
    │      Generates evidence-based reasoning
    │
    └──► submission.py
           Builds final submission CSV
```

---

## Ranking Strategy

### Stage 1 — Weighted Candidate Scoring

Each candidate receives a weighted score using:

- **Semantic Similarity**  
  Measures overall job-description relevance.

- **Production ML Evidence**  
  Rewards candidates with real deployment experience.

- **Retrieval / Ranking Evidence**  
  Prioritizes search, recommendation, ranking, and retrieval system experience.

- **Product Company Relevance**  
  Gives higher preference to product-focused engineering backgrounds.

- **Behavioral Signals**  
  Uses recruiter interaction and availability indicators.

---

### Stage 2 — Pairwise Ranking

Top candidates are compared pairwise across multiple dimensions to improve ordering within close score ranges.

This improves ranking precision in the top shortlist.

---

## Penalty System

Explicit penalties reduce scores for suspicious or irrelevant profiles:

- Honeypot candidates
- Keyword stuffing
- Consulting-only profiles
- Irrelevant roles
- Fake AI buzzword inflation
- Weak retrieval evidence

---

## Constraints Honoured

- CPU-only execution
- Candidate ranking runs fully offline
- Runtime designed for ≤ 5 minutes
- Designed for ≤ 16 GB RAM
- No external API required during ranking

### Offline Embedding Fallback

If sentence-transformer embeddings are unavailable in offline environments, the system automatically falls back to TF-IDF similarity without breaking the pipeline.

---

## Input Formats

Supported candidate / JD formats:

- TXT
- PDF
- DOCX
- JSON
- JSONL
- CSV
- XLSX

---

## Output Format

The system generates:

```csv
candidate_id,rank,score,reasoning
```

Exactly 100 ranked candidates are exported.

---

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python app.py
```

Open browser:

```text
http://localhost:5000
```

Upload:

- Job description file
- Candidate dataset

Click **Rank Candidates** to generate rankings.

---

## Project Modules

| File | Purpose |
|------|---------|
| `app.py` | Flask server and routing |
| `data_loader.py` | Loads job descriptions and candidate files |
| `jd_intelligence.py` | Job description parsing |
| `feature_engineering.py` | Candidate signal extraction |
| `embeddings.py` | Semantic similarity computation |
| `ranker.py` | Stage 1 weighted scoring |
| `pairwise_ranker.py` | Stage 2 reranking |
| `reasoning.py` | Reasoning generation |
| `submission.py` | Submission CSV generation |
| `templates/index.html` | Frontend interface |

---

## Example Reasoning

```text
Candidate Name | 6y exp | ML Engineer at Product Company
Strengths: strong retrieval systems, production ML deployment
Concerns: limited ranking evaluation evidence
Availability: open to work, 30d notice
Scores [semantic=0.72 career=0.81 prod=0.64 total=0.84]
```

---

## AI Tool Usage

AI tools were used for:

- architectural discussion
- debugging
- code review
- documentation support

AI tools were not used during final candidate ranking inference.