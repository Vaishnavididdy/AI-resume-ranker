"""
Semantic similarity using sentence embeddings.
Falls back to TF-IDF if transformer model is unavailable.
"""
from __future__ import annotations
from typing import List
import numpy as np

# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------
_ST_AVAILABLE = False
_encode_fn = None

def _try_load_st():
    global _ST_AVAILABLE, _encode_fn
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _encode_fn = lambda texts: _model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        _ST_AVAILABLE = True
        print("[embeddings] Using sentence-transformers (all-MiniLM-L6-v2)")
    except Exception as e:
        print(f"[embeddings] sentence-transformers unavailable ({e}), falling back to TF-IDF")

_try_load_st()


def _tfidf_similarities(query: str, corpus: List[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vect = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), stop_words="english")
    all_texts = [query] + corpus
    matrix = vect.fit_transform(all_texts)
    sims = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    return sims


def compute_similarities(jd_text: str, candidate_texts: List[str]) -> np.ndarray:
    """Return cosine similarities between jd_text and each candidate text."""
    if not candidate_texts:
        return np.array([])

    if _ST_AVAILABLE and _encode_fn is not None:
        try:
            all_texts = [jd_text] + candidate_texts
            embeddings = _encode_fn(all_texts)
            jd_vec = embeddings[0:1]
            cand_vecs = embeddings[1:]
            # cosine similarity
            norms_jd = np.linalg.norm(jd_vec, axis=1, keepdims=True)
            norms_c = np.linalg.norm(cand_vecs, axis=1, keepdims=True)
            jd_norm = jd_vec / np.maximum(norms_jd, 1e-9)
            c_norm = cand_vecs / np.maximum(norms_c, 1e-9)
            sims = (jd_norm @ c_norm.T).flatten()
            return sims
        except Exception as e:
            print(f"[embeddings] ST encode failed ({e}), falling back to TF-IDF")

    return _tfidf_similarities(jd_text, candidate_texts)