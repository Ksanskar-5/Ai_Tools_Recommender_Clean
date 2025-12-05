"""
Vectorizer & semantic search utilities.
- Loads precomputed FAISS index and embeddings when present
- Falls back to in-memory embedding search using sentence-transformers
- Exposes: build_faiss_index(), load_faiss_index(), semantic_search()
"""

import os
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util

# Optional: import faiss only when needed (faiss may not be available on all environments)
try:
    import faiss
    _FAISS_AVAILABLE = True
except Exception:
    _FAISS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Centralized paths (change here if you move files)
BASE_DATA_PATH = os.environ.get("AI_RECOMMENDER_DATA", "data")
DATA_PATH = os.path.join(BASE_DATA_PATH, "processed", "ai_database_clean1.csv")
INDEX_DIR = os.path.join("models")
INDEX_PATH = os.path.join(INDEX_DIR, "vector_store.faiss")
EMBEDDINGS_PATH = os.path.join(INDEX_DIR, "embeddings.npy")

# Load dataframe once (if module imported)
_df_cache: pd.DataFrame = None
_model: SentenceTransformer = None
_corpus_embeddings = None

def load_corpus_dataframe() -> pd.DataFrame:
    global _df_cache
    if _df_cache is None:
        if not os.path.exists(DATA_PATH):
            # try fallback location used in other scripts
            alt = os.path.join(BASE_DATA_PATH, "ai_database.csv")
            if os.path.exists(alt):
                df = pd.read_csv(alt)
            else:
                raise FileNotFoundError(f"AI database not found at {DATA_PATH} or {alt}")
        else:
            df = pd.read_csv(DATA_PATH)
        _df_cache = df.fillna("")
    return _df_cache
def _ensure_models_exist():
    """
    If FAISS index or embeddings are missing, build and save them.
    This runs once on the first request.
    """
    missing = []
    if not os.path.exists(INDEX_PATH):
        missing.append("FAISS index")
    if not os.path.exists(EMBEDDINGS_PATH):
        missing.append("embeddings.npy")
    if missing:
        logger.info(f"Missing {', '.join(missing)}; building now...")
        build_faiss_index()
def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        # small model for speed, change as needed
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def build_faiss_index():
    """
    Build embeddings and a FAISS index and save to disk.
    """
    df = load_corpus_dataframe()
    model = get_embedding_model()

    texts = df["Task Description"].astype(str).tolist()
    embeddings = model.encode(texts, convert_to_numpy=True)

    os.makedirs(INDEX_DIR, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)

    if not _FAISS_AVAILABLE:
        logger.warning("FAISS not available; skipping index creation")
        return

    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)
    logger.info(f"✅ FAISS index saved at {INDEX_PATH}")

def _load_faiss_index():
    """
    Return a faiss index object or None
    """
    if not _FAISS_AVAILABLE:
        return None
    if os.path.exists(INDEX_PATH):
        return faiss.read_index(INDEX_PATH)
    return None

def _ensure_corpus_embeddings():
    """
    Ensure corpus embeddings are loaded (either from .npy or recompute)
    """
    global _corpus_embeddings
    if _corpus_embeddings is not None:
        return _corpus_embeddings

    df = load_corpus_dataframe()
    model = get_embedding_model()

    if os.path.exists(EMBEDDINGS_PATH):
        try:
            arr = np.load(EMBEDDINGS_PATH)
            # convert to tensor for sentence-transformers util
            _corpus_embeddings = util.to_tensor(arr)
            return _corpus_embeddings
        except Exception:
            logger.exception("Failed to load embeddings.npy; recomputing")

    # fallback: compute embeddings in memory
    texts = df["Task Description"].astype(str).tolist()
    emb = model.encode(texts, convert_to_tensor=True)
    _corpus_embeddings = emb
    return _corpus_embeddings

def semantic_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Semantic search entrypoint. Attempts FAISS nearest neighbor search first (if available),
    otherwise uses sentence-transformers' util.semantic_search with in-memory embeddings.

    Returns list of result dicts with keys matching the dataframe columns plus 'score'.
    """
    ensure_models_exist()
    df = load_corpus_dataframe()
    model = get_embedding_model()

    # Try FAISS for speed if available
    index = _load_faiss_index()
    if index is not None:
        try:
            q_emb = model.encode(query, convert_to_numpy=True).astype("float32")
            # faiss expects shape (1, d)
            D, I = index.search(np.array([q_emb]), top_k)
            results = []
            for score, idx in zip(D[0], I[0]):
                if idx < 0 or idx >= len(df):
                    continue
                row = df.iloc[int(idx)]
                results.append({**row.to_dict(), "score": float(score)})
            return results
        except Exception:
            logger.exception("FAISS search failed; falling back to in-memory semantic search")

    # In-memory semantic search
    corpus_emb = _ensure_corpus_embeddings()
    q_emb = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(q_emb, corpus_emb, top_k=top_k)[0]

    results = []
    for hit in hits:
        row = df.iloc[int(hit["corpus_id"])]
        results.append({**row.to_dict(), "score": float(hit["score"])})
    return results

 