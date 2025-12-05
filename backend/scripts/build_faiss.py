"""
Standalone script to (re)build FAISS index and embeddings.
Run with: python scripts/build_faiss.py
"""

from backend.utils.vectorizer import build_faiss_index

if __name__ == "__main__":
    build_faiss_index()
