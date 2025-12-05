import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

DATA_PATH = "data/processed/ai_database_clean1.csv"
INDEX_PATH = "models/vector_store.faiss"
EMBEDDINGS_PATH = "models/embeddings.npy"

def build_faiss_index():
    df = pd.read_csv(DATA_PATH)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = df["Task Description"].astype(str).tolist()
    embeddings = model.encode(texts, convert_to_numpy=True)

    os.makedirs("models", exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)

    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    print(f"✅ FAISS index saved at {INDEX_PATH}")

if __name__ == "__main__":
    build_faiss_index()
