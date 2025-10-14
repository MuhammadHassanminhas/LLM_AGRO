# src/retriever.py
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

class CPIRetriever:
    def __init__(self, index_path="../Data/cpi_faiss.index", map_path="../Data/cpi_mapping.csv"):
        print("🔍 Loading CPI retriever...")
        self.index = faiss.read_index(index_path)
        self.mapping = pd.read_csv(map_path)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def search(self, query, k=5):
        q_emb = self.embedder.encode([query])
        q_emb = q_emb / np.linalg.norm(q_emb)
        scores, idx = self.index.search(q_emb.astype('float32'), k)
        results = [(self.mapping.iloc[i, 0], float(s)) for i, s in zip(idx[0], scores[0])]
        return results
