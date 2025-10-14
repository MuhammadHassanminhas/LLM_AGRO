# src/embedding_index.py
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss, numpy as np
from config import OUTPUT_FILE

def build_index():
    print("🚀 Building embeddings and FAISS index ...")
    df = pd.read_parquet(OUTPUT_FILE)
    texts = df['text_record'].tolist()

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, "../Data/cpi_faiss.index")
    df[['text_record']].to_csv("../Data/cpi_mapping.csv", index=False)

    print("✅ Saved index (data/cpi_faiss.index) and mapping (data/cpi_mapping.csv)")

if __name__ == "__main__":
    build_index()
