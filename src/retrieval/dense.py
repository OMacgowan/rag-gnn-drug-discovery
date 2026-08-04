import json
import os
import sys

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

EMBEDDINGS_PATH = "data/processed/embeddings.npy"


def load_chunks(path=None):
    path = path or config.CHUNKS_PATH
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


class DenseRetriever:
    def __init__(self, chunks=None, model_name=None, rebuild=False):
        np.random.seed(config.SEED)
        self.chunks = chunks if chunks is not None else load_chunks()
        self.model_name = model_name or config.EMBEDDING_MODEL_NAME
        self.model = SentenceTransformer(self.model_name)

        if not rebuild and os.path.exists(EMBEDDINGS_PATH):
            self.embeddings = np.load(EMBEDDINGS_PATH)
            if self.embeddings.shape[0] != len(self.chunks):
                self.embeddings = self._build_embeddings()
        else:
            self.embeddings = self._build_embeddings()

    def _build_embeddings(self):
        texts = [c["text"] for c in self.chunks]
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
        np.save(EMBEDDINGS_PATH, embeddings)
        return embeddings

    def search(self, query, top_k=None):
        top_k = top_k or config.TOP_K
        query_embedding = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )
        sims = cosine_similarity(query_embedding, self.embeddings)[0]
        ranked_indices = np.argsort(-sims)[:top_k]
        results = []
        for idx in ranked_indices:
            results.append({
                "chunk": self.chunks[idx],
                "score": float(sims[idx]),
            })
        return results


if __name__ == "__main__":
    retriever = DenseRetriever()
    query = "graph neural networks for predicting drug-drug interactions"
    results = retriever.search(query, top_k=5)
    print(f"Query: {query}\n")
    for r in results:
        c = r["chunk"]
        print(f"[{r['score']:.3f}] {c['chunk_id']} — {c['title']}")
