import json
import os
import re
import sys

from rank_bm25 import BM25Okapi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def load_chunks(path=None):
    path = path or config.CHUNKS_PATH
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


class BM25Retriever:
    def __init__(self, chunks=None):
        self.chunks = chunks if chunks is not None else load_chunks()
        self.tokenized_corpus = [tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query, top_k=None):
        top_k = top_k or config.TOP_K
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in ranked_indices:
            results.append({
                "chunk": self.chunks[idx],
                "score": float(scores[idx]),
            })
        return results


if __name__ == "__main__":
    retriever = BM25Retriever()
    query = "graph neural networks for predicting drug-drug interactions"
    results = retriever.search(query, top_k=5)
    print(f"Query: {query}\n")
    for r in results:
        c = r["chunk"]
        print(f"[{r['score']:.3f}] {c['chunk_id']} — {c['title']}")
