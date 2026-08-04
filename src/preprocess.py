import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def chunk_words(words, size, overlap):
    if len(words) <= size:
        return [words]
    chunks = []
    start = 0
    step = size - overlap
    while start < len(words):
        chunk = words[start:start + size]
        chunks.append(chunk)
        if start + size >= len(words):
            break
        start += step
    return chunks


def load_papers(path):
    papers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                papers.append(json.loads(line))
    return papers


def build_chunks(papers):
    all_chunks = []
    for paper in papers:
        title = clean_text(paper["title"])
        abstract = clean_text(paper["abstract"])
        full_text = f"{title}. {abstract}"

        words = full_text.split(" ")
        word_groups = chunk_words(words, config.CHUNK_SIZE_WORDS, config.CHUNK_OVERLAP_WORDS)

        for i, group in enumerate(word_groups):
            chunk_text = " ".join(group)
            chunk = {
                "chunk_id": f"{paper['doc_id']}_c{i}",
                "doc_id": paper["doc_id"],
                "chunk_index": i,
                "num_chunks": len(word_groups),
                "text": chunk_text,
                "title": paper["title"],
                "authors": paper["authors"],
                "published": paper["published"],
                "primary_category": paper["primary_category"],
                "categories": paper["categories"],
                "url": paper["url"],
            }
            all_chunks.append(chunk)

    return all_chunks


def preprocess():
    papers = load_papers(config.RAW_DATA_PATH)
    print(f"Loaded {len(papers)} papers")

    chunks = build_chunks(papers)
    print(f"Produced {len(chunks)} chunks")

    multi_chunk_docs = sum(1 for p in papers if len(clean_text(p["title"] + ". " + p["abstract"]).split(" ")) > config.CHUNK_SIZE_WORDS)
    print(f"{multi_chunk_docs} papers were split into multiple chunks")

    os.makedirs(os.path.dirname(config.CHUNKS_PATH), exist_ok=True)
    with open(config.CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Saved to {config.CHUNKS_PATH}")


if __name__ == "__main__":
    preprocess()
