import os

SEED = 42

ARXIV_QUERY = 'abs:"graph neural network" AND (abs:"drug discovery" OR abs:"molecular property" OR abs:"drug-drug interaction" OR abs:"molecule generation" OR abs:"virtual screening")'
ARXIV_CATEGORIES = ["cs.LG", "q-bio.QM", "q-bio.BM", "cs.AI"]
MAX_PAPERS = 800

RAW_DATA_PATH = "data/raw/papers.jsonl"
CHUNKS_PATH = "data/processed/chunks.jsonl"

CHUNK_SIZE_WORDS = 200
CHUNK_OVERLAP_WORDS = 40

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

GROQ_MODEL_NAME = "llama-3.3-70b-versatile"
GROQ_API_KEY_ENV = "GROQ_API_KEY"

TOP_K = 5

GOLD_QUESTIONS_PATH = "eval/gold_questions.json"
EVAL_RESULTS_DIR = "eval/results"
