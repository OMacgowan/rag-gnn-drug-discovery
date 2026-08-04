# RAG over arXiv Graph Neural Networks for Drug Discovery Papers

Retrieval-Augmented Generation system built over a corpus of arXiv papers on
graph neural networks (GNNs) applied to drug discovery: molecular property
prediction, virtual screening, drug-drug interaction prediction, and molecule
generation.

Course project — see `report/report.md` for the full write-up.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the repo root with a free Groq API key:

```
GROQ_API_KEY=your_key_here
```

## Reproducing results

(To be finalized — will be a single script that runs corpus collection,
preprocessing, retrieval, generation, and evaluation end to end.)

```bash
python run_all.py
```

## Repo structure

- `src/collect_corpus.py` — pulls the corpus from the arXiv API
- `src/preprocess.py` — cleaning, chunking, metadata assignment
- `src/retrieval/` — BM25 and dense retrieval
- `src/generation/` — LLM prompting and answer generation
- `src/eval/` — retrieval and generation evaluation
- `data/raw/` — raw collected corpus
- `data/processed/` — cleaned/chunked corpus
- `eval/gold_questions.json` — hand-written gold evaluation set
- `eval/results/` — evaluation output
- `report/` — project report
