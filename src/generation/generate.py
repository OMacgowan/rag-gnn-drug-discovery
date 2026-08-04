import os
import sys

from dotenv import load_dotenv
from groq import Groq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from generation.prompts import SYSTEM_PROMPT, build_user_prompt

load_dotenv()

_client = None


def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get(config.GROQ_API_KEY_ENV)
        if not api_key:
            raise RuntimeError(
                f"Set {config.GROQ_API_KEY_ENV} in your environment or a .env file."
            )
        _client = Groq(api_key=api_key)
    return _client


def generate_answer(question, retrieved_chunks, model=None, temperature=0.0):
    model = model or config.GROQ_MODEL_NAME
    client = get_client()
    user_prompt = build_user_prompt(question, retrieved_chunks)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        seed=config.SEED,
    )
    return response.choices[0].message.content


def generate_no_context_answer(question, model=None, temperature=0.0):
    """Used for the diagnostic test: ask the LLM directly, no retrieval."""
    model = model or config.GROQ_MODEL_NAME
    client = get_client()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Answer the question as accurately as you can from your own knowledge."},
            {"role": "user", "content": question},
        ],
        temperature=temperature,
        seed=config.SEED,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    from retrieval.bm25 import BM25Retriever

    retriever = BM25Retriever()
    question = "What method does HGNN-DDI use to predict drug-drug interactions?"
    retrieved = retriever.search(question, top_k=5)
    answer = generate_answer(question, retrieved)

    print(f"Question: {question}\n")
    print("Retrieved chunks:")
    for r in retrieved:
        print(f"  [{r['score']:.3f}] {r['chunk']['chunk_id']}")
    print(f"\nAnswer:\n{answer}")
