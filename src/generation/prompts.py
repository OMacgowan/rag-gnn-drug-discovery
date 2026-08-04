SYSTEM_PROMPT = """You are a research assistant answering questions about graph neural networks for drug discovery, using only the retrieved paper excerpts provided to you as context.

Rules:
1. Answer using only information present in the retrieved context below. Do not use outside knowledge.
2. Every claim in your answer must be followed by an inline citation to the chunk(s) that support it, in the form [chunk_id].
3. If the retrieved context does not contain enough information to answer the question, respond exactly with: "I don't know" and briefly state what is missing.
4. Be concise and factual."""


def build_user_prompt(question, retrieved_chunks):
    context_blocks = []
    for r in retrieved_chunks:
        c = r["chunk"]
        context_blocks.append(
            f"[{c['chunk_id']}] (Title: {c['title']}, Published: {c['published']})\n{c['text']}"
        )
    context = "\n\n".join(context_blocks)

    return f"""Retrieved context:

{context}

Question: {question}

Answer the question using only the retrieved context above, with inline [chunk_id] citations. If the context is insufficient, say "I don't know"."""
