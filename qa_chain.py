import os
import ollama
from retriever import get_relevant_chunks

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")


def build_prompt(question: str, context_chunks: list[str]) -> str:
    """Build a RAG prompt from the question and retrieved context chunks."""
    context = "\n\n---\n\n".join(context_chunks)
    return f"""You are a helpful assistant that answers questions based only on the provided document context.
If the answer is not found in the context, say "I couldn't find that information in the document."
Do not make up answers or use outside knowledge.

Context from document:
{context}

Question: {question}

Answer:"""


def answer_question(question: str, collection_name: str) -> dict:
    """
    Full RAG pipeline:
    1. Retrieve relevant chunks from ChromaDB
    2. Build prompt with context
    3. Generate answer using Ollama
    Returns answer text and source chunks.
    """
    # Step 1 — Retrieve
    chunks = get_relevant_chunks(question, collection_name)

    if not chunks:
        return {
            "answer": "No relevant content found in the document.",
            "sources": [],
        }

    # Step 2 — Build prompt
    prompt = build_prompt(question, chunks)

    # Step 3 — Generate (streaming)
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )

    answer = response["message"]["content"]

    return {
        "answer": answer,
        "sources": chunks,
    }


def stream_answer(question: str, collection_name: str):
    """
    Generator that streams the answer token by token.
    Yields text chunks as they arrive from Ollama.
    """
    chunks = get_relevant_chunks(question, collection_name)

    if not chunks:
        yield "No relevant content found in the document."
        return

    prompt = build_prompt(question, chunks)

    stream = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    for chunk in stream:
        token = chunk["message"]["content"]
        if token:
            yield token


if __name__ == "__main__":
    collection = input("Enter collection name: ")
    while True:
        question = input("\nAsk a question (or 'quit'): ")
        if question.lower() == "quit":
            break
        print("\nAnswer: ", end="", flush=True)
        for token in stream_answer(question, collection):
            print(token, end="", flush=True)
        print()