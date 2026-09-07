import os
import csv
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq


DOCS_DIR = "docs"

embedder = SentenceTransformer("all-MiniLM-L6-v2")
groq_client = Groq()  # reads GROQ_API_KEY from environment


def load_and_chunk_documents(docs_dir=DOCS_DIR):
    """Read the feedback CSV and turn each row into a chunk."""
    chunks = []
    csv_path = os.path.join(docs_dir, "feedback.csv")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            feedback_text = row["feedbacks"]
            categories = row["categories"]
            chunk_text = f"Feedback: {feedback_text} | Categories: {categories}"
            chunks.append({"text": chunk_text, "source": "feedback.csv"})
    return chunks


def embed_chunks(chunks):
    """Add an 'embedding' field to each chunk using the local model."""
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb
    return chunks


def cosine_similarity(a, b):
    """Measure how similar two vectors are. 1 = identical meaning, 0 = unrelated."""
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve(question, chunks, top_k=5):
    """Find the top_k most relevant chunks for a given question."""
    question_embedding = embedder.encode(question)
    scored = []
    for chunk in chunks:
        score = cosine_similarity(question_embedding, chunk["embedding"])
        scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for score, chunk in scored[:top_k]]


def generate_answer(question, relevant_chunks):
    """Ask the LLM to answer the question using only the retrieved chunks as context."""
    context = "\n".join(f"- {c['text']}" for c in relevant_chunks)

    prompt = f"""Answer the question using ONLY the context below. If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def main():
    print("Loading documents and building the knowledge base...")
    chunks = load_and_chunk_documents()
    chunks = embed_chunks(chunks)
    print(f"Ready! Loaded {len(chunks)} chunks from your docs.\n")
    print("Ask a question (or type 'quit' to exit):\n")

    while True:
        question = input("You: ")
        if question.lower() in ("quit", "exit"):
            break

        relevant_chunks = retrieve(question, chunks)
        answer = generate_answer(question, relevant_chunks)

        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()