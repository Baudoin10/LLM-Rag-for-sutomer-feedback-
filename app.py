from flask import Flask, request, jsonify, render_template
from rag import load_and_chunk_documents, embed_chunks, retrieve, generate_answer

app = Flask(__name__)

print("Loading documents and building the knowledge base...")
chunks = load_and_chunk_documents()
chunks = embed_chunks(chunks)
print(f"Ready! Loaded {len(chunks)} chunks.")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    question = request.json.get("question", "")
    relevant_chunks = retrieve(question, chunks)
    answer = generate_answer(question, relevant_chunks)
    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=True)