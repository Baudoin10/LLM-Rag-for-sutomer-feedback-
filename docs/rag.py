import os
import glob

DOCS_DIR = "docs"


def load_and_chunk_documents(docs_dir=DOCS_DIR):
    """Read all .txt files and split them into line-level chunks."""
    chunks = []
    for filepath in glob.glob(os.path.join(docs_dir, "*.txt")):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        for line in text.split("\n"):
            line = line.strip()
            if line:
                chunks.append({"text": line, "source": os.path.basename(filepath)})
    return chunks


if __name__ == "__main__":
    chunks = load_and_chunk_documents()
    print(f"Loaded {len(chunks)} chunks:\n")
    for c in chunks:
        print(f"[{c['source']}] {c['text']}")