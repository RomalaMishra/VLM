"""
Run once to build the FAISS index from accessibility_knowledge.md.
Output: data/faiss_index/ (index file + docstore)
"""

import re
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


KNOWLEDGE_PATH = Path(__file__).parent.parent / "data" / "accessibility_knowledge.md"
INDEX_PATH = Path(__file__).parent.parent / "data" / "faiss_index"


def load_chunks(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8")
    raw_chunks = re.split(r"\n## ", text)

    docs = []
    for chunk in raw_chunks:
        chunk = chunk.strip()
        if not chunk or chunk.startswith("#"):
            continue
        lines = chunk.split("\n", 1)
        label = lines[0].strip().lower()
        content = lines[1].strip() if len(lines) > 1 else ""
        if content:
            docs.append(Document(page_content=content, metadata={"object": label}))

    return docs


def build_index():
    print("Loading knowledge chunks...")
    docs = load_chunks(KNOWLEDGE_PATH)
    print(f"  {len(docs)} chunks loaded")

    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    INDEX_PATH.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_PATH))
    print(f"Index saved to {INDEX_PATH}")


if __name__ == "__main__":
    build_index()
