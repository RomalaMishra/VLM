"""
Loads the pre-built FAISS index and retrieves relevant guidance
given a list of detected object labels.
"""

from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


INDEX_PATH = Path(__file__).parent.parent / "data" / "faiss_index"

_vectorstore = None


def _load_store() -> FAISS:
    global _vectorstore
    if _vectorstore is None:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        _vectorstore = FAISS.load_local(
            str(INDEX_PATH),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    return _vectorstore


def retrieve(detected_objects: list[str], top_k: int = 3) -> list[str]:
    """
    Given a list of YOLO-detected object names, returns the top_k
    most relevant guidance paragraphs from the knowledge base.
    """
    if not detected_objects:
        return []

    query = "navigation safety guidance for: " + ", ".join(detected_objects)
    store = _load_store()
    results = store.similarity_search(query, k=top_k)
    return [doc.page_content for doc in results]


if __name__ == "__main__":
    # Quick smoke test
    guidance = retrieve(["chair", "dining table"])
    for i, g in enumerate(guidance, 1):
        print(f"[{i}] {g}\n")
