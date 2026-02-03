from pathlib import Path
from typing import List, Dict

import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parents[1]

DOCS_PATH = BASE_DIR / "data" / "processed" / "documents.json"
FAISS_DIR = BASE_DIR / "embeddings" / "faiss_index"
INDEX_PATH = FAISS_DIR / "anime.index"

EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"

_model = None
_index = None
_docs = None


def _load_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def _load_docs():
    global _docs
    if _docs is None:
        with DOCS_PATH.open(encoding="utf-8") as f:
            _docs = json.load(f)
    return _docs


def _load_index() -> faiss.Index:
    global _index
    if _index is None:
        if not INDEX_PATH.exists():
            raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}")
        _index = faiss.read_index(str(INDEX_PATH))
    return _index


def embed_query(text: str) -> np.ndarray:
    model = _load_model()

    instruction = "Represent this sentence for searching relevant passages: "
    q = model.encode(
        [instruction + text],
        batch_size=1,
        normalize_embeddings=True,  # sentence-transformers can L2-normalize directly
    ).astype("float32")

    # If you did NOT use normalize_embeddings=True in ingestion, then:
    # faiss.normalize_L2(q)

    return q


def search(query: str, k: int = 5) -> List[Dict]:
    docs = _load_docs()
    index = _load_index()
    q_emb = embed_query(query)

    scores, idxs = index.search(q_emb, k)
    scores = scores[0]
    idxs = idxs[0]

    results: List[Dict] = []
    for score, i in zip(scores, idxs):
        if i < 0:
            continue
        doc = docs[int(i)]
        flat = {
            "retrieval_score": float(score),
            "content": doc["content"],
        }
        if isinstance(doc, dict) and "metadata" in doc and isinstance(doc["metadata"], dict):
            flat.update(doc["metadata"])
        results.append(flat)

    return results


if __name__ == "__main__":
    hits = search("high school romance with comedy", k=5)
    for h in hits:
        print("---")
        print("Title:", h.get("title"))
        print("Score:", h["retrieval_score"])
        print("Genres:", h.get("genres"))
        print(h["content"][:300], "...")
