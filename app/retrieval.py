from pathlib import Path
from typing import Any, Dict, List, Optional

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


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _matches_filters(doc: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
    if not filters:
        return True

    genres = _normalize_text(doc.get("genres", "")).lower()
    if "genres" in filters:
        wanted = filters["genres"]
        if isinstance(wanted, str):
            wanted = [wanted]
        for g in wanted:
            g_norm = _normalize_text(g).lower()
            if g_norm and g_norm not in genres:
                return False

    if "studio" in filters:
        studio = _normalize_text(doc.get("studio", "")).lower()
        wanted = _normalize_text(filters["studio"]).lower()
        if wanted and wanted not in studio:
            return False

    if "type" in filters:
        anime_type = _normalize_text(doc.get("type", "")).lower()
        wanted = _normalize_text(filters["type"]).lower()
        if wanted and wanted != anime_type:
            return False

    score = _to_float(doc.get("score"))
    min_score = _to_float(filters.get("min_score")) if "min_score" in filters else None
    max_score = _to_float(filters.get("max_score")) if "max_score" in filters else None
    if min_score is not None and (score is None or score < min_score):
        return False
    if max_score is not None and (score is None or score > max_score):
        return False

    episodes = _to_int(doc.get("episodes"))
    min_eps = _to_int(filters.get("min_episodes")) if "min_episodes" in filters else None
    max_eps = _to_int(filters.get("max_episodes")) if "max_episodes" in filters else None
    if min_eps is not None and (episodes is None or episodes < min_eps):
        return False
    if max_eps is not None and (episodes is None or episodes > max_eps):
        return False

    return True


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


def search(
    query: str,
    k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    candidate_factor: int = 4,
) -> List[Dict]:
    docs = _load_docs()
    index = _load_index()
    q_emb = embed_query(query)

    candidate_k = min(len(docs), max(k, k * candidate_factor))
    scores, idxs = index.search(q_emb, candidate_k)
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
        if not _matches_filters(flat, filters):
            continue
        results.append(flat)
        if len(results) >= k:
            break

    return results


if __name__ == "__main__":
    hits = search("high school romance with comedy", k=5)
    for h in hits:
        print("---")
        print("Title:", h.get("title"))
        print("Score:", h["retrieval_score"])
        print("Genres:", h.get("genres"))
        print(h["content"][:300], "...")
