from pathlib import Path
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

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
_bm25_state = None


def _load_sentence_transformer(model_name: str) -> SentenceTransformer:
    try:
        return SentenceTransformer(model_name)
    except Exception as exc:
        try:
            return SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            raise exc


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _tokenize(text: str) -> List[str]:
    cleaned = _normalize_text(text).lower()
    return re.findall(r"[a-z0-9]+", cleaned)


_STOPWORDS = {
    "a",
    "an",
    "and",
    "the",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "at",
    "by",
    "from",
    "or",
    "is",
    "are",
    "be",
    "about",
    "that",
    "this",
}



def _doc_text_for_bm25(doc: Dict[str, Any]) -> str:
    title = ""
    genres = ""
    if isinstance(doc, dict) and "metadata" in doc and isinstance(doc["metadata"], dict):
        meta = doc["metadata"]
        title = meta.get("title", "") or ""
        genres = meta.get("genres", "") or ""
    else:
        title = doc.get("title", "") or ""
        genres = doc.get("genres", "") or ""
    content = doc.get("content", "") or ""
    return f"{title} {genres} {content}"


def _build_bm25_state(docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    doc_freq: Dict[str, int] = {}
    doc_lens: List[int] = []

    for doc in docs:
        tokens = _tokenize(_doc_text_for_bm25(doc))
        doc_lens.append(len(tokens))
        seen = set(tokens)
        for tok in seen:
            doc_freq[tok] = doc_freq.get(tok, 0) + 1

    avgdl = (sum(doc_lens) / len(doc_lens)) if doc_lens else 0.0
    return {
        "doc_freq": doc_freq,
        "doc_count": len(docs),
        "avgdl": avgdl,
        "k1": 1.2,
        "b": 0.75,
    }


def _load_bm25_state() -> Dict[str, Any]:
    global _bm25_state
    if _bm25_state is None:
        _bm25_state = _build_bm25_state(_load_docs())
    return _bm25_state


def _bm25_score(query: str, doc: Dict[str, Any], state: Dict[str, Any]) -> float:
    q_tokens = [t for t in _tokenize(query) if t not in _STOPWORDS]
    if not q_tokens:
        return 0.0

    tokens = _tokenize(_doc_text_for_bm25(doc))
    if not tokens:
        return 0.0

    tf = Counter(tokens)
    doc_len = len(tokens)
    avgdl = state.get("avgdl", 0.0) or 0.0
    if avgdl <= 0.0:
        return 0.0

    N = max(1, int(state.get("doc_count", 0)))
    doc_freq = state.get("doc_freq", {})
    k1 = float(state.get("k1", 1.2))
    b = float(state.get("b", 0.75))

    score = 0.0
    for tok in q_tokens:
        df = doc_freq.get(tok, 0)
        if df == 0:
            continue
        idf = math.log(1.0 + (N - df + 0.5) / (df + 0.5))
        freq = tf.get(tok, 0)
        denom = freq + k1 * (1 - b + b * (doc_len / avgdl))
        score += idf * ((freq * (k1 + 1)) / max(1e-9, denom))

    return float(score)


def _combine_scores(vector_score: float, bm25_score: float) -> float:
    # Vector scores are cosine similarities ([-1, 1]). BM25 is unbounded.
    # Keep vector dominant; use BM25 as a tie-breaker.
    return float(vector_score) + (0.05 * float(bm25_score))


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
        _model = _load_sentence_transformer(EMBED_MODEL_NAME)
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
    rerank: bool = True,
    dedupe: bool = True,
) -> List[Dict]:
    docs = _load_docs()
    index = _load_index()
    q_emb = embed_query(query)

    candidate_k = min(len(docs), max(k, k * candidate_factor))
    scores, idxs = index.search(q_emb, candidate_k)
    scores = scores[0]
    idxs = idxs[0]

    bm25_state = _load_bm25_state() if rerank else {}
    candidates: List[Tuple[float, float, Dict[str, Any]]] = []
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
        bm25_score = _bm25_score(query, flat, bm25_state) if rerank else 0.0
        combined = _combine_scores(score, bm25_score) if rerank else float(score)
        candidates.append((combined, float(score), flat))

    if not candidates:
        return []

    # Dedupe by title (keeps best chunk per anime)
    if dedupe:
        by_title: Dict[str, Tuple[float, float, Dict[str, Any]]] = {}
        for combined, vec_score, flat in candidates:
            key = _normalize_text(flat.get("title", "")).lower()
            if not key:
                key = f"doc_{len(by_title)}"
            existing = by_title.get(key)
            if existing is None or combined > existing[0]:
                by_title[key] = (combined, vec_score, flat)
        candidates = list(by_title.values())

    # Sort by combined (or vector) score
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [c[2] for c in candidates[:k]]


if __name__ == "__main__":
    hits = search("high school romance with comedy", k=5)
    for h in hits:
        print("---")
        print("Title:", h.get("title"))
        print("Score:", h["retrieval_score"])
        print("Genres:", h.get("genres"))
        print(h["content"][:300], "...")
