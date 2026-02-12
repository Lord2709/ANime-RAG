from typing import Any, Dict, List, Optional, Tuple

import ollama

from generation import build_messages, PromptConfig
from retrieval import search


def run_rag(
    prompt: str,
    history: List[Dict[str, str]],
    config: PromptConfig,
    provider: str,
    model: str,
    max_docs: int,
    filters: Optional[Dict[str, Any]] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    min_confidence = 0.25
    try:
        docs = search(prompt, k=max_docs, filters=filters)
    except Exception as exc:
        return f"Retrieval error: {exc}", []

    top_score = docs[0].get("retrieval_score", 0.0) if docs else 0.0
    if not docs or top_score < min_confidence:
        return (
            "I couldn't find a confident match in the local data. "
            "Try rephrasing or adding genres, studio, or type.",
            [],
        )

    messages = build_messages(
        question=prompt,
        docs=docs,
        history=history,
        config=config,
    )

    if provider == "ollama":
        try:
            result = ollama.chat(model=model, messages=messages)
            response = result["message"]["content"]
        except Exception as exc:
            response = f"Model error: {exc}"
    else:
        response = "Stub response (LLM disabled)."

    return response, docs
