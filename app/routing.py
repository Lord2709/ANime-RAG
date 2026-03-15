from typing import Any, Dict, List, Optional, Tuple

from generation import build_messages, PromptConfig
from providers import generate_chat_response
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

    history_for_prompt = list(history or [])
    if history_for_prompt:
        last_message = history_for_prompt[-1]
        last_role = str(last_message.get("role", "")).strip().lower()
        last_content = str(last_message.get("content", "")).strip()
        if last_role == "user" and last_content == str(prompt).strip():
            history_for_prompt = history_for_prompt[:-1]

    messages = build_messages(
        question=prompt,
        docs=docs,
        history=history_for_prompt,
        config=config,
    )

    try:
        response = generate_chat_response(provider=provider, model=model, messages=messages)
    except Exception as exc:
        response = f"Model error: {exc}"

    return response, docs
