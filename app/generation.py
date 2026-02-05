from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

import utils


SYSTEM_PROMPT_TEXT = """You are an anime RAG assistant that answers using the provided context.
Use only the context for factual claims about titles, studios, episodes, scores, or synopses.
If the context is missing or insufficient, say you do not know and ask a short follow-up question.
When the user asks for recommendations, provide up to {max_recs} options and explain why each fits.
Keep the tone friendly and the output {style}."""

USER_PROMPT_TEXT = """Question:
{question}

Context:
{context}

Answer the question using the context above. If the context does not help, say so clearly."""

SYSTEM_PROMPT = SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT_TEXT)
USER_PROMPT = HumanMessagePromptTemplate.from_template(USER_PROMPT_TEXT)

CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        SYSTEM_PROMPT,
        MessagesPlaceholder(variable_name="history"),
        USER_PROMPT,
    ]
)


@dataclass(frozen=True)
class PromptConfig:
    style: str = "concise"
    max_recs: int = 5
    max_docs: int = 5
    max_context_chars: int = 3200
    max_synopsis_chars: int = 280
    include_scores: bool = True
    include_retrieval_score: bool = False
    max_history_turns: int = 6


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = " ".join(text.split())
    return text


def _is_unknown(text: str) -> bool:
    lowered = text.strip().lower()
    return lowered in {"", "0", "0.0", "unknown", "nan", "none"}


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = _normalize_text(value)
        if not _is_unknown(text):
            return text
    return ""


def _extract_field_from_content(content: str, field: str) -> str:
    if not content:
        return ""
    field_prefix = f"{field}:"
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(field_prefix):
            return line[len(field_prefix):].strip()
    return ""


def _extract_synopsis(content: str) -> str:
    if not content:
        return ""
    marker = "Synopsis:"
    idx = content.find(marker)
    if idx == -1:
        return ""
    return content[idx + len(marker):].strip()


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def filter_nsfw_docs(
    docs: Sequence[Dict[str, Any]],
    enabled: bool = utils.FILTER_NSFW,
    keywords: Sequence[str] = utils.NSFW_KEYWORDS,
) -> List[Dict[str, Any]]:
    if not enabled:
        return list(docs)

    lowered_keywords = [k.lower() for k in keywords]
    safe_docs: List[Dict[str, Any]] = []
    for doc in docs:
        genres = _normalize_text(doc.get("genres", ""))
        content = _normalize_text(doc.get("content", ""))
        haystack = f"{genres} {content}".lower()
        if any(k in haystack for k in lowered_keywords):
            continue
        safe_docs.append(doc)
    return safe_docs


def format_doc(
    doc: Dict[str, Any],
    include_scores: bool = True,
    include_retrieval_score: bool = False,
    max_synopsis_chars: int = 280,
) -> str:
    content = _normalize_text(doc.get("content", ""))

    title = _first_non_empty(doc.get("title"), _extract_field_from_content(content, "Title"))
    genres = _first_non_empty(doc.get("genres"), _extract_field_from_content(content, "Genres"))
    anime_type = _first_non_empty(doc.get("type"), _extract_field_from_content(content, "Type"))
    episodes = _first_non_empty(doc.get("episodes"), _extract_field_from_content(content, "Episodes"))
    studio = _first_non_empty(doc.get("studio"), _extract_field_from_content(content, "Studio"))
    synopsis = _first_non_empty(_extract_synopsis(content))
    synopsis = _truncate(synopsis, max_synopsis_chars)

    lines: List[str] = []
    if title:
        lines.append(f"Title: {title}")
    if genres:
        lines.append(f"Genres: {genres}")
    if anime_type:
        lines.append(f"Type: {anime_type}")
    if episodes:
        lines.append(f"Episodes: {episodes}")
    if studio:
        lines.append(f"Studio: {studio}")

    if include_scores:
        score = _first_non_empty(doc.get("score"))
        if score:
            lines.append(f"Score: {score}")

    if include_retrieval_score:
        retrieval_score = _first_non_empty(doc.get("retrieval_score"))
        if retrieval_score:
            lines.append(f"Retrieval score: {retrieval_score}")

    if synopsis:
        lines.append(f"Synopsis: {synopsis}")

    return "\n".join(lines)


def format_context(
    docs: Sequence[Dict[str, Any]],
    config: Optional[PromptConfig] = None,
) -> str:
    if config is None:
        config = PromptConfig()

    if not docs:
        return "No relevant context was retrieved."

    docs = filter_nsfw_docs(docs, enabled=utils.FILTER_NSFW, keywords=utils.NSFW_KEYWORDS)
    if not docs:
        return "No relevant context was retrieved."

    blocks: List[str] = []
    used_chars = 0
    for i, doc in enumerate(docs[: config.max_docs], start=1):
        block = format_doc(
            doc,
            include_scores=config.include_scores,
            include_retrieval_score=config.include_retrieval_score,
            max_synopsis_chars=config.max_synopsis_chars,
        )
        if not block:
            continue
        block = f"Document {i}\n{block}"
        if used_chars + len(block) > config.max_context_chars and blocks:
            break
        blocks.append(block)
        used_chars += len(block)

    return "\n\n".join(blocks) if blocks else "No relevant context was retrieved."


def trim_history(
    history: Optional[Sequence[Dict[str, Any]]],
    max_turns: int,
) -> List[Dict[str, str]]:
    if not history or max_turns <= 0:
        return []

    filtered: List[Dict[str, str]] = []
    for msg in history:
        role = str(msg.get("role", "")).strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = _normalize_text(msg.get("content", ""))
        if not content:
            continue
        filtered.append({"role": role, "content": content})

    max_messages = max_turns * 2
    return filtered[-max_messages:]


def _history_to_lc_messages(history: Sequence[Dict[str, str]]) -> List[BaseMessage]:
    messages: List[BaseMessage] = []
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def _lc_message_to_dict(message: BaseMessage) -> Dict[str, str]:
    role = getattr(message, "type", "")
    if role == "human":
        role = "user"
    elif role == "ai":
        role = "assistant"
    elif role == "system":
        role = "system"
    else:
        role = str(role) if role else "assistant"
    return {"role": role, "content": message.content}


def build_messages(
    question: str,
    docs: Sequence[Dict[str, Any]],
    history: Optional[Sequence[Dict[str, Any]]] = None,
    config: Optional[PromptConfig] = None,
) -> List[Dict[str, str]]:
    if config is None:
        config = PromptConfig()

    context = format_context(docs, config=config)
    trimmed_history = trim_history(history, config.max_history_turns)
    history_messages = _history_to_lc_messages(trimmed_history)

    lc_messages = CHAT_PROMPT.format_messages(
        max_recs=config.max_recs,
        style=config.style,
        question=_normalize_text(question),
        context=context,
        history=history_messages,
    )

    return [_lc_message_to_dict(m) for m in lc_messages]


def build_prompt(
    question: str,
    docs: Sequence[Dict[str, Any]],
    history: Optional[Sequence[Dict[str, Any]]] = None,
    config: Optional[PromptConfig] = None,
) -> str:
    messages = build_messages(question, docs, history=history, config=config)
    return "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
