import json
import os
from typing import Dict, List, Sequence
from urllib import error, request
from urllib.parse import urlparse

import ollama

PROVIDERS = ("ollama", "nvidia")
OLLAMA_MODELS = [
    "llama3.1:8b",
    "mistral:7b-instruct-v0.3-q4_0",
    "qwen:7b",
    "deepseek-llm:7b-chat",
    "phi3:mini",
]
NVIDIA_MODELS = [
    "deepseek-ai/deepseek-v3_2",
    "moonshotai/kimi-k2-instruct",
    "microsoft/phi-4-mini-flash-reasoning",
]


def list_providers() -> List[str]:
    return list(PROVIDERS)


def get_default_provider() -> str:
    normalized = (os.getenv("LLM_PROVIDER", "ollama") or "ollama").strip().lower()
    return normalized if normalized in PROVIDERS else "ollama"


def get_default_ollama_model() -> str:
    return (os.getenv("OLLAMA_MODEL", "llama3.1:8b") or "llama3.1:8b").strip()


def get_default_nvidia_model() -> str:
    return (os.getenv("NVIDIA_MODEL", "qwen/qwen3.5-122b-a10b") or "qwen/qwen3.5-122b-a10b").strip()


def get_default_nvidia_base_url() -> str:
    return (
        os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
        or "https://integrate.api.nvidia.com/v1"
    ).strip()


def provider_suggestions() -> Dict[str, List[str]]:
    nvidia_default = get_default_nvidia_model()
    nvidia_models = [nvidia_default, *NVIDIA_MODELS]
    unique_nvidia_models: List[str] = []
    for model in nvidia_models:
        if model and model not in unique_nvidia_models:
            unique_nvidia_models.append(model)

    return {
        "ollama": list(OLLAMA_MODELS),
        "nvidia": unique_nvidia_models,
    }


def default_model_for(provider: str) -> str:
    normalized = (provider or get_default_provider()).strip().lower()
    if normalized == "nvidia":
        return get_default_nvidia_model()
    return get_default_ollama_model()


def provider_defaults() -> Dict[str, object]:
    suggestions = provider_suggestions()
    return {
        "default_provider": get_default_provider(),
        "default_models": {name: default_model_for(name) for name in list_providers()},
        "model_suggestions": suggestions,
        "nvidia_base_url": get_default_nvidia_base_url(),
    }


def generate_chat_response(
    provider: str,
    model: str,
    messages: Sequence[Dict[str, str]],
) -> str:
    normalized_provider = (provider or get_default_provider()).strip().lower()
    normalized_model = (model or default_model_for(normalized_provider)).strip()

    if normalized_provider == "ollama":
        return _chat_with_ollama(normalized_model, messages)
    if normalized_provider == "nvidia":
        try:
            return _chat_with_nvidia(normalized_model, messages)
        except RuntimeError as exc:
            if not _should_retry_nvidia_model(exc):
                raise

            last_exc: RuntimeError = exc
            for fallback_model in _nvidia_fallback_models(normalized_model):
                try:
                    return _chat_with_nvidia(fallback_model, messages)
                except RuntimeError as fallback_exc:
                    last_exc = fallback_exc
            raise last_exc
    raise ValueError(f"Unsupported provider: {provider}")


def _chat_with_ollama(model: str, messages: Sequence[Dict[str, str]]) -> str:
    result = ollama.chat(model=model, messages=list(messages))
    content = result.get("message", {}).get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Ollama returned an empty response.")
    return content


def _chat_with_nvidia(model: str, messages: Sequence[Dict[str, str]]) -> str:
    base_url = _normalized_nvidia_base_url(get_default_nvidia_base_url())
    api_key = (os.getenv("NVIDIA_API_KEY", "") or "").strip()
    if not api_key and not _is_local_base_url(base_url):
        raise RuntimeError(
            "NVIDIA_API_KEY is not set. Add it to your environment or point NVIDIA_BASE_URL "
            "to a local NIM endpoint."
        )

    payload = {
        "model": model,
        "messages": list(messages),
        "temperature": 0.25,
        "top_p": 0.8,
        "max_tokens": 800,
    }
    encoded = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    endpoint = f"{base_url}/chat/completions"
    req = request.Request(endpoint, data=encoded, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=90) as response:
            body = response.read()
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace").strip()
        message = details or exc.reason or "Unknown NVIDIA API error"
        raise RuntimeError(f"NVIDIA request failed ({exc.code}): {message}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"NVIDIA connection error: {exc.reason}") from exc

    data = json.loads(body.decode("utf-8"))
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("NVIDIA returned no completion choices.")

    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        content = "\n".join(part for part in text_parts if part)

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("NVIDIA returned an empty response.")
    return content


def _should_retry_nvidia_model(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    retry_markers = (
        "degraded function cannot be invoked",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
    )
    return any(marker in message for marker in retry_markers)


def _nvidia_fallback_models(active_model: str) -> List[str]:
    models = provider_suggestions().get("nvidia", [])
    return [model for model in models if model and model != active_model]


def _normalized_nvidia_base_url(base_url: str) -> str:
    cleaned = (base_url or get_default_nvidia_base_url()).strip().rstrip("/")
    if cleaned.endswith("/chat/completions"):
        cleaned = cleaned[: -len("/chat/completions")]
    if cleaned.endswith("/v1"):
        return cleaned
    return f"{cleaned}/v1"


def _is_local_base_url(base_url: str) -> bool:
    parsed = urlparse(base_url)
    return parsed.hostname in {"localhost", "127.0.0.1", "::1"}
