#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-$PROJECT_ROOT/.cache/sentence-transformers}"

exec uvicorn web_server:app --app-dir app --host 0.0.0.0 --port "${PORT:-8000}"
