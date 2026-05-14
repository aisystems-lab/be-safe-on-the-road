set -euo pipefail

export LLM_PROVIDER="${LLM_PROVIDER:-openai}"
export LLM_FALLBACK="${LLM_FALLBACK:-ollama}"
export OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o-mini}"
export VECTOR_STORE="${VECTOR_STORE:-chroma}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY not set" >&2
  exit 1
fi

python server.py
