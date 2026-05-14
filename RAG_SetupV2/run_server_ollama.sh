set -euo pipefail

export LLM_PROVIDER="${LLM_PROVIDER:-ollama}"
export LLM_FALLBACK="${LLM_FALLBACK:-openai}"
export OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.1}"
export VECTOR_STORE="${VECTOR_STORE:-chroma}"

python server.py
