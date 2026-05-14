set -euo pipefail

if [[ ! -f .env ]]; then
  echo "ERROR: .env missing. Run:  cp .env.example .env  and fill in OPENROUTER_API_KEY" >&2
  exit 1
fi

export LLM_PROVIDER="${LLM_PROVIDER:-openrouter}"
export LLM_FALLBACK="${LLM_FALLBACK:-none}"
export VECTOR_STORE="${VECTOR_STORE:-chroma}"

python server.py
