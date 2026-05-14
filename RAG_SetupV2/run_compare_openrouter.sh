set -euo pipefail

if [[ ! -f .env ]]; then
  echo "ERROR: .env missing. Run:  cp .env.example .env  and fill in OPENROUTER_API_KEY" >&2
  exit 1
fi

TRIALS="${COMPARE_TRIALS:-1}"

python compare_models.py \
  --openrouter \
  --trials "$TRIALS" \
  "$@"
