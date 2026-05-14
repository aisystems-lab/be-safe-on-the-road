set -euo pipefail

if [[ ! -f .env ]]; then
  echo "ERROR: .env missing. Run:  cp .env.example .env  and fill in OPENROUTER_API_KEY" >&2
  exit 1
fi

if [[ $# -lt 1 ]]; then
  cat >&2 <<EOF
Usage: $0 <openrouter-model-id> [extra flags]

Example:
  $0 nvidia/nemotron-3-super-120b-a12b:free
  $0 google/gemma-3-27b-it:free --trials 3
  $0 minimax/minimax-m2.5:free --no-judge
EOF
  exit 1
fi

MODEL="$1"
shift

if [[ "$MODEL" != *:free ]] && [[ ! " $* " =~ " --allow-paid " ]]; then
  echo "ERROR: model id '$MODEL' does not end in ':free'." >&2
  echo "Use --allow-paid only if your prof has approved a budget." >&2
  exit 1
fi

TRIALS="${COMPARE_TRIALS:-1}"

echo "================================================"
echo " Running ONE model: $MODEL"
echo " Trials: $TRIALS"
echo " Mode:   append (existing rows preserved)"
echo "================================================"

python compare_models.py \
  --openrouter \
  --openrouter-models "$MODEL" \
  --trials "$TRIALS" \
  --append \
  "$@"
